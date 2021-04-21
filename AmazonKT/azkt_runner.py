import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from AmazonKT.data_utils import AssistDataset
from AmazonKT.azkt_arch import DKT_Embednet, DKT_Onehotnet
from AmazonKT.metric_utils import AccuracyTeller
import utilities.running_utils as rutl


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

##===== Init Setup =============================================================
INST_NAME = "Training_AZ_test"

##------------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

LOG_PATH = "hypotheses/"+INST_NAME+"/"
WGT_PREFIX = LOG_PATH+"weights/"+INST_NAME
if not os.path.exists(LOG_PATH+"weights"): os.makedirs(LOG_PATH+"weights")

##===== Running Configuration =================================================

num_epochs = 200
batch_size = 4
acc_grad = 1
learning_rate = 0.01

### ASSISTMENT DATASETS
skill_file='data/assist_splits/skill.txt'
student_file='data/assist_splits/student.txt'
question_file='data/assist_splits/question.txt'

train_dataset = AssistDataset(  json_path='data/assist_splits/assist_train.json',
                                skill_file=skill_file,
                                student_file=student_file,
                                question_file=question_file
                                )

train_sampler, valid_sampler = rutl.random_train_valid_samplers(train_dataset)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                sampler=train_sampler,
                                num_workers=0)
valid_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                sampler=valid_sampler,
                                num_workers=0)


## The IDs not present in train will be set as ZERO all below cases; a harsh testing
skill_iso_dataset = AssistDataset(  json_path='data/assist_splits/skill_isolate_test.json',
                    skill_file=skill_file, student_file=student_file, question_file=question_file  )
skill_iso_dataloader = DataLoader(skill_iso_dataset, batch_size=batch_size, num_workers=0)

student_iso_dataset = AssistDataset(  json_path='data/assist_splits/student_isolate_test.json',
                    skill_file=skill_file, student_file=student_file, question_file=question_file  )
student_iso_dataloader = DataLoader(student_iso_dataset, batch_size=batch_size, num_workers=0)

st_sk_iso_dataset = AssistDataset(  json_path='data/assist_splits/stud+skill_isolate_test.json',
                    skill_file=skill_file, student_file=student_file, question_file=question_file  )
st_sk_iso_dataloader = DataLoader(st_sk_iso_dataset, batch_size=batch_size, num_workers=0)


# for i in range(1):
#     print(train_dataset.__getitem__(i))

# for i, batch in enumerate(train_dataloader):
#         print(i, batch)
#         break

##===== Model Configuration =================================================

rnn_type = "lstm"
enc_layers = 1
m_dropout = 0

model = DKT_Embednet(stud_count = len(train_dataset.idxr.stud2idx),
                    stud_embed_dim = 16,
                    skill_count = len(train_dataset.idxr.skill2idx),
                    skill_embed_dim = 32,
                    ques_count = len(train_dataset.idxr.ques2idx),
                    ques_embed_dim = 32,
                    hidden_dim = 64, layers = 1,
                    dropout = 0, device = device)


model = model.to(device)

# model = load_pretrained(model,pretrain_wgt_path) #if path empty returns unmodified


def set_avg_embedding(model):
    ''' Average will be set at 0th index; consistent with dataset
    '''
    e = model.stud_embed
    e.weight.data[0].copy_(torch.mean(e.weight.data[1:], dim=0))
    e = model.skill_embed
    e.weight.data[0].copy_(torch.mean(e.weight.data[1:], dim=0))
    e = model.ques_embed
    e.weight.data[0].copy_(torch.mean(e.weight.data[1:], dim=0))

    return model


##------ Model Details ---------------------------------------------------------

rutl.print_model_arch(model)


##====== Optimizer Zone ===================================================================


criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
def loss_estimator(pred, truth):
    """ Only consider non-zero inputs in the loss; mask needed
    pred: batch
    """
    truth = truth.type(torch.FloatTensor).to(device)
    mask = truth.ge(0).type(torch.bool).to(device)
    loss_ = criterion(pred, truth)[mask]

    return torch.mean(loss_)

accuracy_estimator = AccuracyTeller() # must be reset for each epoch

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                             weight_decay=0)


#===============================================================================


def train_routine(tdataloader, model, set_name = ""):
    """ Return: Trained Model
    """
    model.train()
    acc_loss = 0
    running_loss = []
    for ith, (src1, src2, src3, src_sz, tgt) in enumerate(tdataloader):

        src1 = src1.to(device); src2 = src2.to(device)
        src3 = src3.to(device)
        tgt = tgt.to(device)

        #--- forward ------
        output = model( x1 = src1, x2=src2,
                        x3=src3, x_sz =src_sz)
        loss = loss_estimator(output, tgt) / acc_grad
        acc_loss += loss

        #--- backward ------
        loss.backward()
        if ( (ith+1) % acc_grad == 0):
            optimizer.step()
            optimizer.zero_grad()

            print('epoch[{}/{}], MiniBatch:{} loss:{:.4f}; {}'
                .format(epoch+1, num_epochs, (ith+1)//acc_grad, acc_loss.data, set_name))
            running_loss.append(acc_loss.item())
            acc_loss=0
            # break

    rutl.LOG2CSV(running_loss, LOG_PATH+"trainLoss{}.csv".format(set_name) )

    return model


def validation_routine(vdataloader, model, set_name = ""):
    """Return: Accuracy Metrics
    """
    model = set_avg_embedding(model)
    model.eval()
    accuracy_estimator.reset()

    val_loss = 0
    val_auc = 0
    pred_labels = []; true_labels = []
    for jth, (vsrc1, vsrc2, vsrc3, vsrc_sz, vtgt) in enumerate(tqdm(vdataloader)):

        vsrc1 = vsrc1.to(device); vsrc2 = vsrc2.to(device)
        vsrc3 = vsrc3.to(device)
        vtgt = vtgt.to(device)

        with torch.no_grad():
            voutput = model(x1 = vsrc1, x2= vsrc2,
                            x3=vsrc3, x_sz = vsrc_sz )
            val_loss += loss_estimator(voutput, vtgt)
            accuracy_estimator.register_result(vtgt, voutput)
        # break

    val_loss = val_loss / len(valid_dataloader)
    val_auc = accuracy_estimator.area_under_curve()
    val_acc = accuracy_estimator.accuracy_score()

    print(set_name)
    print('epoch[{}/{}], [-----TEST------] loss:{:.4f} AUC:{:.4f} Accur:{:.4f}'
            .format(epoch+1, num_epochs, val_loss.data, val_auc, val_acc ))

    rutl.LOG2CSV([val_loss.item(), val_auc, val_acc],
                LOG_PATH+"ValMeasures-{}.csv".format(set_name))

    return val_loss, val_auc, val_acc



if __name__ =="__main__":

    best_loss = float("inf")
    best_accuracy = 0
    for epoch in range(num_epochs):

        #-------- Training -------------------
        model =  train_routine(train_dataloader, model)

        #--------- Validate ---------------------

        ## ASSISTMENT
        val_loss, val_auc, val_acc = validation_routine(valid_dataloader, model, 'ValidSplit')
        validation_routine(skill_iso_dataloader, model, 'SkillIsolate')
        validation_routine(student_iso_dataloader, model, 'StudentIsolate')
        validation_routine(st_sk_iso_dataloader, model, 'SkStIsolate')


        #-------- save Checkpoint -------------------
        if val_auc > best_accuracy:
        # if val_loss < best_loss:
            print("***saving best optimal state [Loss:{} Accur:{}] ***".format(val_loss.data,val_auc) )
            best_loss = val_loss
            best_accuracy = val_auc
            torch.save(model.state_dict(), WGT_PREFIX+"_model-{}.pth".format(epoch+1))
            rutl.LOG2CSV([epoch+1, val_loss.item(), val_auc],
                    LOG_PATH+"bestCheckpoint.csv")
