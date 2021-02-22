import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from DKT.data_utils import AssistDataset
from DKT.dkt_arch import DKT_Embednet, DKT_Onehotnet
from DKT.metric_utils import AccuracyTeller
import utilities.running_utils as rutl


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

##===== Init Setup =============================================================
INST_NAME = "Training_Test"

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


train_dataset = AssistDataset(  json_path='data/assist_splits/assist_train.json',
                                skill_file='data/assist_splits/skill_train.txt',
                                student_file='data/assist_splits/student_train.txt',
                                )

train_sampler, valid_sampler = rutl.random_train_valid_samplers(train_dataset)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                sampler=train_sampler,
                                num_workers=0)
valid_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                sampler=valid_sampler,
                                num_workers=0)

# for i in range(20):
#     print(train_dataset.__getitem__(i))

# for i, batch in enumerate(train_dataloader):
#         print(i, batch)
#         break

##===== Model Configuration =================================================

rnn_type = "lstm"
enc_layers = 1
m_dropout = 0

# model = DKT_Embednet(stud_count = len(train_dataset.idxr.stud2idx),
#                     stud_embed_dim = 16,
#                     skill_count = len(train_dataset.idxr.skill2idx),
#                     skill_embed_dim = 8,
#                     hidden_dim = 64, layers = 1,
#                     dropout = 0, device = device)

model = DKT_Onehotnet(stud_count = len(train_dataset.idxr.stud2idx),
                    skill_count = len(train_dataset.idxr.skill2idx),
                    hidden_dim = 64, layers = 1,
                    dropout = 0, device = device)
model = model.to(device)

# model = load_pretrained(model,pretrain_wgt_path) #if path empty returns unmodified


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

accuracy_estimator = AccuracyTeller()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                             weight_decay=0)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#===============================================================================

if __name__ =="__main__":

    best_loss = float("inf")
    best_accuracy = 0
    for epoch in range(num_epochs):

        #-------- Training -------------------
        model.train()
        acc_loss = 0
        running_loss = []
        for ith, (src1, src2, src_sz, tgt) in enumerate(train_dataloader):

            src1 = src1.to(device)
            src2 = src2.to(device)
            tgt = tgt.to(device)

            #--- forward ------
            output = model(x1 = src1, x2=src2, x_sz =src_sz)
            loss = loss_estimator(output, tgt) / acc_grad
            acc_loss += loss

            #--- backward ------
            loss.backward()
            if ( (ith+1) % acc_grad == 0):
                optimizer.step()
                optimizer.zero_grad()

                print('epoch[{}/{}], MiniBatch:{} loss:{:.4f}'
                    .format(epoch+1, num_epochs, (ith+1)//acc_grad, acc_loss.data))
                running_loss.append(acc_loss.item())
                acc_loss=0
                # break

        rutl.LOG2CSV(running_loss, LOG_PATH+"trainLoss.csv")

        #--------- Validate ---------------------
        model.eval()
        val_loss = 0
        val_auc = 0
        pred_labels = []; true_labels = []
        for jth, (vsrc1, vsrc2, vsrc_sz, vtgt) in enumerate(tqdm(valid_dataloader)):

            vsrc1 = vsrc1.to(device)
            vsrc2 = vsrc2.to(device)
            vtgt = vtgt.to(device)

            with torch.no_grad():
                voutput = model(x1 = vsrc1, x2= vsrc2, x_sz = vsrc_sz )
                val_loss += loss_estimator(voutput, vtgt)
                accuracy_estimator.register_result(vtgt, voutput)
            # break

        val_loss = val_loss / len(valid_dataloader)
        val_auc = accuracy_estimator.area_under_curve()
        val_acc = accuracy_estimator.accuracy_score()

        print('epoch[{}/{}], [-----TEST------] loss:{:.4f} AUC:{:.4f} Accur:{:.4f}'
              .format(epoch+1, num_epochs, val_loss.data, val_auc, val_acc ))

        rutl.LOG2CSV([val_loss.item(), val_auc, val_acc],
                    LOG_PATH+"valLoss.csv")

        #-------- save Checkpoint -------------------
        if val_auc > best_accuracy:
        # if val_loss < best_loss:
            print("***saving best optimal state [Loss:{} Accur:{}] ***".format(val_loss.data,val_auc) )
            best_loss = val_loss
            best_accuracy = val_auc
            torch.save(model.state_dict(), WGT_PREFIX+"_model.pth")
            rutl.LOG2CSV([epoch+1, val_loss.item(), val_auc],
                    LOG_PATH+"bestCheckpoint.csv")

        # LR step
        # scheduler.step()