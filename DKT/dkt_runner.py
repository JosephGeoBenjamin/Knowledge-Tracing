import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm
from algorithms.deep_kt import DeepKnowledgeTracing

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
batch_size = 32
acc_grad = 1
learning_rate = 0.1

# train_file = merge_xlit_jsons(["data/hindi/HiEn_train1.json",
#                                 "data/hindi/HiEn_train2.json" ],
#                                 save_prefix= LOG_PATH)

train_dataset = XlitData( src_glyph_obj = src_glyph, tgt_glyph_obj = tgt_glyph,
                        json_file='data/hindi/HiEn_xlit_train.json', file_map = "LangEn",
                        padding=True)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)

val_dataset = XlitData( src_glyph_obj = src_glyph, tgt_glyph_obj = tgt_glyph,
                        json_file='data/hindi/HiEn_xlit_valid.json', file_map = "LangEn",
                        padding=True)


val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)

# for i in range(len(train_dataset)):
#     print(train_dataset.__getitem__(i))

##===== Model Configuration =================================================

rnn_type = "lstm"
enc_layers = 1
m_dropout = 0

model = Encoder(  input_dim= input_dim, embed_dim = enc_emb_dim,
                hidden_dim= enc_hidden_dim,
                rnn_type = rnn_type, layers= enc_layers,
                dropout= m_dropout, device = device,
                bidirectional= enc_bidirect)
model = model.to(device)

# model = load_pretrained(model,pretrain_wgt_path) #if path empty returns unmodified

## ----- Load Embeds -----


##------ Model Details ---------------------------------------------------------
rutl.count_train_param(model)
print(model)


##====== Optimizer Zone ===================================================================


criterion = torch.nn.CrossEntropyLoss()
    # weight = torch.from_numpy(train_dataset.tgt_class_weights).to(device)  )

def loss_estimator(pred, truth):
    """ Only consider non-zero inputs in the loss; mask needed
    pred: batch
    """
    mask = truth.ge(1).type(torch.FloatTensor).to(device)
    loss_ = criterion(pred, truth) * mask
    return torch.mean(loss_)

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
        if epoch >= teach_force_till: teacher_forcing = 0
        else: teacher_forcing = max(0, teacher_forcing - teach_decay_pereph)

        for ith, (src, tgt, src_sz) in enumerate(train_dataloader):

            src = src.to(device)
            tgt = tgt.to(device)

            #--- forward ------
            output = model(src = src, tgt = tgt, src_sz =src_sz,
                            teacher_forcing_ratio = teacher_forcing)
            loss = loss_estimator(output, tgt) / acc_grad
            acc_loss += loss

            #--- backward ------
            loss.backward()
            if ( (ith+1) % acc_grad == 0):
                optimizer.step()
                optimizer.zero_grad()

                print('epoch[{}/{}], MiniBatch-{} loss:{:.4f}'
                    .format(epoch+1, num_epochs, (ith+1)//acc_grad, acc_loss.data))
                running_loss.append(acc_loss.item())
                acc_loss=0
                # break

        LOG2CSV(running_loss, LOG_PATH+"trainLoss.csv")

        #--------- Validate ---------------------
        model.eval()
        val_loss = 0
        val_accuracy = 0
        for jth, (v_src, v_tgt, v_src_sz) in enumerate(tqdm(val_dataloader)):
            v_src = v_src.to(device)
            v_tgt = v_tgt.to(device)
            with torch.no_grad():
                v_output = model(src = v_src, tgt = v_tgt, src_sz = v_src_sz)
                val_loss += loss_estimator(v_output, v_tgt)

                val_accuracy += rutl.accuracy_score(v_output, v_tgt, tgt_glyph)
            #break
        val_loss = val_loss / len(val_dataloader)
        val_accuracy = val_accuracy / len(val_dataloader)

        print('epoch[{}/{}], [-----TEST------] loss:{:.4f}  Accur:{:.4f}'
              .format(epoch+1, num_epochs, val_loss.data, val_accuracy.data))
        LOG2CSV([val_loss.item(), val_accuracy.item()],
                    LOG_PATH+"valLoss.csv")

        #-------- save Checkpoint -------------------
        if val_accuracy > best_accuracy:
        # if val_loss < best_loss:
            print("***saving best optimal state [Loss:{} Accur:{}] ***".format(val_loss.data,val_accuracy.data) )
            best_loss = val_loss
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), WGT_PREFIX+"_model-{}.pth".format(epoch+1))
            LOG2CSV([epoch+1, val_loss.item(), val_accuracy.item()],
                    LOG_PATH+"bestCheckpoint.csv")

        # LR step
        # scheduler.step()