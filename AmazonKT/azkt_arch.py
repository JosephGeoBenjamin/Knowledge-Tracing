import torch
import torch.nn as nn
import random

class XFMR_Embednet(nn.Module):
    '''
    Simple RNN based DKT model
    '''
    def __init__(self,  stud_count, stud_embed_dim,
                        skill_count, skill_embed_dim,
                        ques_count, ques_embed_dim,
                        hidden_dim , layers = 1,
                        dropout = 0, device = "cpu"):
        super(DKT_Embednet, self).__init__()

        self.enc_hidden_dim = hidden_dim
        self.enc_layers = layers
        self.device = device
        self.stud_count, self.stud_embed_dim = stud_count, stud_embed_dim
        self.skill_count, self.skill_embed_dim = skill_count, skill_embed_dim
        self.ques_count, self.ques_embed_dim = ques_count, ques_embed_dim

        self.stud_embed = nn.Embedding(self.stud_count, self.stud_embed_dim, )
        self.skill_embed = nn.Embedding(self.skill_count, self.skill_embed_dim, )
        self.ques_embed = nn.Embedding(self.ques_count, self.ques_embed_dim, )

        inp_sz = self.stud_embed_dim+self.skill_embed_dim+self.ques_embed_dim
        self.enc_rnn = nn.LSTM(
                    input_size= inp_sz,
                    hidden_size= self.enc_hidden_dim,
                    num_layers= self.enc_layers,)
        self.linear = nn.Linear(self.enc_hidden_dim, 1)

    def forward(self, x1, x2, x3, x_sz, hidden = None ):
        '''
        x_sz: (batch_size, 1) -  Unpadded sequence lengths used for pack_pad
        Return:
            output: (batch_size, max_length, hidden_dim)
            hidden: (n_layer*num_directions, batch_size, hidden_dim) | if LSTM tuple -(h_n, c_n)
        '''

        batch_sz = x1.shape[0]
        # replace all -1 values with 0 (unused embeddings)
        x1[x1 == -1] = 0; x2[x2 == -1] = 0;  x3[x3 == -1] = 0

        # x: batch_size, max_length, enc_embed_dim
        stemb = self.stud_embed(x1)
        skemb = self.skill_embed(x2)
        qsemb = self.ques_embed(x3)
        x = torch.cat([stemb, skemb, qsemb], dim = 2)

        ## pack the padded data
        # x: max_length, batch_size, enc_embed_dim -> for pack_pad
        x = x.permute(1,0,2)
        x = nn.utils.rnn.pack_padded_sequence(x, x_sz, enforce_sorted=False) # unpad

        # output: packed_size, batch_size, hidden --> hidden from all timesteps
        # hidden: n_layer**num_directions, batch_size, hidden_dim | if LSTM (h_n, c_n)
        rout, hidden = self.enc_rnn(x)

        ## pad the sequence to the max length in the batch
        # output: max_length, batch_size, hidden*directions
        rout, _ = nn.utils.rnn.pad_packed_sequence(rout, total_length=1000)

        # output: batch_size, max_length, hidden_dim
        rout = rout.permute(1,0,2)

        output = self.linear(rout)

        return output.squeeze(2)

class PositionalEncoding(nn.Module):

    def __init__(self, vector_dim, dropout=0, max_seq_len=100, device = "cpu"):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #pe :shp: (max_seq_len, vector_dim)
        self.pe = torch.zeros(max_seq_len, vector_dim).to(device)
        position = torch.arange(0,max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, vector_dim, 2).float() * (-math.log(10000.0) / vector_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        #pe :shp: max_seq_len, 1 ,vector_dim
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):

        # x :shp: (seq_len, batch_size, vector_dim)
        x = x + self.pe
        return self.dropout(x)

class XFMR_Neophyte(nn.Module):
    '''
    Basic Encoder stage alone sequence to sequence model
    '''
    def __init__(self, input_vcb_sz, output_vcb_sz,
                    emb_dim, n_layers,
                    attention_head = 8, feedfwd_dim = 1024,
                    max_seq_len = 50,
                    dropout = 0, device = "cpu"):

        super(XFMR_Neophyte, self).__init__()
        self.device = device

        self.input_vcb_sz = input_vcb_sz
        self.output_vcb_sz = output_vcb_sz
        self.vector_dim = emb_dim  # same size will be used for all layers in transformer
        self.atten_head = attention_head
        self.n_layers = n_layers
        self.feedfwd_dim = feedfwd_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout


        self.in2embed = nn.Embedding(self.input_vcb_sz, self.vector_dim)
        # pos_encoder non-learnable layer
        self.pos_encoder = PositionalEncoding(self.vector_dim, self.dropout,
                                                device = self.device)

        _enc_layer = nn.TransformerEncoderLayer(d_model= self.vector_dim,
                                                nhead= self.atten_head,
                                                dim_feedforward= self.feedfwd_dim,
                                                dropout= self.dropout
                                                )
        self.xfmr_enc = nn.TransformerEncoder(_enc_layer, num_layers= n_layers)

        self.out_fc = nn.Sequential( nn.LeakyReLU(),
            nn.Linear(self.vector_dim, self.vector_dim),
            nn.LeakyReLU(),
            nn.Linear(self.vector_dim, self.output_vcb_sz),
            )

    def forward(self, src, src_sz):
        '''
        src: (batch, max_seq_len-padded)
        tgt: (batch, max_seq_len-padded)
        '''

        # src_emb: (batch, in_seq_len, vector_dim)
        src_emb = self.in2embed(src)
        # src_emb: (max_seq_len, batch, vector_dim) -> for transformer
        src_emb = src_emb.permute(1,0,2)

        # src_emb: (max_seq_len, batch, vector_dim)
        src_emb =  src_emb * math.sqrt(self.vector_dim)
        src_emb = self.pos_encoder(src_emb)
        # out: (max_seq_len, batch, vector_dim)
        out = self.xfmr_enc(src_emb)

        # out: (batch, max_seq_len, vector_dim)
        out = out.permute(1,0,2).contiguous()

        # out: (batch, max_seq_len, out_vcb_dim)
        out = self.out_fc(out)
        # out: (batch, out_vcb_dim, max_seq_len)
        out = out.permute(0,2,1).contiguous()

        return out

    def inference(self, x):

        # inp: (1, max_seq_len)
        inp = torch.zeros(1, self.max_seq_len, dtype= torch.long).to(self.device)
        in_sz = min(x.shape[0], self.max_seq_len)
        inp[0, 0:in_sz ] = x[0:in_sz]

        # src_emb: (1, max_seq_len, vector_dim)
        src_emb = self.in2embed(inp)
        # src_emb: (max_seq_len,1,vector_dim) -> for transformer
        src_emb = src_emb.permute(1,0,2)

        # src_emb: (max_seq_len, 1, vector_dim)
        src_emb =  src_emb * math.sqrt(self.vector_dim)
        src_emb = self.pos_encoder(src_emb)
        # out: (max_seq_len, 1, vector_dim)
        out = self.xfmr_enc(src_emb)

        # out: (1, max_seq_len, vector_dim)
        out = out.permute(1,0,2).contiguous()

        # out: (1, max_seq_len, out_vcb_dim)
        out = self.out_fc(out)
        # prediction: ( max_seq_len )
        prediction = torch.argmax(out, dim=2).squeeze()

        return prediction

##------------------------------------------------------------------------------
