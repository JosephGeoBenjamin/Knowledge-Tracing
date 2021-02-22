import torch
import torch.nn as nn
import random

class DKT_Embednet(nn.Module):
    '''
    Simple RNN based DKT model
    '''
    def __init__(self,  stud_count, stud_embed_dim,
                        skill_count, skill_embed_dim,
                        hidden_dim , layers = 1,
                        dropout = 0, device = "cpu"):
        super(DKT_Embednet, self).__init__()

        self.enc_hidden_dim = hidden_dim
        self.enc_layers = layers
        self.device = device
        self.stud_count, self.stud_embed_dim = stud_count, stud_embed_dim
        self.skill_count, self.skill_embed_dim = skill_count, skill_embed_dim

        self.stud_embed = nn.Embedding(self.stud_count, self.stud_embed_dim, padding_idx = 0)
        self.skill_embed = nn.Embedding(self.skill_count, self.skill_embed_dim, padding_idx = 0)

        self.enc_rnn = nn.LSTM(
                    input_size= self.stud_embed_dim+self.skill_embed_dim,
                    hidden_size= self.enc_hidden_dim,
                    num_layers= self.enc_layers,)
        self.linear = nn.Linear(self.enc_hidden_dim, 1)

    def forward(self, x1, x2, x_sz, hidden = None ):
        '''
        x_sz: (batch_size, 1) -  Unpadded sequence lengths used for pack_pad
        Return:
            output: (batch_size, max_length, hidden_dim)
            hidden: (n_layer*num_directions, batch_size, hidden_dim) | if LSTM tuple -(h_n, c_n)
        '''

        batch_sz = x1.shape[0]
        # replace all -1 values with 0 (unused embeddings)
        x1[x1 == -1] = 0; x2[x2 == -1] = 0

        # x: batch_size, max_length, enc_embed_dim
        stemb = self.stud_embed(x1)
        skemb = self.skill_embed(x2)
        x = torch.cat([stemb, skemb], dim = 2)

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


class DKT_Onehotnet(nn.Module):
    '''
    Simple RNN based DKT model
    '''
    def __init__(self,  stud_count, skill_count,
                        hidden_dim , layers = 1,
                        dropout = 0, device = "cpu"):
        super(DKT_Onehotnet, self).__init__()

        self.enc_hidden_dim = hidden_dim
        self.enc_layers = layers
        self.device = device
        self.stud_count = stud_count
        self.skill_count = skill_count

        self.enc_rnn = nn.LSTM(
                    input_size= self.stud_count+self.skill_count,
                    hidden_size= self.enc_hidden_dim,
                    num_layers= self.enc_layers,)
        self.linear = nn.Linear(self.enc_hidden_dim, 1)

    def forward(self, x1, x2, x_sz, hidden = None ):
        '''
        x: (batch_sz, max_len_idxs)
        x_sz: (batch_size, 1) -  Unpadded sequence lengths used for pack_pad
        Return:
            output: (batch_size, max_length, hidden_dim)
            hidden: (n_layer*num_directions, batch_size, hidden_dim) | if LSTM tuple -(h_n, c_n)
        '''

        batch_sz = x1.shape[0]
        # replace all -1 values with 0 (unused embeddings)
        x1[x1 == -1] = 0; x2[x2 == -1] = 0

        # x: batch_size, max_length, enc_ohe_dim
        stohe = torch.nn.functional.one_hot(x1, self.stud_count)
        skohe = torch.nn.functional.one_hot(x2, self.skill_count)
        x = torch.cat([stohe, skohe], dim = 2)
        x = x.type(torch.FloatTensor).to(self.device)

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