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

        self.stud_embed = nn.Embedding(self.stud_count, self.stud_embed_dim)
        self.skill_embed = nn.Embedding(self.skill_count, self.skill_embed_dim)

        self.enc_rnn = nn.LSTM(
                    input_size= self.stud_embed_dim+self.skill_embed_dim,
                    hidden_size= self.enc_hidden_dim,
                    num_layers= self.enc_layers,)

    def forward(self, x1, x2, x_sz, hidden = None):
        '''
        x_sz: (batch_size, 1) -  Unpadded sequence lengths used for pack_pad
        Return:
            output: (batch_size, max_length, hidden_dim)
            hidden: (n_layer*num_directions, batch_size, hidden_dim) | if LSTM tuple -(h_n, c_n)
        '''
        batch_sz = x1.shape[0]
        # x: batch_size, max_length, enc_embed_dim
        stemb = self.stud_embed(x1)
        skemb = self.skill_embed(x2)
        x = torch.cat([stemb, skemb], dim = 2)

        ## pack the padded data
        # x: max_length, batch_size, enc_embed_dim -> for pack_pad
        x = x.permute(1,0,2)
        x = nn.utils.rnn.pack_padded_sequence(x, x_sz, enforce_sorted=False) # unpad

        # output: packed_size, batch_size, enc_embed_dim --> hidden from all timesteps
        # hidden: n_layer**num_directions, batch_size, hidden_dim | if LSTM (h_n, c_n)
        output, hidden = self.enc_rnn(x)

        ## pad the sequence to the max length in the batch
        # output: max_length, batch_size, enc_emb_dim*directions)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        # output: batch_size, max_length, hidden_dim
        output = output.permute(1,0,2)

        return output