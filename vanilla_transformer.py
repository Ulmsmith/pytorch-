import torch
import torch.nn as nn
import math
import numpy as np
import anchor_free_helper

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, n_position=1024):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_model))

    def _get_sinusoid_encoding_table(self, n_position, d_model):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)#(1,N,d)

    def forward(self, x):
        # x(B,N,d)
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class Vanilla(nn.Module):
    def __init__(self, d_model=1024, nhead=8, d_hid=512, dropout=0.1, num_encoder_layer=6):
        super().__init__()
        num_hidden = 256
        self.position = PositionalEncoding(d_model=d_model, n_position=1024)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layer)
        self.fc1 = nn.Sequential(
            nn.Linear(d_model, out_features=num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
        )
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)
        self.fc_ctr = nn.Linear(num_hidden, 1)

    def forward(self, x):
        _, seq_len, _ = x.shape
        x = self.position(x)
        x = self.encoder(x)
        out = self.fc1(x)

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len)
        pred_loc = self.fc_loc(out).exp().view(seq_len, 2)

        #10/13
        #pred_loc = self.fc_loc(out).sigmoid().view(seq_len, 2)

        pred_ctr = self.fc_ctr(out).sigmoid().view(seq_len)

        return pred_cls, pred_loc, pred_ctr
    
    def predict(self, seq):
        pred_cls, pred_loc, pred_ctr = self(seq)

        pred_cls *= pred_ctr
        pred_cls /= pred_cls.max() + 1e-8

        pred_cls = pred_cls.cpu().numpy()
        pred_loc = pred_loc.cpu().numpy()

        #10/13
        #pred_loc = pred_loc*seq.shape[0]

        pred_bboxes = anchor_free_helper.offset2bbox(pred_loc)
        return pred_cls, pred_bboxes


if __name__ == '__main__':
    test_input = torch.randn(1, 288, 1024)
    model = Vanilla()
    pred_cls, pred_loc, pred_ctr = model(test_input)
    print(pred_cls.shape)
    print(pred_loc.shape)
    print(pred_ctr.shape)

        


