import torch
import torch.nn as nn
from utils.Utils import args
from blocks.encoder import Encoder
from blocks.net_params import encoder_params, convlstm_encoder_params, head_params, sst_encoder_params
import os
import torch.nn.functional as F
from TC_estimate.MSFN_DC import EF_LSTM
import time


class MSFN_GF(nn.Module):
    def __init__(self, encoder1, encoder2, encoder3, n_hidden=args.hidden_dim):
        super(MSFN_GF, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3
        self.GMF = nn.GRU(input_size=n_hidden * 3, hidden_size=n_hidden)
        self.projector = nn.Sequential(
            nn.Linear(n_hidden, n_hidden, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(n_hidden, n_hidden, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(n_hidden, 1, bias=True),
            nn.LeakyReLU(),
        )

    def forward(self, x_1, x_2, x_3=None):
        x_1 = x_1.transpose(0, 1).contiguous()
        state, output_1 = self.encoder1(x_1)

        x_2 = x_2.transpose(0, 1).contiguous()
        output_2 = self.encoder2(x_2)

        if x_3 is not None:
            x_3 = x_3.transpose(0, 1).contiguous()
            state, output_3 = self.encoder3(x_3)
            out = torch.cat([output_1, output_2, output_3], dim=2)
        else:
            out = torch.cat([output_1, output_2], dim=2)
        out, hn = self.GMF(out)
        y = self.projector(out)
        return y


def get_MSFN_GF(load_states=False):
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1], head_params[0]).to(args.device)
    ef_encoder = EF_LSTM()
    sst_encoder = Encoder(sst_encoder_params[0], sst_encoder_params[1], head_params[0]).to(args.device)

    model = MSFN_GF(encoder, ef_encoder, sst_encoder).to(args.device)
    path = os.path.join(args.save_model, 'MSFN_GF.pth')
    if load_states and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=args.device))
        print('load temporal model from:', path)
    else:
        print('training from scratch')
    return model


if __name__ == '__main__':
    # (batch size,time step, channel, height, length)
    input1 = torch.rand(1, 30, 1, 256, 256).to(args.device)
    input2 = torch.rand(1, 30, 10).to(args.device)
    input3 = torch.rand(1, 30, 1, 60, 60).to(args.device)

    model = get_MSFN_GF()
    nParams = sum([p.nelement() for p in model.parameters()])
    print('number of parameters: %d' % nParams)
    start = time.time()
    output = model(input1, input2, input3)
    end = time.time()
    print(end - start)
    print(output.shape)

    # m = nn.AdaptiveAvgPool2d((5, 7))
    # input = torch.randn(1, 64, 8, 9)
    # output = m(input)
    # print(output.shape)
