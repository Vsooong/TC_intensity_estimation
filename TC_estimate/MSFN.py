import torch
import torch.nn as nn
from utils.Utils import args
from blocks.encoder import Encoder
from blocks.net_params import encoder_params, convlstm_encoder_params, head_params, sst_encoder_params
import os
import torch.nn.functional as F
from TC_estimate.MSFN_DC import EF_LSTM
import time
from blocks.non_local_em_gaussian import NONLocalBlock2D, NONLocalBlock1D


class MSFN(nn.Module):
    def __init__(self, encoder1, encoder2, encoder3, n_hidden=args.hidden_dim):
        super(MSFN, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3

        self.GMF = nn.GRU(input_size=n_hidden * 3, hidden_size=n_hidden)

        self.no_local = NONLocalBlock2D(n_hidden, inter_channels=n_hidden, sub_sample=False)
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

    def forward(self, x_1, x_2, x_3=None, return_nl_map=False):
        B, T, C, H, W = x_1.size()
        x_1 = x_1.transpose(0, 1).contiguous()
        state, output_1 = self.encoder1(x_1)
        out = output_1.permute(1, 2, 0).contiguous()

        x_2 = x_2.transpose(0, 1).contiguous()
        output_2 = self.encoder2(x_2)
        out_2 = output_2.permute(1, 2, 0).contiguous()

        x_3 = x_3.transpose(0, 1).contiguous()
        state, output_3 = self.encoder3(x_3)
        out_3 = output_3.permute(1, 2, 0).contiguous()
        out = torch.stack([out, out_2, out_3], dim=3)
        f_div_C = None
        W_y = None
        if return_nl_map is True:
            out, f_div_C, W_y = self.no_local(out, return_nl_map=True)
        else:
            out = self.no_local(out, return_nl_map=False)
        out = out.permute(2, 0, 1, 3).contiguous()
        out = out.view(T, B, -1)
        out, hn = self.GMF(out)
        y = self.projector(out)
        return y, f_div_C, W_y


def get_MSFN(load_states=False):
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1], head_params[0]).to(args.device)
    ef_encoder = EF_LSTM()
    sst_encoder = Encoder(sst_encoder_params[0], sst_encoder_params[1], head_params[0]).to(args.device)
    model = MSFN(encoder, ef_encoder, sst_encoder).to(args.device)
    path = os.path.join(args.save_model, 'MSFN.pth')
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

    model = get_MSFN()
    nParams = sum([p.nelement() for p in model.parameters()])
    print('number of parameters: %d' % nParams)
    start = time.time()
    output, f_div_C, W_y = model(input1, input2, input3, return_nl_map=True)
    end = time.time()
    print(end - start)
    print(output.shape)
    print(f_div_C.shape)
    print(W_y.shape)
