import torch
import torch.nn as nn
from utils.Utils import args
from blocks.encoder import Encoder
from blocks.net_params import convlstm_encoder_params, head_params, sst_encoder_params
import os
import torch.nn.functional as F
import time

class EF_LSTM(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=args.hidden_dim):
        super(EF_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)

    def forward(self, x):
        output, (h, c) = self.lstm1(x)
        output, (h, c) = self.lstm2(output)
        return output


class MSFN_DC(nn.Module):
    def __init__(self, encoder1, encoder2, encoder3=None, n_features=args.hidden_dim, n_hidden=args.hidden_dim):
        super(MSFN_DC, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        n_features += args.hidden_dim
        self.encoder3 = encoder3
        if encoder3 is not None:
            n_features += args.hidden_dim
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_hidden, bias=False),
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
        out = output_1

        x_2 = x_2.transpose(0, 1).contiguous()
        output_2 = self.encoder2(x_2)
        out_2 = output_2

        if x_3 is not None:
            x_3 = x_3.transpose(0, 1).contiguous()
            state, output_3 = self.encoder3(x_3)
            out_3 = output_3
            out = torch.cat([out, out_2, out_3], dim=-1)
        else:
            out = torch.cat([out, out_2], dim=-1)

        y = self.projector(out)
        return y


def get_MSFN_DC(load_states=False):
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1], head_params[0]).to(args.device)
    ef_encoder = EF_LSTM()
    sst_encoder = Encoder(sst_encoder_params[0], sst_encoder_params[1], head_params[0]).to(args.device)
    model = MSFN_DC(encoder, ef_encoder, sst_encoder).to(args.device)
    path = os.path.join(args.save_model, 'MSFN_DC.pth')
    if load_states and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=args.device))
        print('load temporal model from:', path)
    else:
        print('training from scratch')
    return model


if __name__ == '__main__':
    # (batch size,time step, channel, height, length)
    input1 = torch.rand(2, 32, 1, 256, 256).to(args.device)
    input2 = torch.rand(2, 32, 10).to(args.device)
    input3 = torch.rand(2, 32, 1, 60, 60).to(args.device)

    model = get_MSFN_DC()
    nParams = sum([p.nelement() for p in model.parameters()])
    print('number of parameters: %d' % nParams)
    start = time.time()
    output = model(input1, input2, input3)
    end = time.time()
    print(end - start)
    print(output.shape)
