import torch
import torch.nn as nn
from utils.Utils import args
from blocks.encoder import Encoder
from blocks.net_params import encoder_params, convlstm_encoder_params, head_params, sst_encoder_params
import os
from torch.nn.utils.rnn import pack_padded_sequence
from TC_estimate.MSFN_DC import EF_LSTM
import time
from blocks.non_local_em_gaussian import NONLocalBlock2D, NONLocalBlock1D


class MSFN(nn.Module):
    def __init__(self, encoder1, encoder2, encoder3, n_hidden=args.hidden_dim):
        super(MSFN, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3
        self.no_local = NONLocalBlock2D(n_hidden, inter_channels=n_hidden, sub_sample=False)
        self.pool=nn.AdaptiveMaxPool2d(output_size=(1,1))
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
        window_size = args.past_window
        pad_len = T + window_size - 1

        x_1 = x_1.transpose(0, 1).contiguous()
        state, output_1 = self.encoder1(x_1)
        out_1 = output_1.permute(1, 2, 0).contiguous()

        x_2 = x_2.transpose(0, 1).contiguous()
        output_2 = self.encoder2(x_2)
        out_2 = output_2.permute(1, 2, 0).contiguous()

        x_3 = x_3.transpose(0, 1).contiguous()
        state, output_3 = self.encoder3(x_3)
        out_3 = output_3.permute(1, 2, 0).contiguous()
        out = torch.stack([out_1, out_2, out_3], dim=3)

        new_out=torch.zeros(size=(B,args.hidden_dim,pad_len,3),device=args.device)
        new_out[:,:,window_size-1:,:]=out
        outs=[]
        f_divs=[]
        W_ys=[]
        for idx in range(T):
            out_part=new_out[:,:,idx:idx+window_size,:]
            out_part, f_div_C, W_y = self.no_local(out_part, return_nl_map=True)
            out_part=self.pool(out_part)
            outs.append(out_part)
            f_divs.append(f_div_C)
            W_ys.append(W_y)

        out=torch.cat(outs,dim=-2).squeeze(-1)
        out=out.permute(2,0,1)
        out=self.projector(out)
        if return_nl_map:
            W_ys=torch.cat(W_ys,dim=0)
            f_divs = torch.cat(f_divs, dim=0)
            return out,f_divs,W_ys
        else:
            return out


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
    input1 = torch.rand(2, 32, 1, 256, 256).to(args.device)
    input2 = torch.rand(2, 32, 10).to(args.device)
    input3 = torch.rand(2, 32, 1, 60, 60).to(args.device)

    model = get_MSFN()
    nParams = sum([p.nelement() for p in model.parameters()])
    print('number of parameters: %d' % nParams)
    start = time.time()
    output, f_div_C, W_y = model(input1, input2, input3, return_nl_map=True)
    end = time.time()
    print(end - start)
    print(output.squeeze(-1).shape)
    print(W_y[0].shape)
