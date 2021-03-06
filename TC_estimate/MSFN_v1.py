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

# class MSFNv1(nn.Module):
#     def __init__(self, encoder1, encoder2, encoder3, n_hidden=args.hidden_dim):
#         super(MSFNv1, self).__init__()
#         self.encoder1 = encoder1
#         self.encoder2 = encoder2
#         self.encoder3 = encoder3
#
#         self.no_local = NONLocalBlock2D(n_hidden, inter_channels=n_hidden, sub_sample=True)
#         # self.pool1 = nn.AdaptiveMaxPool2d(output_size=(1, 1))
#         self.projector = nn.Sequential(
#             nn.Dropout(args.dropout),
#             nn.Linear(n_hidden * args.past_window * 3, 1)
#         )
#
#     def forward(self, x_1, x_2, x_3, return_nl_map=False):
#         x_1 = x_1.transpose(0, 1).contiguous()
#         state, output_1 = self.encoder1(x_1)
#         out = output_1.permute(1, 2, 0).contiguous()
#
#         x_2 = x_2.transpose(0, 1).contiguous()
#         output_2 = self.encoder2(x_2)
#         out_2 = output_2.permute(1, 2, 0).contiguous()
#
#         x_3 = x_3.transpose(0, 1).contiguous()
#         state, output_3 = self.encoder3(x_3)
#         out_3 = output_3.permute(1, 2, 0).contiguous()
#         out = torch.stack([out, out_2, out_3], dim=3)
#
#         # go through a relu function to make sure the feature maps are all positive
#         out = torch.relu(out)
#
#         if return_nl_map is True:
#             out, f_div_C, W_y = self.no_local(out, return_nl_map=True)
#             out = torch.flatten(out,1)
#             y = self.projector(out)
#             return y, f_div_C, W_y
#
#         else:
#             out = self.no_local(out, return_nl_map=False)
#             out = torch.flatten(out,1)
#             y = self.projector(out)
#             return y


class MSFNv1(nn.Module):
    def __init__(self, encoder1, encoder2, encoder3, n_hidden=args.hidden_dim):
        super(MSFNv1, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3

        self.no_local = NONLocalBlock2D(n_hidden, inter_channels=n_hidden, sub_sample=True)
        self.pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.projector = nn.Sequential(
            # nn.Linear(n_hidden, n_hidden, bias=False),
            nn.Dropout(args.dropout),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x_1, x_2, x_3, return_nl_map=False):
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

        # use a leaky_relu function to force the feature maps being positive
        out = F.leaky_relu(out, negative_slope=0.1)

        f_div_C = None
        W_y = None
        if return_nl_map is True:
            out, f_div_C, W_y = self.no_local(out, return_nl_map=True)
        else:
            out = self.no_local(out, return_nl_map=False)
        out = torch.relu(out)
        # out = torch.flatten(out, 1)
        out = self.pool(out).squeeze()
        y = self.projector(out)
        if W_y is None:
            return y
        else:
            return y, f_div_C, W_y


def get_MSFN_v1(load_states=False, model_name='MSFN_v1.pth'):
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1], head_params[0]).to(args.device)
    ef_encoder = EF_LSTM()
    sst_encoder = Encoder(sst_encoder_params[0], sst_encoder_params[1], head_params[0]).to(args.device)
    model = MSFNv1(encoder, ef_encoder, sst_encoder).to(args.device)
    path = os.path.join(args.save_model, model_name)
    if load_states and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=args.device))
        print('load temporal model from:', path)
    else:
        print('training from scratch')
    return model


if __name__ == '__main__':
    # (batch size,time step, channel, height, length)
    input1 = torch.rand(4, 3, 1, 256, 256).to(args.device)
    input2 = torch.rand(4, 3, 10).to(args.device)
    input3 = torch.rand(4, 3, 1, 60, 60).to(args.device)

    model = get_MSFN_v1(load_states=False)
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
