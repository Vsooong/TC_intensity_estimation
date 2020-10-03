import torch
import torch.nn as nn
from utils.Utils import args
from blocks.encoder import Encoder
from blocks.net_params import encoder_params, convlstm_encoder_params, head_params
import os


class ConvRNN(nn.Module):
    def __init__(self, encoder, n_features=args.hidden_dim):
        super(ConvRNN, self).__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.LeakyReLU(),
            nn.Linear(n_features, n_features, bias=False),
            nn.LeakyReLU(),
            nn.Linear(n_features, 1, bias=True),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = x.transpose(0, 1).contiguous()
        state, output = self.encoder(x)
        out = output[-1]
        y = self.projector(out)
        return y


def get_basline_model(load_states=False):
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1], head_params[0]).to(args.device)
    model = ConvRNN(encoder).to(args.device)
    path = os.path.join(args.save_model, 'convlstm.pth')
    if load_states and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=args.device))
        print('load temporal model from:', path)
    else:
        print('training from scratch')
    return model


if __name__ == '__main__':
    # (batch size,time step, channel, height, length)
    input = torch.rand(4, 3, 1, 256, 256).to(args.device)
    model =get_basline_model()

    nParams = sum([p.nelement() for p in model.parameters()])
    print('number of parameters: %d' % nParams)
    output = model(input)
    print(output.shape)
    # print(state[0][0].shape)
