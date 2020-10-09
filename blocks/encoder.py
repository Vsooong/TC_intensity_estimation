from torch import nn
import torch
from utils.util_layer import make_layers
from utils.Utils import args
import logging
import torch.nn.functional as F
from blocks.net_params import convlstm_encoder_params, head_params, sst_encoder_params
from blocks.non_local_em_gaussian import NONLocalBlock3D


class Encoder(nn.Module):
    def __init__(self, subnets, rnns, head):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)
        # self.last_no_local3D = NONLocalBlock3D(args.hidden_dim)
        self.head = nn.Sequential(head)

    def forward_by_stage(self, input, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)

        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))

        outputs_stage, state_stage = rnn(input, None)

        return outputs_stage, state_stage

    # input: 5D S*B*I*H*W
    def forward(self, input):
        hidden_states = []
        logging.debug(input.size())
        for i in range(1, self.blocks + 1):
            #
            input, state_stage = self.forward_by_stage(input, getattr(self, 'stage' + str(i)),
                                                       getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)

        seq_number, batch_size, input_channel, height, width = input.size()
        # (b, c, t, h, w)
        # input = input.permute(1, 2, 0, 3, 4).contiguous()
        # input = self.last_no_local3D(input)
        # input = input.permute(2, 0, 1, 3, 4).contiguous()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = F.leaky_relu(self.head(input))

        output = input.squeeze()
        output = torch.reshape(output, (seq_number, batch_size, -1))
        return tuple(hidden_states), output


if __name__ == '__main__':
    # (time step,batch size, channel, height, length)
    input = torch.rand(30, 1, 1, 256, 256).to(args.device)
    model = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1], head_params[0]).to(args.device)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('number of parameters: %d' % nParams)
    state, output = model(input)
    print(output.shape)
    # print(state[0][0].shape)
