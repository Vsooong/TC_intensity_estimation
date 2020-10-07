from utils.Utils import args
from collections import OrderedDict
from blocks.trajGRU import TrajGRU
from blocks.convLSTM import ConvLSTM
from utils.util_layer import Flatten, UnFlatten
import torch.nn as nn


# 卷积后tensor大小
def tensor_size_after_conv(height, width, kernel_size, stride, padding=0):
    return int((height - kernel_size + 2 * padding) / stride) + 1, int((width - kernel_size + 2 * padding) / stride) + 1


def tensor_size_after_deconv(height, width, kernel_size, stride, padding):
    return int((height - 1) * stride + kernel_size - 2 * padding[0]), int(
        (width - 1) * stride + kernel_size - 2 * padding[1])


# build model
encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 0]}),
        OrderedDict({'conv2_leaky_1': [8, 32, 5, 3, 0]}),
        OrderedDict({'conv3_leaky_1': [32, 128, 3, 2, 0]}),
    ],

    [
        TrajGRU(input_channel=8, num_filter=8, b_h_w=(50, 50), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=args.rnn_act),

        TrajGRU(input_channel=32, num_filter=32, b_h_w=(16, 16), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=args.rnn_act),
        TrajGRU(input_channel=128, num_filter=128, b_h_w=(7, 7), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=args.rnn_act)
    ]
]

forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [128, 32, 4, 2, 0]}),
        OrderedDict({'deconv2_leaky_1': [32, 8, 5, 3, 0]}),
        OrderedDict({
            'deconv3_leaky_1': [8, 8, 7, 5, 0],
            'deconv4_leaky_2': [8, 8, 5, 1, 0],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        TrajGRU(input_channel=128, num_filter=128, b_h_w=(7, 7), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=args.rnn_act),

        TrajGRU(input_channel=32, num_filter=32, b_h_w=(16, 16), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=args.rnn_act),
        TrajGRU(input_channel=8, num_filter=8, b_h_w=(50, 50), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=args.rnn_act)
    ]
]

# build model
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 0]}),
        OrderedDict({'conv2_leaky_1': [8, 32, 5, 3, 0]}),
        OrderedDict({'conv3_leaky_1': [32, 128, 3, 2, 0]}),
    ],

    [
        ConvLSTM(input_channel=8, num_filter=8, b_h_w=(50, 50),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=32, num_filter=32, b_h_w=(16, 16),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=128, num_filter=128, b_h_w=(7, 7),
                 kernel_size=3, stride=1, padding=1),
    ]
]

sst_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 3, 2, 1]}),
        OrderedDict({'conv2_leaky_1': [8, 32, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [32, 128, 3, 2, 0]}),
    ],
    [
        ConvLSTM(input_channel=8, num_filter=8, b_h_w=(30, 30),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=32, num_filter=32, b_h_w=(15, 15),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=128, num_filter=128, b_h_w=(7, 7),
                 kernel_size=3, stride=1, padding=1),
    ]
]

convlstm_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [128, 32, 4, 2, 0]}),
        OrderedDict({'deconv2_leaky_1': [32, 8, 5, 3, 0]}),
        OrderedDict({
            'deconv3_leaky_1': [8, 8, 7, 5, 0],
            'deconv4_leaky_2': [8, 8, 5, 1, 0],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM(input_channel=128, num_filter=128, b_h_w=(7, 7),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=32, num_filter=32, b_h_w=(16, 16),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=8, num_filter=8, b_h_w=(50, 50),
                 kernel_size=3, stride=1, padding=1),
    ]
]

conv2d_params = OrderedDict({
    'conv1_relu_1': [5, 64, 7, 5, 1],
    'conv2_relu_1': [64, 192, 5, 3, 1],
    'conv3_relu_1': [192, 192, 3, 2, 1],
    'deconv1_relu_1': [192, 192, 4, 2, 1],
    'deconv2_relu_1': [192, 64, 5, 3, 1],
    'deconv3_relu_1': [64, 64, 7, 5, 1],
    'conv3_relu_2': [64, 20, 3, 1, 1],
    'conv3_3': [20, 20, 1, 1, 0]
})

head_params = [
    OrderedDict([
        ('maxpool1', nn.MaxPool2d(2, 2)),
        ('conv1', nn.Conv2d(128, 256, 3, 1)),

    ])
]
if __name__ == '__main__':
    size = tensor_size_after_conv(15, 15, 3, 2, 0)
    print(size)
