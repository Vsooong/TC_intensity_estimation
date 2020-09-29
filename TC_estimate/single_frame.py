import torchvision
import torch.nn as nn
from utils.Utils import args
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]

class Res_Est(nn.Module):
    def __init__(self, encoder, n_features):
        super(Res_Est, self).__init__()
        self.encoder=encoder
        self.n_feature=n_features
        self.encoder.fc=Identity()
        self.projector = nn.Sequential(
            nn.Linear(self.n_feature, self.n_feature, bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.n_feature, 1, bias=False),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        y = self.projector(self.encoder(x))
        return y

def get_pretrained_model(load_states=False):
    if load_states:
        encoder = get_resnet('resnet50', False)
    else:
        encoder = get_resnet('resnet50', True)
    n_features = encoder.fc.in_features
    predict_model = Res_Est(encoder, n_features).to(args.device)
    if load_states:
        pass
    return predict_model