import os
import yaml
import argparse
import torch
import torch.nn.functional as F


class activation():

    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file, 'rb') as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


def get_config():
    parser = argparse.ArgumentParser(description="estimation")
    f = os.path.dirname(__file__)
    f = os.path.join(os.path.dirname(f), "config_tc.yaml")
    config = yaml_config_hook(f)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    for dirs in args.model_save1:
        if os.path.exists(dirs):
            args.save_model = dirs
    args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    args.rnn_act = activation('leaky', negative_slope=0.2, inplace=True)
    for dirs in args.img_root_dir:
        if os.path.exists(dirs):
            args.img_root = dirs
    return args


args = get_config()
