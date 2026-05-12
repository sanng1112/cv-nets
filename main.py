from layers import activation
from layers.activation import SUPPORTED_ACT_FNS
from layers import arguments_nn_layers
import yaml
from types import SimpleNamespace

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{
            k: dict_to_namespace(v)
            for k, v in d.items()
        })
    return d

CONFIG = 'config/demo.yaml'

with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

opts = dict_to_namespace(cfg)
print(opts)

print(opts.model.name)

