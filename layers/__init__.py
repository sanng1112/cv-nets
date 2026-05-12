import argparse
import importlib
import inspect
import os


from layers.conv_layer import *
from layers.dropout import *
from layers.linear_layer import *
from layers.activation import *
from layers.flatten import *
from layers.pooling import *
from layers.base_layer import BaseLayer




def layer_specific_args(parser: argparse.ArgumentParser):
    layer_dir = os.path.dirname(__file__)
    parsed_layers = []
    for file in os.listdir(layer_dir):
        path = os.path.join(layer_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            layer_name = file[: file.find(".py")] if file.endswith(".py") else file
            module = importlib.import_module("layers." + layer_name)
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, BaseLayer) and name not in parsed_layers:
                    parser = cls.add_arguments(parser)
                    parsed_layers.append(name)
    return parser


def arguments_nn_layers(parser: argparse.ArgumentParser):

    parser = layer_specific_args(parser)

    from layers.activation import arguments_activation_fn

    parser = arguments_activation_fn(parser)

    # from layers.normalization import arguments_norm_layers

    # parser = arguments_norm_layers(parser)

    return parser
