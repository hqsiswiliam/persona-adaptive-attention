import copy

from torch import nn
import torch

def copy_model(model):
    copyOfModel = copy.deepcopy(model)
    return copyOfModel


def selective_layers(model, layers_to_start, layer_to_end):  # must pass in the full bert model
    oldModuleList = model.h
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keep selective layers.
    for i in range(layers_to_start, layer_to_end):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.h = newModuleList

    return copyOfModel


def check_layer_equivalence(layer1, layer2, key_to_compare='attn.c_attn.weight'):
    result = (layer1.state_dict()[key_to_compare] == layer2.state_dict()[key_to_compare]).flatten().unique()
    list_result = result.numpy().tolist()
    if len(list_result) == 1 and list_result[0]:
        return True
    return False
