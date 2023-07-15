import torch
import torch.nn as nn


def prune(model, p=0.1, type='taylor'):
    """
    Prune the model parameters with the smallest 'p' weights
    Pruning is taken inplace
    Arguments:
        model (torch.nn.Module): the model to be pruned.
        p (float): the percentage of weights to be pruned.
        type (str): the type of pruning, 'taylor', 'l1', 'grad'
            - 'taylor': use first order taylor expansion to approximate the score (Pruning Convolutional Neural Networks for Resource Efficient Inference https://arxiv.org/abs/1611.06440)
            - 'l1': use l1 norm of the weights as score (Learning both Weights and Connections for Efficient Neural Networks https://arxiv.org/abs/1506.02626)
            - 'grad': use gradient of the weights as score
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if type == 'taylor':
                scores = torch.abs(module.weight.data * module.weight.grad)
            elif type == 'l1':
                scores = torch.norm(module.weight.data, p=1, dim=(2, 3))
            elif type == 'grad':
                scores = torch.abs(module.weight.grad)
            else:
                raise ValueError('Type {} not supported'.format(type))

            idx = int(scores.numel() * p)
            values, _ = scores.view(-1).sort()
            threshold = values[idx]
            mask = (scores > threshold).float().cuda()
            prune_from_mask(module, mask)
    return model


def prune_from_mask(module, mask):
    """
    Prune the model according to the mask. And add new attribute to module, named 'weight_orig' to save the original weight
    Pruning is taken inplace
    """
    module.weight_orig = module.weight.clone()  # must use .clone()
    module.weight = nn.Parameter(module.weight * mask)
    return module


def restore_weight(model):
    """
    Restore the model weights from 'weight_orig' attribute
    Pruning is taken inplace
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and hasattr(module, 'weight_orig'):
            module.weight = nn.Parameter(
                module.weight_orig.clone())  # must use .clone()
            del module.weight_orig
    return model
