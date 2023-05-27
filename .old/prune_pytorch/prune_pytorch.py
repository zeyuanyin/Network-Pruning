import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.utils.prune as prune
import numpy as np

def prune_connection(model, p_prune = 0.3, p_bern = 1.):
    ### Step 1: consider weights in cnn & l1 norm
    l1_norm_list = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            # conv_weights.append(module.weight.data)
            l1_norm = torch.norm(weight, p=1, dim=(2, 3))
            assert l1_norm.shape[0] == weight.shape[0]
            l1_norm_list.append(l1_norm.view(-1))


    ### Step 2: find the smallest 'p_prune' weights


    l1_norm_list_tensor = torch.cat(l1_norm_list)
    sorted_l1_norm_list_tensor, _ = torch.sort(l1_norm_list_tensor)
    # print(sorted_l1_norm_list_tensor[0]) # smallest
    # print(sorted_l1_norm_list_tensor[-1]) # largest

    index = int(p_prune * len(sorted_l1_norm_list_tensor))
    threshold = sorted_l1_norm_list_tensor[index]

    ### Step 3: get prune mask & prune


    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            out_,in_,_,_ = module.weight.shape

            l1_norm = torch.norm(module.weight, p=1, dim=(2, 3))
            assert l1_norm.shape == (out_, in_), f"{l1_norm.shape}, ({out_}, {in_})"

            prune_candidate_mask = l1_norm < threshold # True: prune candidate

            p_bern_matrix = torch.ones_like(prune_candidate_mask) * float(p_bern)
            prunt_bernoulli_mask = torch.bernoulli(p_bern_matrix)

            prune_mask = prune_candidate_mask * prunt_bernoulli_mask

            new_mask = 1 - prune_mask # 1: remain, 0: prune
            new_mask = new_mask.view(out_, in_, 1, 1).expand(module.weight.shape)

            prune.CustomFromMask.apply(module=module, name="weight", mask=new_mask)

    return model



def prune_neuron(model, p_prune = 0.3, p_bern = 1.):
    ### Step 1: consider weights in cnn & l1 norm
    l1_norm_list = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            l1_norm = torch.norm(weight, p=1, dim=(1, 2, 3))
            assert l1_norm.shape[0] == weight.shape[0]
            l1_norm_list.append(l1_norm)


    ### Step 2: find the smallest 'p_prune' weights
    # p_prune = 0
    l1_norm_list_tensor = torch.cat(l1_norm_list)
    sorted_l1_norm_list_tensor, _ = torch.sort(l1_norm_list_tensor)
    # print(sorted_l1_norm_list_tensor[0]) # check smallest
    index = int(p_prune * len(sorted_l1_norm_list_tensor))
    threshold = sorted_l1_norm_list_tensor[index]
    # print(index, threshold)

    ### Step 3: get prune mask & prune
    # p_bern = 0.5

    i = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            out_,in_,_,_ = module.weight.shape
            assert l1_norm_list[i].shape[0] == out_, f"{i},{l1_norm_list[i].shape[0]}, {out_}"

            prune_candidate_mask = l1_norm_list[i] < threshold # True: prune candidate

            p_bern_matrix = torch.ones_like(prune_candidate_mask) * float(p_bern)
            prunt_bernoulli_mask = torch.bernoulli(p_bern_matrix)

            prune_mask =prune_candidate_mask * prunt_bernoulli_mask

            new_mask = 1 - prune_mask # 1: remain, 0: prune
            new_mask = new_mask.view(out_, 1, 1, 1).expand(module.weight.shape)

            prune.CustomFromMask.apply(module=module, name="weight", mask=new_mask)
            i += 1

    return model

def restore_weight(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and  prune.is_pruned(module):
            orig = module.weight_orig.clone() # must use .clone()
            prune.remove(module, 'weight') # remove the mask & forward hook
            module.weight = nn.Parameter(orig)

    return model