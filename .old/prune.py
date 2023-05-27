import torch
import torch.nn as nn


def prune_connection_X(model, p_prune = 0.3, p_bern = 1.):
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

            module = prune_from_mask(module, new_mask)

    return model

# add new attribute to module, named 'weight_orig', and save the original weight
# prune the weight
def prune_from_mask(module, mask):
    module.weight_orig = module.weight.clone() # must use .clone()
    module.weight = nn.Parameter(module.weight * mask)
    return module

def restore_weight_X(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and hasattr(module, 'weight_orig'):
            module.weight = nn.Parameter (module.weight_orig.clone()) # must use .clone()
            del module.weight_orig

    return model
