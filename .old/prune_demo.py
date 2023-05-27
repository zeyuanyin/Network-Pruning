import torch
import torchvision.models as models
import torch.nn.utils.prune as prune
import numpy as np
import torch
import torch.nn as nn


cnn_neurons = 0
model = models.resnet18(pretrained=True)
# for name, param in model.named_parameters():
#     if 'conv' in name:
#         # print(name)
#         # print(param.shape)

# for name, module in model.named_modules():
#     if isinstance(module, nn.Conv2d):
#         print(name)
#         cnn_neurons += module.weight.data.shape[0]
#         if 'conv' not in name: #layer3.0.downsample.0
#             print(module.weight.data.shape)
# print(cnn_neurons)
# print(model)
# exit()




######
# 获取模型中所有卷积层的权重 & l1 norm
conv_weights = []

l1_norm_list = []
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        weight = module.weight.data
        # conv_weights.append(module.weight.data)
        l1_norm = torch.norm(weight, p=1, dim=(1, 2, 3))
        assert l1_norm.shape[0] == weight.shape[0]
        l1_norm_list.append(l1_norm)

# print(l1_norms_list)
l1_norm_list_tensor = torch.cat(l1_norm_list)
print(l1_norm_list_tensor.shape)

# 找到数值较小的后50%权重作为‘待剪枝权重’


# sort the tensor in ascending order
sorted_l1_norm_list_tensor, _ = torch.sort(l1_norm_list_tensor)

print(sorted_l1_norm_list_tensor[0]) # smallest

p_prune = 0.7

# compute the index of the element that is 30% from the start of the tensor
index = int(p_prune * len(sorted_l1_norm_list_tensor))

# get the element at the computed index
threshold = sorted_l1_norm_list_tensor[index]
print(index, threshold)


#****************
# 获取掩码的形状
# mask_shape = model.conv1.weight.shape
# new_mask = torch.zeros(mask_shape)

# print(mask_shape)
# # 将新的掩码应用到模型中
# prune.CustomFromMask.apply(module=model.conv1, name="weight", mask=new_mask)
# print(model.conv1.weight_mask)
#************

p_bern = 0.5

i = 0
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        out_,in_,_,_ = module.weight.shape

        assert l1_norm_list[i].shape[0] == out_, f"{i},{l1_norm_list[i].shape[0]}, {out_}"

        # print(l1_norm_list[i])
        prune_candidate_mask = l1_norm_list[i] < threshold # True: prune candidate
        print(prune_candidate_mask)


        p_bern_matrix = torch.ones_like(prune_candidate_mask) * p_bern
        prunt_bernoulli_mask = torch.bernoulli(p_bern_matrix)
        print(prunt_bernoulli_mask)


        prune_mask =prune_candidate_mask * prunt_bernoulli_mask
        print(prune_mask)
        print(prune_mask.shape)

        new_mask = 1 - prune_mask # 1: remain, 0: prune
        print(new_mask)

        new_mask = new_mask.view(out_, 1, 1, 1).expand(module.weight.shape)



        prune.CustomFromMask.apply(module=module, name="weight", mask=new_mask)

        i += 1
        #


exit()
