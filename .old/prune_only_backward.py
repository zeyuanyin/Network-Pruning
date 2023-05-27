import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision.models as models



def model_with_hook(model, p_prune):

    # threshold = get_threshold(model, p_prune)
    threshold = get_threshold_unstructured(model, p_prune)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):

            # Create a new Conv2dWithHooks layer
            conv_hook = Conv2dWithHooks(module.in_channels, module.out_channels, module.kernel_size,
                                        module.stride, module.padding, module.dilation,
                                        module.groups, module.bias, module.padding_mode, weight_mask_threshold = threshold)

            # Assign the Conv2dWithHooks attributes to match the nn.Conv2d layer
            conv_hook.weight = nn.Parameter(module.weight.clone())
            if module.bias is not None:
                conv_hook.bias = nn.Parameter(module.bias.clone())
            # conv_hook.padding = module.padding
            # conv_hook.padding_mode = module.padding_mode

            # Replace the existing nn.Conv2d layer with the new Conv2dWithHooks layer


            parent_module, name = get_parent_module_and_name(model, module)
            setattr(parent_module, name, conv_hook)
            # name = get_name(model, module)
            # setattr(model, name, conv_hook)

    return model

# threshold for structured pruning
def get_threshold(model,p_prune):
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
    return threshold

# threshold for unstructured pruning
def get_threshold_unstructured(model,p_prune):
    ### Step 1: consider weights in cnn
    l1_norm_list = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            l1_norm = abs(weight.view(-1))
            assert len(l1_norm.shape) == 1
            l1_norm_list.append(l1_norm)

    ### Step 2: find the smallest 'p_prune' weights
    l1_norm_list_tensor = torch.cat(l1_norm_list)
    sorted_l1_norm_list_tensor, _ = torch.sort(l1_norm_list_tensor)
    # print(sorted_l1_norm_list_tensor[0]) # check smallest
    index = int(p_prune * len(sorted_l1_norm_list_tensor))
    threshold = sorted_l1_norm_list_tensor[index]
    # print(index, threshold)
    return threshold



def get_parent_module_and_name(model, module):
    """
    Get the parent module and name of a given module in a PyTorch model.
    Args:
        model (nn.Module): The PyTorch model containing the module.
        module (nn.Module): The module for which to find the parent module.
    Returns:
        parent_module (nn.Module): The parent module of the given module.
        name (str): The name of the given module.
    """
    for name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            if child_module == module:
                return parent_module, child_name
    return None, None




class Conv2dWithHooks(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # remove the weight_mask_threshold argument from kwargs
        self.weight_mask_threshold = kwargs.pop('weight_mask_threshold')
        # print(self.weight_mask_threshold)
        super().__init__(*args, **kwargs)
        self.weight_shape = self.weight.shape


    def forward(self, input):
        with torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack):
            return super().forward(input)

    def pack(self, x): # nothing to do in the forward pass
        return x

    def unpack(self, x):
        if x.shape == self.weight_shape: # x is weight
            # weight_mask = self.get_weight_mask()
            weight_mask = self.get_weight_mask_unstructured()
            x = x * weight_mask
        return x

    # weight_mask for unstructured pruning
    def get_weight_mask_unstructured(self):
        p_bern = 0.5
        ### Step 3: get prune mask & prune
        out_,in_,_,_ = self.weight.shape

        l1_norm = abs(self.weight)

        prune_candidate_mask = l1_norm < self.weight_mask_threshold

        p_bern_matrix = torch.ones_like(prune_candidate_mask) * float(p_bern)
        prunt_bernoulli_mask = torch.bernoulli(p_bern_matrix)

        prune_mask = prune_candidate_mask * prunt_bernoulli_mask

        weight_mask = 1 - prune_mask # 1: remain, 0: prune
        # weight_mask = weight_mask.view(out_, in_, 1, 1).expand(self.weight.shape)

        return weight_mask

    # weight_mask for structured pruning
    # def get_weight_mask(self):
    #     p_bern = 0.5
    #     ### Step 3: get prune mask & prune
    #     out_,in_,_,_ = self.weight.shape

    #     l1_norm = torch.norm(self.weight, p=1, dim=(2, 3))
    #     assert l1_norm.shape == (out_, in_), f"{l1_norm.shape}, ({out_}, {in_})"

    #     prune_candidate_mask = l1_norm < self.weight_mask_threshold # True: prune candidate

    #     p_bern_matrix = torch.ones_like(prune_candidate_mask) * float(p_bern)
    #     prunt_bernoulli_mask = torch.bernoulli(p_bern_matrix)

    #     prune_mask = prune_candidate_mask * prunt_bernoulli_mask

    #     weight_mask = 1 - prune_mask # 1: remain, 0: prune
    #     weight_mask = weight_mask.view(out_, in_, 1, 1).expand(self.weight.shape)

    #     return weight_mask

if __name__ == "__main__":
    model = models.resnet18(pretrained=True)

    # print(model)
    # for name, param in model.named_parameters():
    #     print(name, param.shape)
    # exit()

    # module = model.conv1


    # conv_hook = Conv2dWithHooks(module.in_channels, module.out_channels, module.kernel_size,
    #                                     module.stride, module.padding, module.dilation,
    #                                     module.groups, module.bias, module.padding_mode)
    # conv_hook.weight = module.weight
    # conv_hook.bias = module.bias
    # conv_hook.padding = module.padding
    # conv_hook.padding_mode = module.padding_mode
    # setattr(model, 'conv1', conv_hook)



    for module in model.modules():
        if isinstance(module, nn.Conv2d):

            # Create a new Conv2dWithHooks layer
            conv_hook = Conv2dWithHooks(module.in_channels, module.out_channels, module.kernel_size,
                                        module.stride, module.padding, module.dilation,
                                        module.groups, module.bias, module.padding_mode,weight_mask_threshold = 0.4)

            # Assign the Conv2dWithHooks attributes to match the nn.Conv2d layer
            conv_hook.weight = module.weight
            if module.bias is not None:
                conv_hook.bias = module.bias
            # conv_hook.padding = module.padding
            # conv_hook.padding_mode = module.padding_mode

            # Replace the existing nn.Conv2d layer with the new Conv2dWithHooks layer
            parent_module, name = get_parent_module_and_name(model, module)
            setattr(parent_module, name, conv_hook)



            # parent_module, name = get_parent_module_and_name(model, module)
            # print(parent_module, name)
            # print('--')


    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_input.requires_grad = True
    output = model(dummy_input)
    print(output.shape)
    loss = output.sum()
    print(loss)
    loss.backward()
    print(dummy_input.grad.sum())
    exit()
