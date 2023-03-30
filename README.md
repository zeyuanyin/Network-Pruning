# Network Pruning (The project is in progress)


There are three files to prune a network

- Global pruning (modify the weights before the forward, most common)
  - handwritten pruning  [prune.py](prune.py)
  - torch.nn.utils.prune  [./prune_pytorch/](./prune_pytorch/)
- DWP pruning (modify the weights after the forward and before the backward)
  - handwritten pruning [prune_dwp.py](prune_dwp.py)


The implement details follow the Section 3.2 in [Enhancing Targeted Attack Transferability via Diversified Weight Pruning](https://arxiv.org/abs/2208.08677), where `p_prune` and `p_bern` are the hyperparameters to control the pruning ratio.

<p align="center">
  <img src="https://github.com/zeyuanyin/network-prune/blob/main/img/algorithm_1.png" width="400" alt="Image 1" />
  <img src="https://github.com/zeyuanyin/network-prune/blob/main/img/algorithm_2.png" width="400" alt="Image 2" />
</p>



## HOW TO USE

### Global pruning

Choose pruned method `prune connections` or `prune neurons` in the file `eval.py`.

https://github.com/zeyuanyin/network-prune/blob/3864370df119b4c1457bd714c2749d45eeac3149/eval.py#L197-L201


### DWP pruning
```model = model_with_hook(model, p_prune = 0.7)```

## Evaluation of pruned networks

Run the script
```
python eval.py --arch=resnet18 --p_prune=0.2 --p_bern=1
```
## Results on torch.nn.utils.prune
### prune connections
| network | p_prune | p_bern | pruning ratio | eval performance|
|:-------:|:-------:|:---------------:|:---------------:|:------------:|
| resnet18 | 0 | 0 | 0 | Loss 1.247      Acc@1 69.758    Acc@5 89.078 |
| resnet18 | 0.1 | 1 | 10% | Loss 1.313      Acc@1 67.988    Acc@5 88.240 |
| resnet18 | 0.2 | 1 | 20% | Loss 1.800      Acc@1 59.350    Acc@5 82.404 |
| resnet18 | 0.2 | 0.5 | 10% | Loss 1.377      Acc@1 66.662    Acc@5 87.448 |
| resnet18 | 0.4 | 0.5 | 20% | Loss 1.747      Acc@1 59.488    Acc@5 82.566 |
| --- | --- | --- | --- | --- |
| resnet50 | 0 | 0 | 0 | Loss 0.962      Acc@1 76.124    Acc@5 92.858 |
| resnet50 | 0.1 | 1 | 10% | Loss 0.962      Acc@1 76.114    Acc@5 92.896 |
| resnet50 | 0.2 | 1 | 20% | Loss 0.964      Acc@1 76.036    Acc@5 92.890 |
| resnet50 | 0.3 | 1 | 30% | Loss 0.978      Acc@1 75.646    Acc@5 92.732 |
| resnet50 | 0.4 | 1 | 40% | Loss 1.006      Acc@1 75.054    Acc@5 92.384 |

### prune neurons
| network | p_prune | p_bern | pruning ratio | eval performance|
|:-------:|:-------:|:---------------:|:---------------:|:------------:|
| resnet18 | 0 | 0 | 0 | Loss 1.247      Acc@1 69.758    Acc@5 89.078 |
| resnet18 | 0.05 | 1 | 5% | Loss 1.962      Acc@1 55.118    Acc@5 78.726 |
| resnet18 | 0.08 | 1 | 8% | Loss 4.992      Acc@1 16.980    Acc@5 35.244 |
| resnet18 | o.1 | 1 | 10% | Loss 8.508      Acc@1 2.038     Acc@5 6.142 |
| --- | --- | --- | --- | --- |
| resnet50 | 0 | 0 | 0 | Loss 0.962      Acc@1 76.124    Acc@5 92.858 |
| resnet50 | 0.05 | 1 | 5% | Loss 0.980      Acc@1 75.536    Acc@5 92.596 |
| resnet50 | 0.08 | 1 | 8% | Loss 1.784      Acc@1 59.102    Acc@5 81.838 |
| resnet50 | 0.1 | 1 | 10% | Loss 8.296      Acc@1 2.480     Acc@5 6.466 |

## Restore weights on torch.nn.utils.prune

After applying `prune.CustomFromMask`, original weights are stored in `module.weight_orig` while `module.weight` has been replaced by pruned weights.

- save the cloned `module.weight_orig` in `orig`
- use `torch.nn.utils.prune.remove` to remove the mask & forward hook
- replace `module.weight` with `orig`.

https://github.com/zeyuanyin/network-prune/blob/3864370df119b4c1457bd714c2749d45eeac3149/prune.py#L103-L105


Another way is to reload the original weights from scratch.

## Reference

https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#remove-pruning-re-parametrization

Wang, Hung-Jui, Yu-Yu Wu, and Shang-Tse Chen. "Enhancing Targeted Attack Transferability via Diversified Weight Pruning." arXiv preprint arXiv:2208.08677 (2022).

Han, Song, et al. "Learning both weights and connections for efficient neural network." Advances in neural information processing systems 28 (2015).
