# Network Pruning (The project is in progress)

There are three files to prune a network

- Global pruning (modify the weights before the forward, most common)
  - Structured pruning
    - self-implemented code [prune.py](prune.py)
    - torch.nn.utils.prune [./prune_pytorch/](./prune_pytorch/) (not recommended)
  - Unstructured pruning
    - self-implemented code [prune.py](prune.py)
- Only backward pruning (original model in the forwards and sparse model in the backward when computing gradients)
  - self-implemented code [prune_only_backward.py](prune_only_backward.py)


The implement details follow the Section 3.2 in [Enhancing Targeted Attack Transferability via Diversified Weight Pruning](https://arxiv.org/abs/2208.08677), where `p_prune` and `p_bern` are the hyperparameters to control the pruning ratio.

## Reference

https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#remove-pruning-re-parametrization

Wang, Hung-Jui, Yu-Yu Wu, and Shang-Tse Chen. "Enhancing Targeted Attack Transferability via Diversified Weight Pruning." arXiv preprint arXiv:2208.08677 (2022).

Han, Song, et al. "Learning both weights and connections for efficient neural network." Advances in neural information processing systems 28 (2015).
