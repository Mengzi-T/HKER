import torch
from torch import Tensor
from torch.nn.init import uniform_, constant_


@torch.no_grad()
def get_uni_init_euc_weights(tensor: Tensor, requires_grad: bool = True, scale: float = 0.1, padding_idx=None) -> torch.nn.Parameter:
    assert scale > 0
    uniform_(tensor, -scale, scale)
    if padding_idx is not None:
        constant_(tensor[padding_idx], 0)
    return torch.nn.Parameter(tensor, requires_grad=requires_grad)

@torch.no_grad()
def get_pop_init_euc_weights(tensor: Tensor, pop: Tensor, requires_grad: bool = True, scale: float = 1., padding_idx=None) -> torch.nn.Parameter:
    assert scale > 0
    uniform_(tensor, -scale, scale)
    bias = 1. - pop.min()
    # power-law
    tensor.div_((pop + bias).pow(1.1))
    if padding_idx is not None:
        constant_(tensor[padding_idx], 0)
    return torch.nn.Parameter(tensor, requires_grad=requires_grad)