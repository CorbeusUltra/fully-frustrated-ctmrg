"""Build double-layer tensors from physical PEPS tensors."""

from torch import Tensor, einsum

from .objects import Config
from .unitcell import Site

type TensorSites = dict[Site, Tensor]


def double_layer(a: Tensor, config: Config | None = None) -> Tensor:
    if config:
        D = config.D
        d = config.d
    else :
        D = a.shape[0]
        d = a.shape[-1]
    expected_shape = (D, D, D, D, d)
    if tuple(a.shape) != expected_shape:
        raise ValueError(f"Expected a.shape == {expected_shape}, got {tuple(a.shape)}.")
    A = einsum("ijkls,IJKLs->iIjJkKlL", a, a)
    D2 = D**2
    return A.reshape(D2,D2,D2,D2)
