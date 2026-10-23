# TODO

from torch import Tensor, einsum

from .objects import Config, SiteWeights


def simple_update(a: Tensor, b: Tensor, weights_a: SiteWeights, weights_b: SiteWeights, G: Tensor, config: Config) -> tuple[Tensor,Tensor,Tensor]:
    abar = einsum("rulds,r,u,l->rulds", a, weights_a.r, weights_a.u, weights_a.l)
    bbar = einsum("rulds,r,l,d->rulds", b, weights_b.r, weights_b.l, weights_b.d)
    M = ...
    a_new = ...
    b_new = ...
    lambda_new = ...
    return a_new, b_new, lambda_new
