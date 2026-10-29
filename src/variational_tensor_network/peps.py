"""Build double-layer tensors from physical PEPS tensors."""

from torch import Tensor, einsum

from .objects import BondWeights, SiteWeights
from .tilings import Site, UnitCellCoord

type TensorSites = dict[Site, Tensor]


def double_layer(a: Tensor,) -> Tensor:
    """Build a real double layer with contracted physical legs."""
    D = a.shape[0]
    if tuple(a.shape[:4]) != (D, D, D, D):
        raise ValueError("All four virtual dimensions must match.")
    A = einsum("rulds,RULDs->rRuUlLdD", a, a)
    D2 = D**2
    return A.reshape(D2,D2,D2,D2)


def open_double_layer(a: Tensor) -> Tensor:
    """Build a real double layer with both physical indices open."""
    D, d = a.shape[0], a.shape[-1]
    if tuple(a.shape[:4]) != (D, D, D, D):
        raise ValueError("All four virtual dimensions must match.")
    A = einsum("rulds,RULDS->rRuUlLdDsS", a, a)
    return A.reshape(D**2, D**2, D**2, D**2, d, d)


def get_site_weights(site: Site, bond_weights: BondWeights, unitcell_coord: UnitCellCoord) -> SiteWeights:
    x, y = site
    center = unitcell_coord(x, y)
    up = unitcell_coord(x, y - 1)
    left = unitcell_coord(x - 1,y)
    return SiteWeights(r = bond_weights[(center, "r")], u = bond_weights[(up, "d")], l = bond_weights[(left, "r")], d = bond_weights[(center, "d")])


def rotate_site_weights(weights: SiteWeights) -> SiteWeights:
    """Reorder weights to match tensor.permute(1, 2, 3, 0, 4)."""
    return SiteWeights(r=weights.u, u=weights.l, l=weights.d, d=weights.r)


def absorb_sqrt_weights(a: Tensor, weights: SiteWeights) -> Tensor:
    """Absorb the square root of each adjacent bond weight."""
    return einsum("rulds,r,u,l,d->rulds", a, weights.r.sqrt(), weights.u.sqrt(), weights.l.sqrt(), weights.d.sqrt())
