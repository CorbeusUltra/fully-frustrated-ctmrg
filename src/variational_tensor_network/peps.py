"""Build double-layer tensors from physical PEPS tensors."""

from torch import Tensor, einsum

from .objects import BondWeights, Config, SiteWeights
from .unitcell import Site, UnitCellCoord

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


def get_site_weights(site: Site, bond_weights: BondWeights, unitcell_coord: UnitCellCoord) -> SiteWeights:
    x, y = site
    center = unitcell_coord(x, y)
    up = unitcell_coord(x, y - 1)
    left = unitcell_coord(x - 1,y)
    return SiteWeights(r = bond_weights[(center, "r")], u = bond_weights[(up, "d")], l = bond_weights[(left, "r")], d = bond_weights[(center, "d")])
