"""Estimate energies using the simple-update product environment."""

import torch
from torch import Tensor, einsum

from .objects import BondWeights, SiteWeights
from .peps import TensorSites, get_site_weights
from .simple_update import rotate_site_weights
from .tilings import UnitCellCoord


def simple_update_bond_energy(a: Tensor, b: Tensor, weights_a: SiteWeights, weights_b: SiteWeights, h: Tensor) -> float:
    """Estimate a vertical bond energy, with a above b."""
    torch.testing.assert_close(weights_a.d, weights_b.u)
    abar = einsum("rulds,r,u,l->rulds", a, weights_a.r, weights_a.u, weights_a.l)
    bbar = einsum("rulds,r,l,d->rulds", b, weights_b.r, weights_b.l, weights_b.d)
    abarbbar = einsum("rulqs,RqLDS,q->rulLDRsS", abar, bbar, weights_a.d)
    d = a.shape[-1]
    abarbbar = abarbbar.reshape(-1, d**2)
    numerator = einsum("vi,ij,vj->", abarbbar, h, abarbbar)
    denominator = abarbbar.square().sum()
    if denominator.item() == 0:
        raise ValueError("Cannot measure a zero-norm pair.")
    energy = (numerator / denominator).item()
    # print(f"bond energy : {energy}")
    return energy


def simple_update_energy(a_sites: TensorSites, bond_weights: BondWeights, unitcell_coord: UnitCellCoord, h: Tensor) -> float:
    """Estimate the energy per site, counting each r/d bond once."""
    total = 0.0
    for site, direction in bond_weights:
        x, y = site
        if direction == "d":
            neighbor = unitcell_coord(x, y + 1)
        elif direction == "r":
            neighbor = unitcell_coord(x + 1, y)
        a, b = a_sites[site], a_sites[neighbor]
        weights_a = get_site_weights(site, bond_weights, unitcell_coord)
        weights_b = get_site_weights(neighbor, bond_weights, unitcell_coord)
        if direction == "r":
            a = a.permute(1, 2, 3, 0, 4)
            b = b.permute(1, 2, 3, 0, 4)
            weights_a = rotate_site_weights(weights_a)
            weights_b = rotate_site_weights(weights_b)
        total += simple_update_bond_energy(a, b, weights_a, weights_b, h)
    return total / len(a_sites)


def ctmrg_two_site_expectation() -> ...:
    ...


def ctmrg_energy() -> ...:
    ...
