"""Do a simple update."""

import torch
from torch import Tensor, einsum
from torch.linalg import vector_norm

from .objects import BondWeights, Config, SiteWeights
from .peps import TensorSites, get_site_weights
from .unitcell import UnitCellCoord


def inverse_weights(weights: Tensor, rtol: float = 1e-12) -> Tensor:
    scale = weights.max()
    if scale == 0:
        raise ValueError("Cannot invert an identically zero bond weight.")
    mask = weights > rtol * scale
    inverse = torch.zeros_like(weights)
    inverse[mask] = 1 / weights[mask]
    return inverse


def simple_update_iteration(a: Tensor, b: Tensor, weights_a: SiteWeights, weights_b: SiteWeights, G: Tensor, config: Config) -> tuple[Tensor,Tensor,Tensor]:
    """
    Assuming the common weight is the down weight of a and the up weight of b.
    """
    abar = einsum("rulds,r,u,l->rulds", a, weights_a.r, weights_a.u, weights_a.l)
    bbar = einsum("rulds,r,l,d->rulds", b, weights_b.r, weights_b.l, weights_b.d)
    torch.testing.assert_close(weights_a.d, weights_b.u)
    abarbbar = einsum("rulqs,RqLDS,q->rulLDRsS",abar,bbar,weights_a.d)
    M = einsum("rulLDRsS,tTsS->rulLDRtT", abarbbar, G) # M or abarbbar_gated
    D, d = config.D, config.d
    M = M.permute(0,1,2,6,3,4,5,7)
    M = M.reshape(D**3 * d, D**3 * d)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    V = Vh.mT
    U, S, V = U[:, :D], S[:D], V[:, :D]
    U, V = U.reshape(D,D,D,d,D), V.reshape(D,D,D,d,D)
    U = U.permute(0,1,2,4,3)
    V = V.permute(2,4,0,1,3)
    a_new = einsum("rulds,r,u,l->rulds", U, inverse_weights(weights_a.r), inverse_weights(weights_a.u), inverse_weights(weights_a.l))
    b_new = einsum("rulds,r,l,d->rulds", V, inverse_weights(weights_b.r), inverse_weights(weights_b.l), inverse_weights(weights_b.d))
    a_new /= vector_norm(a_new)
    b_new /= vector_norm(b_new)
    S /= vector_norm(S)
    return a_new, b_new, S


def rotate_site_weights(weights: SiteWeights) -> SiteWeights:
    """Reorder weights to match tensor.permute(1, 2, 3, 0, 4)."""
    return SiteWeights(r=weights.u, u=weights.l, l=weights.d, d=weights.r)


def simple_update_sweep(a_sites: TensorSites, bond_weights: BondWeights, unitcell_coord: UnitCellCoord, G: Tensor, config: Config) -> None:
    for site, direction in bond_weights:
        x, y = site
        if direction == "d":
            neighbor = unitcell_coord(x, y + 1)
            a = a_sites[site]
            b = a_sites[neighbor]
            weights_a = get_site_weights(site, bond_weights, unitcell_coord)
            weights_b = get_site_weights(neighbor, bond_weights, unitcell_coord)
            a_new, b_new, lambda_new = simple_update_iteration(a, b, weights_a, weights_b, G, config)
        if direction == "r":
            neighbor = unitcell_coord(x + 1, y)
            a = a_sites[site]
            b = a_sites[neighbor]
            weights_a = get_site_weights(site, bond_weights, unitcell_coord)
            weights_b = get_site_weights(neighbor, bond_weights, unitcell_coord)
            a_rot = a.permute(1, 2, 3, 0, 4)
            b_rot = b.permute(1, 2, 3, 0, 4)
            weights_a_rot = rotate_site_weights(weights_a)
            weights_b_rot = rotate_site_weights(weights_b)
            a_rot_new, b_rot_new, lambda_new = simple_update_iteration(a_rot, b_rot, weights_a_rot, weights_b_rot, G, config)
            a_new = a_rot_new.permute(3, 0, 1, 2, 4)
            b_new = b_rot_new.permute(3, 0, 1, 2, 4)
        a_sites[site] = a_new
        a_sites[neighbor] = b_new
        bond_weights[(site,direction)] = lambda_new


def weights_error(old_weights: BondWeights, new_weights: BondWeights) -> float:
    errors = []
    for bond, new_weight in new_weights.items():
        old_weight = old_weights[bond]
        old_normalized = old_weight / vector_norm(old_weight)
        new_normalized = new_weight / vector_norm(new_weight)
        error = (new_normalized - old_normalized).abs().max().item()
        errors.append(error)
    return max(errors)


def simple_update(a_sites: TensorSites, bond_weights: BondWeights, unitcell_coord: UnitCellCoord, h: Tensor, config: Config, *, schedule: tuple[tuple[float, float], ...], max_sweeps: int = 5000) -> list[tuple[float, float]]:
    history = []
    d = config.d
    for tau, tol in schedule:
        G = torch.linalg.matrix_exp(-tau * h).reshape(d, d, d, d)
        for sweep in range(1, max_sweeps + 1):
            old_weights = {bond: weight.clone() for bond, weight in bond_weights.items()}
            simple_update_sweep(a_sites, bond_weights, unitcell_coord, G, config)
            error = weights_error(old_weights, bond_weights)
            history.append((tau, error))
            print(f"tau={tau:.1e}, sweep={sweep}, error={error:.3e}")
            if error <= tol:
                break
        else:
            raise RuntimeError(f"No convergence at tau={tau} after {max_sweeps} sweeps.")
    return history
