"""Initialize the bulk states and their environments."""

import torch
from torch import Tensor, einsum
from torch.linalg import norm

from .objects import Config, Environment
from .peps import TensorSites, double_layer
from .unitcell import Site

torch.set_default_dtype(torch.float64)


def initial_a(config: Config) -> Tensor:
    D = config.D
    d = config.d
    a = torch.rand(D,D,D,D,d)
    return a / norm(a)


def initial_sites(config: Config, sites_coords: tuple[Site, ...]) -> tuple[TensorSites, TensorSites]:
    a_sites = {r: initial_a(config) for r in sites_coords}
    A_sites = {r: double_layer(a, config) for r, a in a_sites.items()}
    return a_sites, A_sites


def initial_env(delta: Tensor, A: Tensor, config: Config) -> Environment:
    C0 = einsum("ijkl,k,l->ij", A, delta, delta)
    T0 = einsum("k,ijkl->lij", delta, A)
    D2 = config.D2
    chi = config.chi
    if chi <= D2:
        C = C0[:chi, :chi].clone()
        T = T0[:chi, :, :chi].clone()
    else:
        C = A.new_zeros((chi, chi))
        T = A.new_zeros((chi, D2, chi))
        C[:D2, :D2] = C0
        T[:D2, :, :D2] = T0
    C_norm = norm(C)
    T_norm = norm(T)
    if not torch.isfinite(C_norm) or C_norm <= 0:
        raise ValueError("Cannot normalize C: its norm must be finite and strictly positive.")
    if not torch.isfinite(T_norm) or T_norm <= 0:
        raise ValueError("Cannot normalize T: its norm must be finite and strictly positive.")
    C /= C_norm
    T /= T_norm
    env = Environment(C,C,C,C,T,T,T,T,config)
    # env = Environment(C.clone(),C.clone(),C.clone(),C.clone(),T.clone(),T.clone(),T.clone(),T.clone(),config)
    # Becareful here to indeed replace C's and T's with new tensors and not change them in place. In case of doubt replace with .clone() version.
    return env


def initial_env_sites(A_sites: TensorSites, delta: Tensor, config: Config) -> dict[Site, Environment]:
    env_sites: dict[Site, Environment] = {}
    for r, A in A_sites.items():
        env_sites[r] = initial_env(delta, A, config)
    return env_sites


def uniform_boundary(config: Config) -> Tensor:
    """Assign equal weights to all entries of a fused virtual leg.

    Each ket/bra index pair receives weight 1.
    """
    return torch.ones(config.D2)


def main():
    from .unitcell import TILINGS
    sites_coords, _ = TILINGS["checkerboard"]
    config = Config()
    a_sites, A_sites = initial_sites(config,sites_coords)
    for r in sites_coords:
        print(r, a_sites[r].shape, A_sites[r].shape)
    delta = uniform_boundary(config)
    env_sites = initial_env_sites(A_sites, delta, config)
    print(env_sites)


if __name__ == "__main__":
    main()
