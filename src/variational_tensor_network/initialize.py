"""Initialize the bulk state and its environment."""

import torch
from torch import Tensor, einsum
from torch.linalg import norm

from .objects import Config, Environment

torch.set_default_dtype(torch.float64)

def initial_a(config: Config) -> Tensor:
    D = config.D
    d = config.d
    a = torch.rand(D,D,D,D,d)
    return a / norm(a)

def initial_A(a: Tensor, config: Config) -> Tensor:
    D2 = config.D2
    A = einsum("ijkls,IJKLs->iIjJkKlL", a, a)
    A = A.reshape(D2,D2,D2,D2)
    return A

def initial_sites(config: Config, sites_coords: tuple[(int,int)]):
    a_sites = {r: initial_a(config) for r in sites_coords}
    A_sites = {r: initial_A(a, config) for r, a in a_sites.items()}
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
        C = torch.zeros(chi, chi)
        T = torch.zeros(chi, D2, chi)

        C[:D2, :D2] = C0
        T[:D2, :, :D2] = T0

    C /= norm(C)
    T /= norm(T)
    
    env = Environment(C,C,C,C,T,T,T,T,config)

    return env

def initial_env_sites(A_sites,sites_coords: tuple[tuple[int, int], ...], delta, config):
    env_sites = dict()
    for r in sites_coords:
        env_sites[r] = initial_env(delta,A_sites[r],config)
    return env_sites
    

def main():
    from .unitcell import sites_coords
    config = Config()
    a_sites, A_sites = initial_sites(config,sites_coords)

    for r in sites_coords:
        print(r, a_sites[r].shape, A_sites[r].shape)

    delta = torch.ones(config.D2)

    env_sites = initial_env_sites(A_sites,sites_coords, delta, config)

    print(env_sites)


if __name__ == "__main__":
    main()
