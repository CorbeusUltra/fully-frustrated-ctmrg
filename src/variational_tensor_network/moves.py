"""Make one CTMRG move, here on the left of the environments."""

import torch
from torch import Tensor, einsum
from torch.linalg import norm

from .objects import Config, Environment
from .unitcell import unitcell_coord

type Site = tuple[int, int]
type TensorSites = dict[Site, Tensor]



def get_halves(site: Site, A_sites: TensorSites, env_sites: dict[Site, Environment]) -> tuple[Tensor, Tensor]:
    # The reference site is the upper-left bulk tensor of the 2x2 patch.
    x, y = site
    def A_at(dx: int, dy: int) -> Tensor:
        return A_sites[unitcell_coord(x + dx, y + dy)]
    def env_at(dx: int, dy: int) -> Environment:
        return env_sites[unitcell_coord(x + dx, y + dy)]
    Q_ul = einsum("IA,AJB,iaI,bJaj->ijBb", env_at(-1,-1).C_ul, env_at(0,-1).T_u, env_at(-1,0).T_l, A_at(0,0))
    Q_ur = einsum("BKC,CL,cKbk,Lcl->Bbkl", env_at(+1,-1).T_u, env_at(+2,-1).C_ur, A_at(+1,0), env_at(+2,0).T_r)
    Q_dl = einsum("AI,BJA,Iai,bjaJ->ijBb", env_at(-1,+2).C_dl, env_at(0,+2).T_d, env_at(-1,+1).T_l, A_at(0,+1))
    Q_dr = einsum("CKB,LC,ckbK,lcL->Bbkl", env_at(+1,+2).T_d, env_at(+2,+2).C_dr, A_at(+1,+1), env_at(+2,+1).T_r)
    R = einsum("ijBb,Bbkl->ijkl", Q_ul, Q_ur)
    Rbar = einsum("ijBb,Bbkl->ijkl", Q_dl, Q_dr)
    return R, Rbar


def get_projectors(R: Tensor, Rbar: Tensor, config: Config) -> tuple[Tensor, Tensor]:
    RRbar    = einsum("abij,abkl->ijkl",R,Rbar)
    dim_i, dim_j, dim_k, dim_l = RRbar.shape
    r_left   = dim_i * dim_j
    r_right  = dim_k * dim_l
    M = RRbar.reshape(r_left, r_right)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    V = Vh.mT
    rtol = max(M.shape) * torch.finfo(S.dtype).eps
    rank = (S > rtol * S[0]).sum().item()
    if rank == 0:
        raise ValueError("M has no numerically significant singular values.")
    p = min(config.chi, rank)
    U, S, V = U[:, :p], S[:p], V[:, :p]
    U = U.reshape(dim_i, dim_j, p)
    V = V.reshape(dim_k, dim_l, p)
    Srsqrt = S.rsqrt()
    P    = einsum("ijkl,klm->ijm", R   , U) * Srsqrt
    Pbar = einsum("ijkl,klm->ijm", Rbar, V) * Srsqrt
    return P, Pbar






def left_move(env: Environment, A: Tensor, config: Config) -> tuple[Environment, Tensor]:
    C_ur, C_ul, C_dl, C_dr, T_r, T_u, T_l, T_d = env
    Q_ul = einsum("IA,AJB,iaI,bJaj->ijBb", C_ul, T_u, T_l, A)
    Q_ur = einsum("BKC,CL,cKbk,Lcl->Bbkl", T_u, C_ur, A, T_r)
    Q_dl = einsum("AI,BJA,Iai,bjaJ->ijBb", C_dl, T_d, T_l, A)
    Q_dr = einsum("CKB,LC,ckbK,lcL->Bbkl", T_d, C_dr, A, T_r)
    R = einsum("ijBb,Bbkl->ijkl", Q_ul, Q_ur)
    Rbar = einsum("ijBb,Bbkl->ijkl", Q_dl, Q_dr)
    # # Works but not as efficient as the previous method using "quarters"
    # R        = einsum("IA,AJB,BKC,CL,iaI,bJaj,cKbk,Lcl->ijkl", C_ul, T_u, T_u, C_ur, T_l, A, A, T_r)
    # Rbar     = einsum("AI,BJA,CKB,LC,Iai,bjaJ,ckbK,lcL->ijkl", C_dl, T_d, T_d, C_dr, T_l, A, A, T_r)
    RRbar    = einsum("abij,abkl->ijkl",R,Rbar)
    dim_i, dim_j, dim_k, dim_l = RRbar.shape
    r_left   = dim_i * dim_j
    r_right  = dim_k * dim_l
    M = RRbar.reshape(r_left, r_right)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    V = Vh.mT
    rtol = max(M.shape) * torch.finfo(S.dtype).eps
    rank = (S > rtol * S[0]).sum().item()
    if rank == 0:
        raise ValueError("M has no numerically significant singular values.")
    p = min(config.chi, rank)
    xi = S.new_zeros(()) if rank == 1 else 1 / torch.log(S[0] / S[1])
    U, S, V = U[:, :p], S[:p], V[:, :p]
    U = U.reshape(dim_i, dim_j, p)
    V = V.reshape(dim_k, dim_l, p)
    Srsqrt = S.rsqrt()
    P    = einsum("ijkl,klm->ijm", R   , U) * Srsqrt
    Pbar = einsum("ijkl,klm->ijm", Rbar, V) * Srsqrt
    C_ul_new = einsum("ijk,il,ljm->km",Pbar,C_ul,T_u)
    T_l_new = einsum("ijkl,nkm,mjo,nlp->pio", A, T_l, P, Pbar)
    C_dl_new = einsum("ijk,li,mjl->mk", P, C_dl, T_d)
    C_ul_new /= norm(C_ul_new)
    T_l_new /= norm(T_l_new)
    C_dl_new /= norm(C_dl_new)
    env_new = Environment(C_ur, C_ul_new, C_dl_new, C_dr, T_r, T_u, T_l_new, T_d, config)
    return env_new, xi


def all_moves(env: Environment, A: Tensor, config: Config) -> tuple[Environment, Tensor]:
    for _ in range(4):
        env, xi = left_move(env, A, config)
        C_ur, C_ul, C_dl, C_dr, T_r, T_u, T_l, T_d = env
        env = Environment(C_dr, C_ur, C_ul, C_dl, T_d, T_r, T_u, T_l)
        A = A.permute(3, 0, 1, 2)
    return env, xi


if __name__ == "__main__":
    from . import initialize
    from .unitcell import sites_coords

    config = Config()
    _, A_sites = initialize.initial_sites(config, sites_coords)
    delta = initialize.uniform_boundary(config)
    env_sites = initialize.initial_env_sites(A_sites, delta, config)
    for site in sites_coords:
        R, Rbar = get_halves(site, A_sites, env_sites)
        P, Pbar = get_projectors(R, Rbar, config)
        assert torch.isfinite(P).all() and torch.isfinite(Pbar).all()
        overlap = einsum("ijp,ijq->pq", P, Pbar)
        identity = torch.eye(P.shape[-1], dtype=P.dtype, device=P.device)
        assert torch.allclose(overlap, identity, atol=1e-8, rtol=1e-8)
        print(site, "OK:", tuple(P.shape), tuple(Pbar.shape))
