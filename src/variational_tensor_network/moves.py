"""Make one CTMRG move, here on the left of the environment"""

import torch
from torch import Tensor, einsum
from torch.linalg import norm

from .initialize import DEFAULT_A, DEFAULT_env
from .objects import Config, DEFAULT_config, Environment


def left_move(env: Environment = DEFAULT_env, A: Tensor = DEFAULT_A, config: Config = DEFAULT_config) -> tuple[Environment, Tensor]:
    C_ur, C_ul, C_dl, C_dr, T_r, T_u, T_l, T_d = env

    Q_ul = einsum("IA,AJB,iaI,bJaj->ijBb", C_ul, T_u, T_l, A)
    Q_ur = einsum("BKC,CL,cKbk,Lcl->Bbkl", T_u, C_ur, A, T_r)
    Q_dl = einsum("AI,BJA,Iai,bjaJ->ijBb", C_dl, T_d, T_l, A)
    Q_dr = einsum("CKB,LC,ckbK,lcL->Bbkl", T_d, C_dr, A, T_r)

    R = einsum("ijBb,Bbkl->ijkl", Q_ul, Q_ur)
    Rbar = einsum("ijBb,Bbkl->ijkl", Q_dl, Q_dr)

    # # Works but not as efficient as the previous method
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
    p = min(config.chi, rank)
    if rank == 0:
        raise ValueError("M has no numerically significant singular values.")

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


def all_moves(env: Environment = DEFAULT_env, A: Tensor = DEFAULT_A, config: Config = DEFAULT_config) -> tuple[Environment, Tensor]:
    for _ in range(4):
        env, xi = left_move(env, A, config)
        C_ur, C_ul, C_dl, C_dr, T_r, T_u, T_l, T_d = env
        env = Environment(C_dr, C_ur, C_ul, C_dl, T_d, T_r, T_u, T_l)
        A = A.permute(3, 0, 1, 2)
    return env, xi
