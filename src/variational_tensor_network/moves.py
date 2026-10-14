"""Make one CTMRG move, here on the left of the environment"""

import torch
from torch import Tensor, einsum, log

from .initialize import DEFAULT_A, DEFAULT_env
from .objects import Config, DEFAULT_config, Environment


def left_move(env: Environment = DEFAULT_env, A: Tensor = DEFAULT_A, config: Config = DEFAULT_config) -> tuple[Environment, Tensor]:
    C_ur, C_ul, C_dl, C_dr, T_r, T_u, T_l, T_d = env

    R        = einsum("IA,AJB,BKC,CL,iaI,bJaj,cKbk,Lcl->ijkl", C_ul, T_u, T_u, C_ur, T_l, A, A, T_r)
    Rbar     = einsum("AI,BJA,CKB,LC,Iai,bjaJ,ckbK,lcL->ijkl", C_dl, T_d, T_d, C_dr, T_l, A, A, T_r)
    RRbar    = einsum("abij,abkl->ijkl",R,Rbar)
    dim_i, dim_j, dim_k, dim_l = RRbar.shape
    r_left   = dim_i * dim_j
    r_right  = dim_k * dim_l
    M = RRbar.reshape(r_left, r_right)
    U, S, V = torch.svd(M)

    # print("R shape:", R.shape)
    # print("Rbar shape:", Rbar.shape)
    # print("RRbar shape:", RRbar.shape)
    # print("M shape:", M.shape)
    # print("\n\nbefore truncation\n")
    # print("U shape:", U.shape)
    # print("V shape:", V.shape)
    # print("S shape:", S.shape)
    # #print(f"S:{S}")

    rtol = max(M.shape) * torch.finfo(S.dtype).eps
    p = min(config.chi, (S > rtol * S[0]).sum().item())

    # print("\n\ntruncation infos\n")
    # print(f"rtol:{rtol}")
    # print(f"chi:{config.chi}")
    # print(f"non-zeros count:{(S > rtol * S[0]).sum().item()}")

    U = U[:, :p]
    S = S[:p]
    V = V[:, :p]

    if len(S)<2:
        raise ValueError("S must have a more than 2 values.")

    xi = 1 / log(S[0]/S[1])
   
    # print("\n\nafter truncation\n")
    # print("U shape:", U.shape)
    # print("V shape:", V.shape)
    # print("S shape:", S.shape)
    # print(f"S:{S}")

    U = U.reshape(dim_i, dim_j, p)
    V = V.reshape(dim_k, dim_l, p)
    invsqrtS = torch.diag(S**(-.5))

    # print("\n\nafter reshape\n")
    # print("U shape:", U.shape)
    # print("V shape:", V.shape)
    # print("S shape:", S.shape)
    # #print(f"S:{S}")

    P    = einsum("ijkl,klm,ms->ijs", R   , U, invsqrtS)
    Pbar = einsum("ijkl,klm,ms->ijs", Rbar, V, invsqrtS)

    # print("\n")
    # print(f"P:{P.shape}")
    # print(f"Pbar:{Pbar.shape}")
    # print("\n")
    # print(f"\n\n T_l:{T_l.shape}\n")

    C_ul_new = einsum("ijk,il,ljm->km",Pbar,C_ul,T_u)
    T_l_new = einsum("ijkl,nkm,mjo,nlp->pio", A, T_l, P, Pbar)
    C_dl_new = einsum("ijk,li,mjl->mk", P, C_dl, T_d)

    C_ul_new = C_ul_new / torch.linalg.vector_norm(C_ul_new)
    T_l_new = T_l_new / torch.linalg.vector_norm(T_l_new)
    C_dl_new = C_dl_new / torch.linalg.vector_norm(C_dl_new)

    env_new = Environment(C_ur, C_ul_new, C_dl_new, C_dr, T_r, T_u, T_l_new, T_d, config)

    return env_new, xi


def all_moves(env: Environment = DEFAULT_env, A: Tensor = DEFAULT_A, config: Config = DEFAULT_config) -> tuple[Environment, Tensor]:
    for _ in range(4):
        env, xi = left_move(env, A, config)
        C_ur, C_ul, C_dl, C_dr, T_r, T_u, T_l, T_d = env
        env = Environment(C_dr, C_ur, C_ul, C_dl, T_d, T_r, T_u, T_l)
        A = A.permute(3, 0, 1, 2)
    return env, xi

