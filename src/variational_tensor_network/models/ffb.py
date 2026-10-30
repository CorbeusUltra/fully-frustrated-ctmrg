"""Define the fully-frustrated bilayer model interaction."""


from torch import Tensor, eye, kron


def ffb_bond(J_D: float = 2.0, J: float = 1.0, h: float = 0) -> Tensor:
    """Use the product basis (↑↑, ↑↓, ↓↑, ↓↓) for each dimer."""
    I = eye(4)
    Tz = Tensor([
        [1.0,0.0,0.0, 0.0],
        [0.0,0.0,0.0, 0.0],
        [0.0,0.0,0.0, 0.0],
        [0.0,0.0,0.0,-1.0],
    ])
    Tp = Tensor([
        [0.0,1.0,1.0,0.0],
        [0.0,0.0,0.0,1.0],
        [0.0,0.0,0.0,1.0],
        [0.0,0.0,0.0,0.0],
    ])
    Tm = Tp.mT.contiguous()
    H_Heis = 1/4 * Tensor([
        [1.0,  0.0,  0.0, 0.0],
        [0.0, -1.0,  2.0, 0.0],
        [0.0,  2.0, -1.0, 0.0],
        [0.0,  0.0,  0.0, 1.0],
    ])
    H_J_D = kron(H_Heis, I) + kron(I, H_Heis)
    H_J   = kron(Tz, Tz) + .5 * (kron(Tp, Tm) + kron(Tm, Tp))
    H_h   = kron(Tz, I) + kron(I, Tz)
    return (J_D / 4) * H_J_D + J * H_J + (h / 4) * H_h
