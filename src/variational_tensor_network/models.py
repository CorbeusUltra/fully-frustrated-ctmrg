"""Define two-site Hamiltonians and their model metadata."""

from torch import Tensor, eye, kron


def heisenberg_bond(J: float = 1.0) -> Tensor:
    """Uses the product basis (↑↑,↑↓,↓↑,↓↓)."""
    H = Tensor([
        [1.0,  0.0,  0.0, 0.0],
        [0.0, -1.0,  2.0, 0.0],
        [0.0,  2.0, -1.0, 0.0],
        [0.0,  0.0,  0.0, 1.0],
    ])
    return (J / 4) * H


def ffb_bond(JD: float = 2.0, J: float = 1.0, h: float = 0) -> Tensor:
    """Use the product basis (↑↑, ↑↓, ↓↑, ↓↓) for each dimer."""
    I = eye(4)
    Tz = Tensor([
        [1.0, 0.0, 0.0,  0.0],
        [0.0, 0.0, 0.0,  0.0],
        [0.0, 0.0, 0.0,  0.0],
        [0.0, 0.0, 0.0, -1.0],
    ])
    Tp = Tensor([
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    Tm = Tp.mT.contiguous()
    H_Heis = heisenberg_bond()
    H_JD = kron(H_Heis, I) + kron(I, H_Heis)
    H_J   = kron(Tz, Tz) + (1/2) * (kron(Tp, Tm) + kron(Tm, Tp))
    H_h   = kron(Tz, I) + kron(I, Tz)
    return (JD/4) * H_JD + J * H_J + (h/4) * H_h


MODELS = {
    "heisenberg": {
        "bond": heisenberg_bond,
        "d": 2,
        "label": "Heisenberg",
    },
    "ffb": {
        "bond": ffb_bond,
        "d": 4,
        "label": "FFB",
    },
}
