"""Define the spin-1/2 Heisenberg interaction."""


from torch import Tensor


def heisenberg_bond(J: float = 1.0) -> Tensor:
    """Uses the product basis (↑↑,↑↓,↓↑,↓↓)."""
    H = Tensor([
        [1.0,  0.0,  0.0, 0.0],
        [0.0, -1.0,  2.0, 0.0],
        [0.0,  2.0, -1.0, 0.0],
        [0.0,  0.0,  0.0, 1.0],
    ])
    return (J / 4) * H
