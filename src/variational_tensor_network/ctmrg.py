"""Full CTMRG algorithm"""

# import torch
from pathlib import Path

import matplotlib.pyplot as plt
from torch import Tensor

from .initialize import DEFAULT_A, DEFAULT_env
from .moves import all_moves
from .objects import Config, DEFAULT_config, Environment

root = next((p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()), None)

if root is None:
    raise RuntimeError("Could not locate the project root.")

outputs_dir = root / "outputs"

def run_ctmrg(env: Environment = DEFAULT_env, A: Tensor = DEFAULT_A, config: Config = DEFAULT_config) -> Environment:
    conv_atol = 1e-10
    conv_rtol = 1e-10
    max_iter = 100

    env, xi_new = all_moves(env, A, config)

    errors = []
    thresholds = []

    for _ in range(max_iter - 1):
        xi_old = xi_new
        env, xi_new = all_moves(env, A, config)
        error = abs(xi_new - xi_old).item()
        errors.append(error)

        scale = max(abs(xi_old).item(), abs(xi_new).item())

        threshold = conv_atol + conv_rtol * scale
        thresholds.append(threshold)

        print("\n\n")
        print(f"error    = {error:.3e}")
        print(f"conv_tol = {threshold:.3e}")

        # if error <= threshold:
        #     return env

    plt.semilogy(range(2, len(errors) + 2), errors, "o-", color="blue", label=r"$|\xi_\text{new} - \xi_\text{old}|$")
    plt.semilogy(range(2, len(thresholds) + 2), thresholds, "-", color="red", label="threshold")
    plt.legend()
    plt.xlabel("number of cycle")
    plt.ylabel("Convergence error")
    plt.tight_layout()
    plt.grid()
    plt.savefig(outputs_dir / "convergence.pdf")
    # plt.show()

    # raise RuntimeError("Convergence was not reached within max_iter cycles.")



if __name__ == "__main__":
    run_ctmrg()
