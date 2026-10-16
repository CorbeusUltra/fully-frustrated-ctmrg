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

        if error <= threshold:
            return env
    
    raise RuntimeError("Convergence was not reached within max_iter cycles.")


def run_ctmrg_convergence_info(env: Environment = DEFAULT_env, A: Tensor = DEFAULT_A, config: Config = DEFAULT_config) -> tuple[Environment, list[float], list[float]]:
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

    return env, errors, thresholds


def plot_convergence(errors: list[float], thresholds: list[float]) -> None:
    cycles = range(2, len(errors) + 2)
    fig, ax = plt.subplots()
    ax.semilogy(cycles, errors, "o-",  color="blue", label=r"$|\xi_\text{new} - \xi_\text{old}|$")
    ax.semilogy(cycles, thresholds, "-", color="red", label="threshold")
    ax.set(xlabel="number of cycle", ylabel="convergence error")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(outputs_dir / "convergence.pdf")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    env, errors, thresholds = run_ctmrg_convergence_info()
    plot_convergence(errors, thresholds)
