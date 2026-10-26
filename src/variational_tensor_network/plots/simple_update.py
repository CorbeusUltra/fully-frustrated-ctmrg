from pathlib import Path

import matplotlib.pyplot as plt


def plot_simple_update(history: list[tuple[float, float]], output_path: Path, show: bool = False) -> None:
    if not history:
        raise ValueError("History must not be empty.")
    taus, errors = zip(*history)
    sweeps = range(1, len(history) + 1)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), height_ratios=(3, 1))
    axes[0].semilogy(sweeps, errors)
    axes[0].set_ylabel(r"Maximum change in normalized bond weights") # Maximum change in normalized bond weights $\max\|\Delta\widehat{\lambda}\|_\infty$
    axes[0].set_title("Simple-update convergence")
    axes[1].step(sweeps, taus, where="post")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Sweep", ylabel=r"$\tau$")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    if show:
        plt.show()
    plt.close(fig)
