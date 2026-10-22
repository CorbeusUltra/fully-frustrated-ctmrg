from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from .unitcell import Site, UnitCellCoord


def plot_convergence(errors: list[float], thresholds: list[float], output_path: Path, show: bool = False) -> None:
    cycles = range(2, len(errors) + 2)
    fig, ax = plt.subplots()
    ax.semilogy(cycles, errors, "o-", color="blue", label=r"$|\xi_\mathrm{new} - \xi_\mathrm{old}|$")
    ax.semilogy(cycles, thresholds, "--", color="red", label="threshold")
    ax.set(xlabel="number of cycle", ylabel=r"change in $\xi$")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(output_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_convergence_comparison(errors: dict[str, list[float]], thresholds: list[float], output_path: Path, show: bool = False) -> None:
    cycles = range(2, len(thresholds) + 2)
    fig, ax = plt.subplots()
    ax.semilogy(cycles, errors["Single-site"], "k-", label="Single-site")
    for name, style in (("checkerboard", "--"), ("rectangle_2x3", ":")):
        ax.semilogy(cycles, errors[name], style, label=name)
    ax.semilogy(cycles, thresholds, "r--", label="Threshold")
    ax.set(xlabel="number of cycle", ylabel=r"change in $\xi$")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(output_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_convergence_sites(errors: list[Tensor], thresholds: list[Tensor], sites_coord: tuple[Site, ...], unitcell_coord: UnitCellCoord, name: str, output_path: Path, show: bool = False) -> None:
    errors_array = torch.stack(errors).numpy()
    thresholds_array = torch.stack(thresholds).numpy()
    cycles = range(2, len(errors) + 2)

    fig, ax = plt.subplots()
    for i, site in enumerate(sites_coord):
        line, = ax.semilogy(cycles, errors_array[:, i], color=f"C{i}", label=f"Site {i}")
        ax.semilogy(cycles, thresholds_array[:, i], "--", color=line.get_color())

    ax.set(xlabel="number of cycle", ylabel=r"change in $\xi$", title=name)
    ax.legend(loc="upper center")
    ax.grid()
    inset = ax.inset_axes([0.68, 0.55, 0.28, 0.38])
    for y in range(6):
        for x in range(6):
            i = sites_coord.index(unitcell_coord(x, y))
            inset.scatter(x, y, color=f"C{i}", s=20)
    inset.invert_yaxis()
    inset.set_aspect("equal")
    inset.set_axis_off()
    inset.set_title("Tiling", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    if show:
        plt.show()
    plt.close(fig)
