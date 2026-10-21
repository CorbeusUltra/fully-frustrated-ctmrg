"""Full CTMRG algorithm."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from .initialize import initial_A, initial_a, initial_env, uniform_boundary
from .moves import all_moves, all_moves_sites
from .objects import Config, Environment

type Site = tuple[int, int]
type TensorSites = dict[Site, Tensor]
type EnvSites = dict[Site, Environment]
type UnitCellCoord = Callable[[int, int], Site]

root = next((p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()), None)

if root is None:
    raise RuntimeError("Could not locate the project root.")

outputs_dir = root / "outputs"

def run_ctmrg(env: Environment, A: Tensor, config: Config, stop_on_convergence: bool = True) -> tuple[Environment, list[float], list[float]]:
    conv_atol = 1e-10
    conv_rtol = 1e-10
    max_iter = 100
    env, xi_new = all_moves(A, env, config)
    errors: list[float] = []
    thresholds: list[float] = []
    for _ in range(max_iter - 1):
        xi_old = xi_new
        env, xi_new = all_moves(A, env, config)
        error = (xi_new - xi_old).abs().max().item()
        scale = max(xi_old.abs().max().item(), xi_new.abs().max().item())
        threshold = conv_atol + conv_rtol * scale
        errors.append(error)
        thresholds.append(threshold)
        if stop_on_convergence and error <= threshold:
            return env, errors, thresholds
    if not stop_on_convergence:
        return env, errors, thresholds
    raise RuntimeError("Convergence was not reached within max_iter cycles.")


def run_ctmrg_sites(env_sites: EnvSites, A_sites: TensorSites, unitcell_coord: UnitCellCoord, config: Config, stop_on_convergence: bool = True) -> tuple[EnvSites, list[Tensor], list[Tensor]]:
    conv_atol = 1e-10
    conv_rtol = 1e-10
    max_iter = 100
    env_sites, xi_new = all_moves_sites(A_sites, env_sites, unitcell_coord, config)
    errors: list[Tensor] = []
    thresholds: list[Tensor] = []
    for _ in range(max_iter - 1):
        xi_old = xi_new
        env_sites, xi_new = all_moves_sites(A_sites, env_sites, unitcell_coord, config)
        error = (xi_new - xi_old).abs().amax(dim=0)
        scale = torch.maximum(xi_old.abs().amax(dim=0), xi_new.abs().amax(dim=0))
        threshold = conv_atol + conv_rtol * scale
        errors.append(error.detach().cpu())
        thresholds.append(threshold.detach().cpu())
        if stop_on_convergence and (error <= threshold).all().item():
            return env_sites, errors, thresholds
    if not stop_on_convergence:
        return env_sites, errors, thresholds
    raise RuntimeError("Convergence was not reached within max_iter cycles.")



def plot_convergence(errors: list[float], thresholds: list[float]) -> None:
    cycles = range(2, len(errors) + 2)
    fig, ax = plt.subplots()
    ax.semilogy(cycles, errors, "o-", color="blue", label=r"$|\xi_\mathrm{new} - \xi_\mathrm{old}|$")
    ax.semilogy(cycles, thresholds, "--", color="red", label="threshold")
    ax.set(xlabel="number of cycle", ylabel=r"change in $\xi$")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(outputs_dir / "ctmrg_single_site.pdf")
    # plt.show()
    plt.close(fig)


def plot_convergence_comparison(errors: dict[str, list[float]], thresholds: list[float]) -> None:
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
    fig.savefig(outputs_dir / "ctmrg_uniform_comparison.pdf")
    # plt.show()
    plt.close(fig)


def plot_convergence_sites(errors: list[Tensor], thresholds: list[Tensor], sites_coord: tuple[Site, ...], unitcell_coord: UnitCellCoord, name: str) -> None:
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
    fig.savefig(outputs_dir / f"ctmrg_sites_{name}.pdf")
    # plt.show()
    plt.close(fig)



if __name__ == "__main__":
    from .initialize import initial_env_sites, initial_sites
    from .unitcell import TILINGS
    config = Config()
    torch.manual_seed(0)
    A = initial_A(initial_a(config), config)
    delta = uniform_boundary(config)
    env = initial_env(delta, A, config)
    _, reference_errors, reference_thresholds = run_ctmrg(env, A, config, stop_on_convergence=False)
    errors: dict[str, list[float]] = {"Single-site": reference_errors}
    thresholds = reference_thresholds
    for name in ("checkerboard", "rectangle_2x3"):
        sites_coord, unitcell_coord = TILINGS[name]
        A_sites = {site: A for site in sites_coord}
        env_sites = initial_env_sites(A_sites, delta, config)
        _, site_errors, _ = run_ctmrg_sites(env_sites, A_sites, unitcell_coord, config, stop_on_convergence=False)
        errors[name] = torch.stack(site_errors).amax(dim=1).tolist()
    plot_convergence(reference_errors, reference_thresholds)
    plot_convergence_comparison(errors, thresholds)
    for name in ("checkerboard", "rectangle_2x3"):
        sites_coord, unitcell_coord = TILINGS[name]
        torch.manual_seed(0)
        _, A_sites = initial_sites(config, sites_coord)
        env_sites = initial_env_sites(A_sites, delta, config)
        _, site_errors, site_thresholds = run_ctmrg_sites(env_sites, A_sites, unitcell_coord, config, stop_on_convergence=False)
        plot_convergence_sites(site_errors, site_thresholds, sites_coord, unitcell_coord, name)
