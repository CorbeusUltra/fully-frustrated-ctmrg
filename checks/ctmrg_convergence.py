"""Collect and plot CTMRG convergence diagnostics over a fixed number of cycles."""

import torch # noqa: I001
from torch import Tensor

from variational_tensor_network.peps import double_layer
from variational_tensor_network.ctmrg import run_ctmrg, run_ctmrg_sites
from variational_tensor_network.initialize import initial_a, initial_env, initial_env_sites, initial_single_and_double_layer_sites, uniform_boundary
from variational_tensor_network.objects import Config
from variational_tensor_network.plots import plot_convergence, plot_convergence_comparison, plot_convergence_sites, plot_output_path
from variational_tensor_network.tilings import TILINGS, Site, UnitCellCoord


def collect_uniform_convergence(config: Config, max_iter: int = 1000) -> tuple[dict[str, list[float]], list[float]]:
    """Collect single-site and multisite histories for the same uniform state."""
    A = double_layer(initial_a(config))
    delta = uniform_boundary(config)
    env = initial_env(delta, A, config)
    _, reference_errors, thresholds = run_ctmrg(env, A, config, stop_on_convergence=False, max_iter=max_iter)
    errors = {"Single-site": reference_errors}
    for name in ("checkerboard", "rectangle_2x3"):
        sites_coords, unitcell_coord = TILINGS[name]
        A_sites = {site: A for site in sites_coords}
        env_sites = initial_env_sites(A_sites, delta, config)
        _, site_errors, _ = run_ctmrg_sites(env_sites, A_sites, unitcell_coord, config, stop_on_convergence=False, max_iter=max_iter)
        errors[name] = torch.stack(site_errors).amax(dim=1).tolist()
    return errors, thresholds


def collect_convergence_sites(sites_coords: tuple[Site, ...], unitcell_coord: UnitCellCoord, config: Config, max_iter: int = 1000) -> tuple[list[Tensor], list[Tensor]]:
    """Collect per-site CTMRG histories for independently initialized site tensors."""
    _, A_sites = initial_single_and_double_layer_sites(config, sites_coords)
    delta = uniform_boundary(config)
    env_sites = initial_env_sites(A_sites, delta, config)
    _, errors, thresholds = run_ctmrg_sites(env_sites, A_sites, unitcell_coord, config, stop_on_convergence=False, max_iter=max_iter)
    return errors, thresholds



def main() -> None:
    config = Config()
    torch.manual_seed(0)
    errors, thresholds = collect_uniform_convergence(config)
    output_path = plot_output_path(__file__, "single_site.pdf")
    plot_convergence(errors["Single-site"], thresholds, output_path=output_path)
    output_path = plot_output_path(__file__, "uniform_comparison.pdf")
    plot_convergence_comparison(errors, thresholds, output_path=output_path)
    tilings_names = ("checkerboard", "rectangle_2x3")
    for name in tilings_names:
        sites_coords, unitcell_coord = TILINGS[name]
        site_errors, site_thresholds = collect_convergence_sites(sites_coords, unitcell_coord, config)
        output_path = plot_output_path(__file__, f"{name}.pdf")
        plot_convergence_sites(site_errors, site_thresholds, sites_coords, unitcell_coord, name, output_path=output_path)


if __name__ == "__main__":
    main()
