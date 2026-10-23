from datetime import datetime # noqa: I001
from pathlib import Path
import torch

from variational_tensor_network.peps import double_layer
from variational_tensor_network.ctmrg import run_ctmrg, run_ctmrg_sites
from variational_tensor_network.initialize import initial_a, initial_env, initial_env_sites, initial_sites, uniform_boundary
from variational_tensor_network.objects import Config
from variational_tensor_network.plots import plot_convergence, plot_convergence_comparison, plot_convergence_sites
from variational_tensor_network.unitcell import TILINGS


def main(*, archive: bool = False):
    config = Config()
    torch.manual_seed(0)

    root = Path(__file__).resolve().parents[1]
    base_dir = root / "outputs" / "checks" / "ctmrg_uniform"
    if archive:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_dir = base_dir / f"{timestamp}_D={config.D}_chi={config.chi}"
    else:
        output_dir = base_dir / "latest"
    output_dir.mkdir(parents=True, exist_ok=not archive)
    print(f"Saving figures to: {output_dir}")

    A = double_layer(initial_a(config))
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

    plot_convergence(reference_errors, reference_thresholds, output_path=output_dir / "single_site.pdf")
    plot_convergence_comparison(errors, thresholds, output_path=output_dir / "uniform_comparison.pdf")

    for name in ("checkerboard", "rectangle_2x3"):
        sites_coord, unitcell_coord = TILINGS[name]
        _, A_sites = initial_sites(config, sites_coord)
        env_sites = initial_env_sites(A_sites, delta, config)
        _, site_errors, site_thresholds = run_ctmrg_sites(env_sites, A_sites, unitcell_coord, config, stop_on_convergence=False)
        plot_convergence_sites(site_errors, site_thresholds, sites_coord, unitcell_coord, name, output_path=output_dir / f"{name}.pdf")



if __name__ == "__main__":
    main(archive=False)
