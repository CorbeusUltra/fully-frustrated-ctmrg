"""Apply simple update and CTMRG to the Heisenberg spin 1/2 model on a checkerboard."""

from pathlib import Path  # noqa: I001
from time import perf_counter

import torch

from variational_tensor_network.initialize import initial_a_sites, initial_bond_weights, initial_env_sites, uniform_boundary
from variational_tensor_network.ctmrg import run_ctmrg_sites
from variational_tensor_network.models.heisenberg import heisenberg_bond
from variational_tensor_network.objects import Config
from variational_tensor_network.observables import simple_update_rdms, ctmrg_rdms, expectation_from_rdm
from variational_tensor_network.peps import absorb_sqrt_weights, double_layer, get_site_weights
from variational_tensor_network.plots import plot_simple_update
from variational_tensor_network.simple_update import simple_update
from variational_tensor_network.tilings import TILINGS


def main() -> None:
    total_start = perf_counter()
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "outputs" / "experiments" / "heisenberg_checkerboard" / "latest"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    config = Config()
    print(config)
    sites_coords, unitcell_coord = TILINGS["checkerboard"]
    a_sites = initial_a_sites(config, sites_coords)
    bond_weights = initial_bond_weights(a_sites)
    h = heisenberg_bond(J=1.0)
    schedule = ((0.1, 1e-5), (0.01, 1e-7), (0.001, 1e-8))
    start = perf_counter()
    history = simple_update(a_sites, bond_weights, unitcell_coord, h, schedule=schedule)
    print(f"Simple update: {perf_counter() - start:.3f} s")
    plot_simple_update(history, output_path=output_dir / "convergence.pdf", show=False)
    print(f"\nTotal sweeps: {len(history)}\n")
    for tau, tolerance in schedule:
        errors = [error for recorded_tau, error in history if recorded_tau == tau]
        print(f"tau={tau:.1e}, tolerance={tolerance:.1e}, final error={errors[-1]:.3e}, sweep={len(errors)}")
    rdms_su = simple_update_rdms(a_sites, bond_weights, unitcell_coord)
    energy_per_site_su = sum(expectation_from_rdm(rho, h).item() for rho in rdms_su.values()) / len(a_sites)
    print(f"\nenergy_per_site_su : {energy_per_site_su}")
    a_tilde_sites = {}
    for site, a in a_sites.items():
        weights = get_site_weights(site, bond_weights, unitcell_coord)
        a_tilde_sites[site] = absorb_sqrt_weights(a, weights)
    A_sites = {site: double_layer(a) for site, a in a_tilde_sites.items()}
    delta = uniform_boundary(config)
    env_sites = initial_env_sites(A_sites, delta, config)
    start = perf_counter()
    env_sites, _, _ = run_ctmrg_sites(env_sites, A_sites, unitcell_coord, config)
    print(f"\nCTMRG: {perf_counter() - start:.3f} s")
    rdms_ctmrg = ctmrg_rdms(a_tilde_sites, env_sites, unitcell_coord)
    energy_per_site_ctmrg = sum(expectation_from_rdm(rho, h).item() for rho in rdms_ctmrg.values()) / len(a_sites)
    print(f"\nenergy_per_site_ctmrg : {energy_per_site_ctmrg}")
    print(f"\nTotal runtime: {perf_counter() - total_start:.3f} s\n")



if __name__ == "__main__":
    main()
