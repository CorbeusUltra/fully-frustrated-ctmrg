"""Apply simple update and CTMRG to the Heisenberg spin 1/2 model on a checkerboard."""

from dataclasses import replace # noqa: I001
from time import perf_counter

import torch

from variational_tensor_network.initialize import initial_a_sites, initial_bond_weights, initial_env_sites, uniform_boundary
from variational_tensor_network.ctmrg import run_ctmrg_sites
from variational_tensor_network.models.heisenberg import heisenberg_bond
from variational_tensor_network.objects import Config
from variational_tensor_network.observables import simple_update_rdms, ctmrg_rdms, expectation_from_rdm
from variational_tensor_network.peps import absorb_sqrt_weights, double_layer, get_site_weights
from variational_tensor_network.plots import plot_energy_by_D_vs_chi, plot_simple_update, plot_output_path
from variational_tensor_network.simple_update import simple_update
from variational_tensor_network.tilings import TILINGS

def format_scientific(value: float, decimals: int = 2) -> str:
    mantissa, exponent = f"{value:.{decimals}e}".split("e")
    superscript = str(int(exponent)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    power = f"10{superscript}"
    return power if float(mantissa) == 1 else f"{mantissa} × {power}"


sites_coords, unitcell_coord = TILINGS["checkerboard"]

def main() -> None:
    total_start = perf_counter()
    torch.manual_seed(0)
    config = Config(D=2)
    D = config.D
    D_values = [D + i for i in range(3)]
    energies_by_D = {}
    for D in D_values:
        config = replace(config, D=D)
        chi_values = [D + i for i in range(8)]
        # chi_values = [16 + 2*D*i for i in range(D+1)]
        a_sites = initial_a_sites(config, sites_coords)
        bond_weights = initial_bond_weights(a_sites)
        h = heisenberg_bond(J=1.0)
        schedule = ((0.1, 1e-5), (0.01, 1e-7), (0.001, 1e-8))
        start = perf_counter()
        history = simple_update(a_sites, bond_weights, unitcell_coord, h, schedule=schedule)
        print(f"\n\nD={D}\nSimple update: {perf_counter() - start:.2g} s\n")
        output_path = plot_output_path(__file__, f"convergence_D={D}.pdf")
        plot_simple_update(history, output_path=output_path)
        for tau, tolerance in schedule:
            errors = [error for recorded_tau, error in history if recorded_tau == tau]
            print(f"tau={format_scientific(tau)}, tolerance={format_scientific(tolerance)}, final error={format_scientific(errors[-1])}, sweeps={len(errors)}")
        print(f"total sweeps: {len(history)}")
        rdms_su = simple_update_rdms(a_sites, bond_weights, unitcell_coord)
        energy_per_site_su = sum(expectation_from_rdm(rho, h).item() for rho in rdms_su.values()) / len(a_sites)
        print(f"energy per site simple update : {energy_per_site_su:.4g}\n")
        a_tilde_sites = {}
        for site, a in a_sites.items():
            weights = get_site_weights(site, bond_weights, unitcell_coord)
            a_tilde_sites[site] = absorb_sqrt_weights(a, weights)
        A_sites = {site: double_layer(a) for site, a in a_tilde_sites.items()}
        energies = []
        for chi in chi_values:
            config = replace(config, chi=chi)
            delta = uniform_boundary(config)
            env_sites = initial_env_sites(A_sites, delta, config)
            start = perf_counter()
            env_sites, _, _ = run_ctmrg_sites(env_sites, A_sites, unitcell_coord, config)
            rdms_ctmrg = ctmrg_rdms(a_tilde_sites, env_sites, unitcell_coord)
            energy = sum(expectation_from_rdm(rho, h).item() for rho in rdms_ctmrg.values()) / len(a_sites)
            energies.append(energy)
            print(f"chi={chi}, time : {perf_counter() - start:.2g} s") # energy per site ctmrg={energy:.4g}
        energies_by_D[D] = (chi_values, energies)
    output_path = plot_output_path(__file__, "energy_by_D_vs_chi.pdf")
    plot_energy_by_D_vs_chi(energies_by_D, output_path=output_path)
    print(f"\nTotal runtime: {perf_counter() - total_start:.2g} s\n")



if __name__ == "__main__":
    main()
