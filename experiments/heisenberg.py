"""Apply simple update and CTMRG to the Heisenberg spin-1/2 model on several tilings."""

from dataclasses import replace # noqa: I001
from time import perf_counter

import torch

from variational_tensor_network.initialize import initial_a_sites, initial_bond_weights, initial_env_sites, uniform_boundary
from variational_tensor_network.ctmrg import run_ctmrg_sites
from variational_tensor_network.models.heisenberg import heisenberg_bond
from variational_tensor_network.objects import Config
from variational_tensor_network.observables import simple_update_rdms, ctmrg_rdms, expectation_from_rdm
from variational_tensor_network.peps import absorb_sqrt_weights, double_layer, get_site_weights
from variational_tensor_network.plots import plot_energy_by_D_vs_chi, plot_simple_update, plot_output_path  # noqa: F401
from variational_tensor_network.simple_update import simple_update
from variational_tensor_network.tilings import TILINGS

def format_scientific(value: float, decimals: int = 2) -> str:
    """Better display of float values in terminal."""
    mantissa, exponent = f"{value:.{decimals}e}".split("e")
    superscript = str(int(exponent)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    power = f"10{superscript}"
    return power if float(mantissa) == 1 else f"{mantissa} × {power}"


def run_heisenberg(tiling_name: str, config: Config, chi_values: list[int], schedule: tuple[tuple[float, float], ...]) -> tuple[list[tuple[float, float]], float, list[float]]:
    """Optimize one state and measure its energy at several CTMRG dimensions."""
    sites_coords, unitcell_coord = TILINGS[tiling_name]
    a_sites = initial_a_sites(config, sites_coords)
    bond_weights = initial_bond_weights(a_sites)
    H = heisenberg_bond(J=1.0)
    history = simple_update(a_sites, bond_weights, unitcell_coord, H, schedule=schedule)
    rdms_su = simple_update_rdms(a_sites, bond_weights, unitcell_coord)
    energy_per_site_su = sum(expectation_from_rdm(rho, H).item() for rho in rdms_su.values()) / len(a_sites)
    a_tilde_sites = {}
    for site, a in a_sites.items():
        weights = get_site_weights(site, bond_weights, unitcell_coord)
        a_tilde_sites[site] = absorb_sqrt_weights(a, weights)
    A_sites = {site: double_layer(a_tilde) for site, a_tilde in a_tilde_sites.items()}
    energies = []
    for chi in chi_values:
        config_chi = replace(config, chi=chi)
        delta = uniform_boundary(config_chi)
        env_sites = initial_env_sites(A_sites, delta, config_chi)
        env_sites, _, _ = run_ctmrg_sites(env_sites, A_sites, unitcell_coord, config_chi)
        rdms_ctmrg = ctmrg_rdms(a_tilde_sites, env_sites, unitcell_coord)
        energy = sum(expectation_from_rdm(rho, H).item() for rho in rdms_ctmrg.values()) / len(a_sites)
        energies.append(energy)
    return history, energy_per_site_su, energies


def compare_D(D_values: list[int], tiling_name: str, config: Config, chi_values: list[int], schedule: tuple[tuple[float, float], ...]) -> dict[int, dict]:
    """Collect Heisenberg results at several bond dimensions for one tiling."""
    results_by_D = {}
    for D in D_values:
        config_D = replace(config, D=D)
        history, energy_su, energies = run_heisenberg(tiling_name, config_D, chi_values, schedule)
        results_by_D[D] = {"chi_values": chi_values, "history": history, "energy_su": energy_su, "energies": energies}
    return results_by_D



def main() -> None:
    start = perf_counter()
    torch.manual_seed(0)
    tiling_names = ["checkerboard"] # "checkerboard", "abcd"
    D_values = [2, 3, 4]
    schedule = ((.1, 1e-4), (.01, 1e-5), (.001, 1e-6)) # less precision
    # schedule = ((.1, 1e-5), (.01, 1e-7), (.001, 1e-8)) # more precision
    config = Config()
    D = config.D
    chi_values = [16 + D * i for i in range(D + 1)]
    # chi_values = [16 + 2 * D * i for i in range(D + 1)] # Advised range of values
    # chi_values = [i+1 for i in range(16)] # Do not depends on D, could be bad for 'abcd' because it starts too low. Good to see the curve decreasing for the checkerboard tiling
    for tiling_name in tiling_names:
        print(f"\n\n{tiling_name}\n")
        results_by_D = compare_D(D_values, tiling_name, config, chi_values, schedule)
        energies_by_D = {}
        for D, result in results_by_D.items():
            history = result["history"]
            chi_values_case = result["chi_values"]
            energies = result["energies"]
            print(f"\n\n{tiling_name} | D={D}\n")
            for tau, tolerance in schedule:
                errors = [error for recorded_tau, error in history if recorded_tau == tau]
                print(f"tau={format_scientific(tau)}, tolerance={format_scientific(tolerance)}, sweeps: {len(errors)}")
            print(f"total sweeps: {len(history)}")
            print(f"energy per site simple update : {result['energy_su']:.4g}\n")
            for chi, energy in zip(chi_values_case, energies):
                print(f"chi={chi}, energy per site ctmrg={energy:.4g}")
            energies_by_D[D] = (chi_values_case, energies)
        output_path = plot_output_path(__file__, f"energy_by_D_vs_chi_{tiling_name}.pdf")
        plot_energy_by_D_vs_chi(energies_by_D, title=f"Heisenberg — {tiling_name}", output_path=output_path)
    minutes, seconds = divmod(int(perf_counter() - start), 60)
    hours, minutes = divmod(minutes, 60)
    runtime = f"{hours}h " if hours else ""
    runtime += f"{minutes}min " if minutes else ""
    runtime += f"{seconds}s"
    print(f"\nTotal runtime: {runtime}\n")


if __name__ == "__main__":
    main()



# Legacy code

def compare_tilings(tiling_names: list[str], config: Config, chi_values: list[int], schedule: tuple[tuple[float, float], ...]) -> dict[str, dict]:
    """Collect Heisenberg results for several tilings at fixed bond dimension."""
    results_by_tiling = {}
    for tiling_name in tiling_names:
        history, energy_su, energies = run_heisenberg(tiling_name, config, chi_values, schedule)
        results_by_tiling[tiling_name] = {"chi_values": chi_values, "history": history, "energy_su": energy_su, "energies": energies}
    return results_by_tiling
