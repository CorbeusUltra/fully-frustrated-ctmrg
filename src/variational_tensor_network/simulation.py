"""Run simple-update optimization and CTMRG energy measurements."""

from dataclasses import replace  # noqa: I001

from torch import Tensor

from .ctmrg import run_ctmrg_sites
from .initialize import initial_a_sites, initial_bond_weights, initial_env_sites, uniform_boundary
from .objects import Config
from .observables import simple_update_rdms, ctmrg_rdms, expectation_from_rdm
from .peps import absorb_sqrt_weights, double_layer, get_site_weights
from .simple_update import simple_update
from .tilings import TILINGS


def run_simulation(tiling_name: str, config: Config, H: Tensor, chi_values: list[int], schedule: tuple[tuple[float, float], ...]) -> tuple[list[tuple[float, float]], float, list[float]]:
    """Optimize one state and measure energy per PEPS site at several CTMRG dimensions."""
    sites_coords, unitcell_coord = TILINGS[tiling_name]
    a_sites = initial_a_sites(config, sites_coords)
    bond_weights = initial_bond_weights(a_sites)
    history = simple_update(a_sites, bond_weights, unitcell_coord, H, schedule=schedule)
    rdms_su = simple_update_rdms(a_sites, bond_weights, unitcell_coord)
    energy_per_peps_site_su = sum(expectation_from_rdm(rho, H).item() for rho in rdms_su.values()) / len(a_sites)
    a_tilde_sites = {}
    for site, a in a_sites.items():
        weights = get_site_weights(site, bond_weights, unitcell_coord)
        a_tilde_sites[site] = absorb_sqrt_weights(a, weights)
    A_sites = {site: double_layer(a_tilde) for site, a_tilde in a_tilde_sites.items()}
    energies_per_peps_site_ctmrg = []
    for chi in chi_values:
        config_chi = replace(config, chi=chi)
        delta = uniform_boundary(config_chi)
        env_sites = initial_env_sites(A_sites, delta, config_chi)
        env_sites, _, _ = run_ctmrg_sites(env_sites, A_sites, unitcell_coord, config_chi)
        rdms_ctmrg = ctmrg_rdms(a_tilde_sites, env_sites, unitcell_coord)
        energy_per_peps_site_ctmrg = sum(expectation_from_rdm(rho, H).item() for rho in rdms_ctmrg.values()) / len(a_sites)
        energies_per_peps_site_ctmrg.append(energy_per_peps_site_ctmrg)
    return history, energy_per_peps_site_su, energies_per_peps_site_ctmrg


def run_D_scan(D_values: list[int], tiling_name: str, config: Config, H: Tensor, chi_values_by_D: list[int], schedule: tuple[tuple[float, float], ...]) -> dict[int, dict]:
    """Run simulations at several bond dimensions for one tiling."""
    results_by_D = {}
    for D in D_values:
        chi_values = chi_values_by_D[D]
        config_D = replace(config, D=D)
        print(f"\n\nRunning simple update and ctmrg with {tiling_name} for D={D}\n")
        history, energy_per_peps_site_su, energies_per_peps_site_ctmrg = run_simulation(tiling_name, config_D, H, chi_values, schedule)
        results_by_D[D] = {"chi_values": chi_values, "history": history, "energy_per_peps_site_su": energy_per_peps_site_su, "energies_per_peps_site_ctmrg": energies_per_peps_site_ctmrg}
    return results_by_D



def format_scientific(value: float, decimals: int = 2) -> str:
    """Better display of float values in terminal."""
    mantissa, exponent = f"{value:.{decimals}e}".split("e")
    superscript = str(int(exponent)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    power = f"10{superscript}"
    return power if float(mantissa) == 1 else f"{mantissa} × {power}"


def format_duration(duration: float) -> str:
    """Format a duration in seconds as days, hours, minutes, and seconds."""
    minutes, seconds = divmod(int(duration), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    text = f"{days}d " if days else ""
    text += f"{hours}h " if hours else ""
    text += f"{minutes}m " if minutes else ""
    text += f"{seconds}s" if seconds else ""
    return text.strip() or "<1s"


def print_scan_summary(results_by_D: dict[int, dict], tiling_name: str, schedule: tuple[tuple[float, float], ...]) -> None:
    """Print optimization and energy results for each bond dimension."""
    for D, result in results_by_D.items():
        history = result["history"]
        print(f"\n\n{tiling_name} | D={D}\n")
        for tau, tolerance in schedule:
            errors = [error for recorded_tau, error in history if recorded_tau == tau]
            print(f"tau={format_scientific(tau)}, tolerance={format_scientific(tolerance)}, sweeps: {len(errors)}")
        print(f"total sweeps: {len(history)}")
        print(f"energy per PEPS site simple update : {result['energy_per_peps_site_su']:.12f}\n")
        for chi, energy_per_peps_site_ctmrg in zip(result["chi_values"], result["energies_per_peps_site_ctmrg"]):
            print(f"chi={chi}, energy per PEPS site ctmrg={energy_per_peps_site_ctmrg:.12f}")



# Legacy code

# def compare_tilings(tiling_names: list[str], config: Config, chi_values: list[int], schedule: tuple[tuple[float, float], ...]) -> dict[str, dict]:
#     """Collect Heisenberg results for several tilings at fixed bond dimension."""
#     results_by_tiling = {}
#     for tiling_name in tiling_names:
#         history, energy_su, energies = run_heisenberg(tiling_name, config, chi_values, schedule)
#         results_by_tiling[tiling_name] = {"chi_values": chi_values, "history": history, "energy_su": energy_su, "energies": energies}
#     return results_by_tiling
