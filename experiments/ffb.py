"""Apply simple update and CTMRG to the fully-frustrated bilayer model (ffb). One PEPS site represents one dimer of two spins-1/2."""

from time import perf_counter  # noqa: I001

import torch

from variational_tensor_network.models.ffb import ffb_bond
from variational_tensor_network.objects import Config
from variational_tensor_network.plots import plot_energy_by_D_vs_chi, plot_energy_vs_chi, plot_output_path # noqa: F401
from variational_tensor_network.simulation import run_D_scan, format_duration, print_scan_summary


def main() -> None:
    start = perf_counter()
    torch.manual_seed(0)
    tiling_names = ["checkerboard"] # "checkerboard", "abcd"
    D_values = [2, 3, 4]
    schedule = ((.1, 1e-4), (.01, 1e-5), (.001, 1e-6)) # less precision
    # schedule = ((.1, 1e-5), (.01, 1e-7), (.001, 1e-8)) # more precision
    config = Config(d = 4,D = 3, chi = 5)
    D = config.D
    H = ffb_bond(JD=2.0, J=1.0, h=0.0)
    chi_values = [16 + D * i for i in range(max(D + 1,6))]
    # chi_values = [16 + 2 * D * i for i in range(D + 1)] # Advised range of values
    # chi_values = [i+1 for i in range(16)] # Do not depends on D, could be bad for 'abcd' because it starts too low. Good to see the curve decreasing for the checkerboard tiling
    for tiling_name in tiling_names:
        print(f"\n\n{tiling_name}\n")
        results_by_D = run_D_scan(D_values, tiling_name, config, H, chi_values, schedule)
        print_scan_summary(results_by_D, tiling_name, schedule)
        # energies_by_D = {D: (result["chi_values"], result["energies_per_peps_site_ctmrg"]) for D, result in results_by_D.items()}
        # output_path = plot_output_path(__file__, f"energy_by_D_vs_chi_{tiling_name}.pdf")
        # plot_energy_by_D_vs_chi(energies_by_D, title=f"FFB — {tiling_name}",show=True, output_path=output_path)
        for D, result in results_by_D.items():
            output_path = plot_output_path(__file__, f"energy_vs_chi_{tiling_name}_D={D}.pdf")
            plot_energy_vs_chi(result["chi_values"], result["energies_per_peps_site_ctmrg"], D, title=f"FFB — {tiling_name}", show=True, output_path=output_path)
    print(f"\nTotal runtime: {format_duration(perf_counter() - start)}\n")


if __name__ == "__main__":
    main()
