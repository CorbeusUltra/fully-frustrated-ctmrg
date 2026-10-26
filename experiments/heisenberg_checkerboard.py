"""Apply a simple update to the Heisenberg spin 1/2 model on a checkerboard"""

from pathlib import Path

import torch

from variational_tensor_network.initialize import initial_a_sites, initial_bond_weights
from variational_tensor_network.models.heisenberg import heisenberg_bond
from variational_tensor_network.objects import Config
from variational_tensor_network.observables import simple_update_energy
from variational_tensor_network.plots import plot_simple_update
from variational_tensor_network.simple_update import simple_update
from variational_tensor_network.unitcell import TILINGS


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "outputs" / "experiments" / "heisenberg_checkerboard" / "latest"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    config = Config()
    sites_coords, unitcell_coord = TILINGS["checkerboard"]
    a_sites = initial_a_sites(config, sites_coords)
    bond_weights = initial_bond_weights(a_sites, config)
    h = heisenberg_bond(J=1.0)
    schedule = ((0.1, 1e-5), (0.01, 1e-7), (0.001, 1e-8))
    history = simple_update(a_sites, bond_weights, unitcell_coord, h, config, schedule=schedule)
    plot_simple_update(history, output_path=output_dir / "convergence.pdf", show=False)
    print(f"\nTotal sweeps: {len(history)}")
    for tau, tolerance in schedule:
        errors = [error for recorded_tau, error in history if recorded_tau == tau]
        print(f"tau={tau:.1e}: {len(errors)} sweeps, final error={errors[-1]:.3e}, tolerance={tolerance:.1e}")
    energy  = simple_update_energy(a_sites,bond_weights,unitcell_coord,h)
    print(f"energy : {energy}")


if __name__ == "__main__":
    main()
