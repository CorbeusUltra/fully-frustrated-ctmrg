import matplotlib.pyplot as plt
import torch

from variational_tensor_network.initialize import initial_a_sites, initial_bond_weights
from variational_tensor_network.objects import BondWeights, Config
from variational_tensor_network.peps import get_site_weights
from variational_tensor_network.tilings import TILINGS, Site, UnitCellCoord


def plot_weights(name: str, sites_coords: tuple[Site, ...], unitcell_coord: UnitCellCoord, bond_weights: BondWeights, size: int = 4) -> None:
    """Display a finite patch with site representatives and bond labels."""
    fig, ax = plt.subplots(figsize=(7, 7))
    labels = {id(weight): rf"$\lambda_{{{i}}}$" for i, weight in enumerate(bond_weights.values())}
    for y in range(size):
        for x in range(size):
            site = unitcell_coord(x, y)
            weights = get_site_weights((x, y), bond_weights, unitcell_coord)
            i = sites_coords.index(site)
            color = f"C{i}"
            ax.text(x, y, rf"$a_{{{i}}}$", ha="center", va="center", color="white", bbox={"boxstyle": "round,pad=0.3", "facecolor": color, "edgecolor": "none"}, zorder=3)
            for direction, dx, dy in (("r", 1, 0), ("d", 0, 1)):
                if x + dx >= size or y + dy >= size:
                    continue
                label = labels[id(getattr(weights, direction))]
                ax.plot([x, x + dx], [y, y + dy], color="0.5", zorder=1)
                ax.text(x + dx / 2, y + dy / 2, label, ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, zorder=2)
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(name)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def main() -> None:
    torch.manual_seed(0)
    config = Config()
    for name, (sites_coords, unitcell_coord) in TILINGS.items():
        a_sites = initial_a_sites(config, sites_coords)
        bond_weights = initial_bond_weights(a_sites)
        assert len(bond_weights) == 2 * len(a_sites)
        labels = {id(weight): f"𝜆{i}" for i, weight in enumerate(bond_weights.values())}
        print(f"\n{name}: {len(a_sites)} sites, {len(bond_weights)} bond weights")
        for site in sites_coords:
            print(f"  a[{site}].shape = {tuple(a_sites[site].shape)}") if site == (0,0) else ...
            x, y = site
            weights = get_site_weights(site, bond_weights, unitcell_coord)
            right = get_site_weights((x + 1, y), bond_weights, unitcell_coord)
            down = get_site_weights((x, y + 1), bond_weights, unitcell_coord)
            shared_r = weights.r is right.l
            shared_d = weights.d is down.u
            assert shared_r and shared_d
            names = ", ".join(f"{direction}={labels[id(getattr(weights, direction))]}" for direction in ("r", "u", "l", "d"))
            print(f"  {site}: {names}; shared_r={shared_r}, shared_d={shared_d}")
        plot_weights(name, sites_coords, unitcell_coord, bond_weights)


if __name__ == "__main__":
    main()
