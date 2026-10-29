"""Visualize square-lattice tilings and bond weights."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..objects import BondWeights
from ..peps import get_site_weights
from ..tilings import Site, UnitCellCoord
from ._utils import finish_plot


def _draw_tiling(ax: Axes, sites_coords: tuple[Site, ...], unitcell_coord: UnitCellCoord, size: int) -> None:
    """Draw site representatives and nearest-neighbor bonds on existing axes."""
    for y in range(size):
        for x in range(size):
            site = unitcell_coord(x, y)
            i = sites_coords.index(site)
            color = f"C{i}"
            ax.text(x, y, rf"$a_{{{i}}}$", ha="center", va="center", color="white", bbox={"boxstyle": "round,pad=0.3", "facecolor": color, "edgecolor": "none"}, zorder=3)
            for dx, dy in ((1, 0), (0, 1)):
                if x + dx < size and y + dy < size:
                    ax.plot([x, x + dx], [y, y + dy], color="0.5", zorder=1)
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_axis_off()


def plot_tiling(sites_coords: tuple[Site, ...], unitcell_coord: UnitCellCoord, size: int = 4, title: str = "", show: bool = False, output_path: Path | None = None) -> None:
    """Plot a finite patch of a square-lattice tiling."""
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_tiling(ax, sites_coords, unitcell_coord, size)
    finish_plot(fig, title=title, show=show, output_path=output_path)


def plot_weights(sites_coords: tuple[Site, ...], unitcell_coord: UnitCellCoord, bond_weights: BondWeights, size: int = 4, title: str = "", show: bool = False, output_path: Path | None = None) -> None:
    """Plot a tiling with labels identifying shared bond weights."""
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_tiling(ax, sites_coords, unitcell_coord, size)
    labels = {id(weight): rf"$\lambda_{{{i}}}$" for i, weight in enumerate(bond_weights.values())}
    for y in range(size):
        for x in range(size):
            weights = get_site_weights((x, y), bond_weights, unitcell_coord)
            for direction, dx, dy in (("r", 1, 0), ("d", 0, 1)):
                if x + dx >= size or y + dy >= size:
                    continue
                label = labels[id(getattr(weights, direction))]
                ax.text(x + dx / 2, y + dy / 2, label, ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, zorder=2)
    finish_plot(fig, title=title, show=show, output_path=output_path)
