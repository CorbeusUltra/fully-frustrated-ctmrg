"""Full CTMRG algorithm."""

import torch
from torch import Tensor

from .moves import all_moves, all_moves_sites
from .objects import Config, Environment, EnvSites
from .peps import TensorSites
from .unitcell import UnitCellCoord


def run_ctmrg(env: Environment, A: Tensor, config: Config, *, stop_on_convergence: bool = True, max_iter: int = 100, conv_atol: float = 1e-10, conv_rtol: float = 1e-10) -> tuple[Environment, list[float], list[float]]:
    if max_iter < 2:
        raise ValueError("max_iter must be at least 2.")
    env, xi_new = all_moves(A, env, config)
    errors: list[float] = []
    thresholds: list[float] = []
    for _ in range(max_iter - 1):
        xi_old = xi_new
        env, xi_new = all_moves(A, env, config)
        error = (xi_new - xi_old).abs().max().item()
        scale = max(xi_old.abs().max().item(), xi_new.abs().max().item())
        threshold = conv_atol + conv_rtol * scale
        errors.append(error)
        thresholds.append(threshold)
        if stop_on_convergence and error <= threshold:
            return env, errors, thresholds
    if not stop_on_convergence:
        return env, errors, thresholds
    raise RuntimeError("Convergence was not reached within max_iter cycles.")


def run_ctmrg_sites(env_sites: EnvSites, A_sites: TensorSites, unitcell_coord: UnitCellCoord, config: Config, *, stop_on_convergence: bool = True, max_iter: int = 100, conv_atol: float = 1e-10, conv_rtol: float = 1e-10) -> tuple[EnvSites, list[Tensor], list[Tensor]]:
    if max_iter < 2:
        raise ValueError("max_iter must be at least 2.")
    env_sites, xi_new = all_moves_sites(A_sites, env_sites, unitcell_coord, config)
    errors: list[Tensor] = []
    thresholds: list[Tensor] = []
    for _ in range(max_iter - 1):
        xi_old = xi_new
        env_sites, xi_new = all_moves_sites(A_sites, env_sites, unitcell_coord, config)
        error = (xi_new - xi_old).abs().amax(dim=0)
        scale = torch.maximum(xi_old.abs().amax(dim=0), xi_new.abs().amax(dim=0))
        threshold = conv_atol + conv_rtol * scale
        errors.append(error.detach().cpu())
        thresholds.append(threshold.detach().cpu())
        if stop_on_convergence and (error <= threshold).all().item():
            return env_sites, errors, thresholds
    if not stop_on_convergence:
        return env_sites, errors, thresholds
    raise RuntimeError("Convergence was not reached within max_iter cycles.")



if __name__ == "__main__":
    ...
