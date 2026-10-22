import torch

from variational_tensor_network import initialize
from variational_tensor_network.moves import UnitCellCoord, all_moves, all_moves_sites
from variational_tensor_network.objects import Config
from variational_tensor_network.peps import double_layer
from variational_tensor_network.unitcell import TILINGS, Site


def check_tilings(sites_coord: tuple[Site, ...], unitcell_coord: UnitCellCoord, config: Config, steps: int = 5) -> None:
    torch.manual_seed(0)
    a = initialize.initial_a(config)
    A = double_layer(a, config)
    delta = initialize.uniform_boundary(config)
    A_sites = {site: A for site in sites_coord}
    env_sites = initialize.initial_env_sites(A_sites, delta, config)
    env_reference = initialize.initial_env(delta, A, config)
    for _ in range(steps):
        env_reference, _ = all_moves(A, env_reference, config)
        env_sites, _ = all_moves_sites(A_sites, env_sites, unitcell_coord, config)
        for env in env_sites.values():
            for actual, expected in zip(env, env_reference):
                torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
    print(f"  OK | uniform reference: {steps} matching left moves")


def main():
    from time import perf_counter

    start = perf_counter()
    config = Config()

    for name, (sites_coord, unitcell_coord) in TILINGS.items():
        print(f"\n[{name}] sites={len(sites_coord)} | D={config.D} | chi={config.chi}", flush=True)
        check_tilings(sites_coord, unitcell_coord, config)

    print(f"\nAll {len(TILINGS)} cases passed in {perf_counter() - start:.2f} s.")

if __name__ == "__main__":
    main()
