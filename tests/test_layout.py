from itertools import permutations

import numpy as np
import pytest

from rlagent.envs.warehouse.layout import Layout, bfs_distances, generate_layout
from rlagent.envs.warehouse.planner import optimal_plan, trip_cost


def test_bfs_on_hand_built_grid():
    #  . . .
    #  # # .      (row 1 blocked except the right column)
    #  . . .
    blocked = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
    dist = bfs_distances(blocked, (0, 0))
    assert dist.tolist() == [[0, 1, 2], [-1, -1, 3], [6, 5, 4]]


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("density", [0.0, 0.5, 1.0])
def test_every_walkable_cell_is_reachable(seed, density):
    layout = generate_layout(10, 10, density, seed)
    reach = layout.dist_map(layout.station)
    assert np.all(reach[~layout.shelves] >= 0)
    assert layout.free_cells, "must leave room for items"
    assert layout.station not in layout.free_cells


def test_generation_is_deterministic_and_seed_dependent():
    a = generate_layout(10, 10, 0.7, 5)
    b = generate_layout(10, 10, 0.7, 5)
    c = generate_layout(10, 10, 0.7, 6)
    assert np.array_equal(a.shelves, b.shelves)
    assert not np.array_equal(a.shelves, c.shelves)


def test_border_aisles_stay_open():
    layout = generate_layout(9, 9, 1.0, 0)
    assert not layout.shelves[0].any()
    assert not layout.shelves[-1].any()
    assert not layout.shelves[:, 0].any()


def test_too_small_grid_rejected():
    with pytest.raises(ValueError):
        generate_layout(3, 8, 0.5, 0)


def test_step_towards_follows_a_shortest_path():
    layout = generate_layout(10, 10, 0.6, 3)
    target = layout.free_cells[-1]
    cell = layout.station
    steps = 0
    from rlagent.envs.warehouse.layout import MOVES

    while cell != target:
        action = layout.step_towards(cell, target)
        dy, dx = MOVES[action]
        cell = (cell[0] + dy, cell[1] + dx)
        steps += 1
    assert steps == layout.distance(layout.station, target)


def _brute_force(layout: Layout, items, capacity):
    k = len(items)
    best = float("inf")
    for perm in permutations(range(k)):

        def solve(i, perm=perm):
            if i == k:
                return 0
            return min(
                trip_cost(layout, items, perm[i:j]) + solve(j)
                for j in range(i + 1, min(i + capacity, k) + 1)
            )

        best = min(best, solve(0))
    return best


@pytest.mark.parametrize("capacity", [1, 2, 3])
@pytest.mark.parametrize("seed", range(4))
def test_planner_matches_brute_force(capacity, seed):
    layout = generate_layout(10, 10, 0.6, seed)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(layout.free_cells), size=5, replace=False)
    items = [layout.free_cells[i] for i in idx]
    cost, trips = optimal_plan(layout, items, capacity)
    assert cost == _brute_force(layout, items, capacity)
    assert sorted(i for t in trips for i in t) == list(range(5))
    assert all(len(t) <= capacity for t in trips)
    assert cost == sum(trip_cost(layout, items, t) for t in trips)


def test_capacity_one_cost_is_order_independent():
    layout = generate_layout(10, 10, 0.6, 1)
    items = layout.free_cells[:4]
    expected = sum(2 * layout.distance(layout.station, c) + 2 for c in items)
    assert optimal_plan(layout, items, 1)[0] == expected


def test_more_capacity_never_costs_more():
    layout = generate_layout(12, 12, 0.6, 2)
    items = layout.free_cells[::7][:5]
    costs = [optimal_plan(layout, items, c)[0] for c in (1, 2, 3, 5)]
    assert costs == sorted(costs, reverse=True)
