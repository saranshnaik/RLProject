"""Exact optimal order-picking plans (subset DP over trips).

With capacity 1 the total cost is order-independent, so the interesting case is capacity > 1,
where choosing which items share a trip and in what order is a small TSP-with-capacity problem.
A trip is: walk station -> items (in order) -> station, picking each item and dropping once.
"""

from __future__ import annotations

from itertools import combinations, permutations

from rlagent.envs.warehouse.layout import Cell, Layout

MAX_EXACT_ITEMS = 10


def trip_cost(layout: Layout, items: list[Cell], order: tuple[int, ...]) -> int:
    path = [layout.station, *(items[i] for i in order), layout.station]
    walking = sum(layout.distance(a, b) for a, b in zip(path, path[1:], strict=False))
    return walking + len(order) + 1  # one pick per item, one drop at the end


def optimal_plan(
    layout: Layout, items: list[Cell], capacity: int
) -> tuple[int, list[tuple[int, ...]]]:
    """Return (minimum number of steps, ordered trips of item indices) starting at the station."""
    k = len(items)
    if k == 0:
        return 0, []
    if k > MAX_EXACT_ITEMS:
        raise ValueError(f"Exact planner supports at most {MAX_EXACT_ITEMS} items, got {k}")

    best_trip: dict[int, tuple[int, tuple[int, ...]]] = {}
    for size in range(1, min(capacity, k) + 1):
        for combo in combinations(range(k), size):
            mask = sum(1 << i for i in combo)
            best_trip[mask] = min(
                (trip_cost(layout, items, order), order) for order in permutations(combo)
            )

    inf = float("inf")
    total = [inf] * (1 << k)
    choice: list[int] = [0] * (1 << k)
    total[0] = 0
    for mask in range(1, 1 << k):
        low = mask & -mask  # the lowest item must be in exactly one trip: fix it to cut symmetry
        rest = mask ^ low
        sub = rest
        while True:
            trip = sub | low
            if trip in best_trip:
                cost = best_trip[trip][0] + total[mask ^ trip]
                if cost < total[mask]:
                    total[mask], choice[mask] = cost, trip
            if sub == 0:
                break
            sub = (sub - 1) & rest

    trips: list[tuple[int, ...]] = []
    mask = (1 << k) - 1
    while mask:
        trip = choice[mask]
        trips.append(best_trip[trip][1])
        mask ^= trip
    return int(total[(1 << k) - 1]), trips
