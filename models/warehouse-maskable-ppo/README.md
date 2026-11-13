# warehouse-maskable-ppo

MaskablePPO (sb3-contrib) with the `SmallGridCNN` features extractor, trained on Warehouse-v0.

- **Config:** `config.yaml` (same as `configs/warehouse/maskable_grid.yaml`), seed 1, 5M steps, curriculum 8x8/1 -> 10x10/2 -> 12x12/3.
- **Target:** 12x12 grid, 3 items, capacity 2, grid observation, PBRS shaping, action masks required at inference.
- **Held-out result** (`eval_final_test.json`, 100 episodes, layouts never seen in training): 100% success, efficiency 0.947, mean 59.7 steps vs 54.9 optimal.
- **Caveat:** this is one of two seeds that were used while choosing the configuration; the 5-seed confirmation is in Phase 5. See `docs/experiments.md`.

Load it with `rlagent.agents.factory.load_agent` or `rlagent evaluate`; predict with `rlagent.evaluation.evaluate.predict_action` so the action masks are supplied.
