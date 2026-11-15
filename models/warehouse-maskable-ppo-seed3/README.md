# warehouse-maskable-ppo-seed3

Headline agent, seed 3 (MaskablePPO + `SmallGridCNN`, PBRS, 3-stage curriculum, 5M steps), same configuration as `../warehouse-maskable-ppo` (seed 1).

- **Held-out result** (`eval_final_test.json`, final policy, 100 episodes, 12x12 / 3 items): 99% success, efficiency 0.916, mean 63.8 steps vs 54.9 optimal.
- The five headline seeds give efficiency IQM 0.93 [0.92, 0.96]; see `docs/experiments.md`. Action masks are required at inference.
