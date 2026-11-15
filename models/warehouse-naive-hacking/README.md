# warehouse-naive-hacking

A deliberately **bad** agent: MaskablePPO + `SmallGridCNN` trained for 3M steps with the hackable `naive` shaping at `naive_bonus: 1.0` (10x the default), seed 0. It is the reward-hacking demo for experiment E1.

- **Behaviour:** it steps back and forth near its start and never delivers anything. Held-out result (`eval_final_test.json`, 100 episodes): 0% success, 0 items delivered, every episode runs the full 308 steps, mean return about +146.
- **Why:** the naive bonus pays +1 for each step towards the subgoal and nothing for stepping away, so oscillating earns more than finishing. Success and efficiency are the honest metrics; the return is not.
- **Config:** `config.yaml` (the run's resolved config; the `name` field and curriculum are as trained).
- Replay: `docs/figures/warehouse_reward_hacking.gif`, or the dashboard's Reward Lab page. See `docs/experiments.md`, E1.
