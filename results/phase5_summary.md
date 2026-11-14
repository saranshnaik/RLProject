### Headline agent (5 seeds, 5M steps)

| Group | Seeds | Success (held-out) | Efficiency (held-out) | Episode length |
|---|---|---|---|---|
| headline | 5 | 1.00 [0.99, 1.00] | 0.93 [0.92, 0.96] | 62 [58, 63] |

### E1: reward shaping (sparse vs naive vs PBRS)

| Group | Seeds | Success (held-out) | Efficiency (held-out) | Episode length |
|---|---|---|---|---|
| e1_sparse | 3 | 0.99 [0.98, 0.99] | 0.81 [0.80, 0.82] | 72 [70, 73] |
| e1_naive | 3 | 1.00 [0.99, 1.00] | 0.83 [0.81, 0.87] | 69 [65, 72] |
| ref3m | 3 | 0.99 [0.97, 1.00] | 0.78 [0.77, 0.79] | 74 [72, 76] |

### E2: PPO vs DQN (both without masking; masked PPO shown for reference; DQN arm not run yet)

| Group | Seeds | Success (held-out) | Efficiency (held-out) | Episode length |
|---|---|---|---|---|
| e6_ppo_penalty | 3 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 308 [308, 308] |
| ref3m | 3 | 0.99 [0.97, 1.00] | 0.78 [0.77, 0.79] | 74 [72, 76] |

### E3: observation design (vector MLP vs grid CNN)

| Group | Seeds | Success (held-out) | Efficiency (held-out) | Episode length |
|---|---|---|---|---|
| e3_vector | 3 | 0.24 [0.20, 0.30] | 0.16 [0.12, 0.21] | 259 [242, 270] |
| ref3m | 3 | 0.99 [0.97, 1.00] | 0.78 [0.77, 0.79] | 74 [72, 76] |

### E4: training-layout diversity (10 / 100 / unlimited)

| Group | Seeds | Success (held-out) | Efficiency (held-out) | Episode length |
|---|---|---|---|---|
| e4_layouts10 | 3 | 0.94 [0.93, 0.96] | 0.80 [0.76, 0.87] | 81 [77, 86] |
| e4_layouts100 | 3 | 1.00 [0.99, 1.00] | 0.80 [0.78, 0.81] | 72 [70, 75] |
| ref3m | 3 | 0.99 [0.97, 1.00] | 0.78 [0.77, 0.79] | 74 [72, 76] |

### E5: curriculum on vs off

| Group | Seeds | Success (held-out) | Efficiency (held-out) | Episode length |
|---|---|---|---|---|
| e5_no_curriculum | 1 | 0.53 (n=1) | 0.33 (n=1) | 201 (n=1) |
| ref3m | 3 | 0.99 [0.97, 1.00] | 0.78 [0.77, 0.79] | 74 [72, 76] |

### E6: invalid-action penalty vs masking

| Group | Seeds | Success (held-out) | Efficiency (held-out) | Episode length |
|---|---|---|---|---|
| e6_ppo_penalty | 3 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 308 [308, 308] |
| ref3m | 3 | 0.99 [0.97, 1.00] | 0.78 [0.77, 0.79] | 74 [72, 76] |
