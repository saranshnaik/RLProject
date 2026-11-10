"""Custom environments. Importing this package registers them with Gymnasium."""

import gymnasium as gym

WAREHOUSE_ID = "Warehouse-v0"

if WAREHOUSE_ID not in gym.registry:
    gym.register(id=WAREHOUSE_ID, entry_point="rlagent.envs.warehouse.env:WarehouseEnv")
