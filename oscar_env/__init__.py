
from gym.envs.registration import register

# abstract env: no need to register it
# register(
#     id='pysc2-v0',
#     entry_point='oscar_env.envs:Pysc2Env',
# )
register(
    id='pysc2-mineralshard-v0',
    entry_point='oscar_env.envs:Pysc2MineralshardEnv',
)