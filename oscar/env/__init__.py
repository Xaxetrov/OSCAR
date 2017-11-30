from gym.envs.registration import register

# abstract env: no need to register it
# register(
#     id='pysc2-v0',
#     entry_point='oscar.env.envs:Pysc2Env',
# )
register(
    id='pysc2-mineralshard-v0',
    entry_point='oscar.env.envs:Pysc2MineralshardEnv',
)
register(
    id='pysc2-mineralshard-v1',
    entry_point='oscar.env.envs:Pysc2MineralshardEnv2',
)
register(
    id='pysc2-simple64-meta-v0',
    entry_point='oscar.env.envs:Pysc2Simple64MetaEnv',
)
