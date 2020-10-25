from lib import grid_world
from lib import models

GRID_SHAPE=[4,4]

env = grid_world.GridWorld(grid_shape=GRID_SHAPE, step_cost=-0.1, discount=0.9, start=[0,0])
model = models.Model(GRID_SHAPE, feature_type=1)
agent = grid_world.Agent(grid_shape=GRID_SHAPE, model=model, lr=0.1)
game_starter = grid_world.Game_Starter(env, agent, target_reward=200, print_every=10, clean_history=True)

# play the game
game_starter.start(play_game=True, state_mode='V', agent_mode="AP_TD0", sampling_times=1000, policy_update_times = 1) # state_mode = Q, V ; agent_mode = DP, MC, TD0, AP, AP_TD0