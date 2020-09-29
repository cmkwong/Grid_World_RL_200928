from lib import grid_world

GRID_SHAPE=[4,4]

env = grid_world.GridWorld(grid_shape=GRID_SHAPE, step_cost=0, discount=0.9, start=[0,0])
agent = grid_world.Agent(grid_shape=GRID_SHAPE, discount=0.9, lr=0.1)
game_starter = grid_world.Game_Starter(env, agent, target_reward=100, print_every=10, playGame=False, clean_history=True)

# play the game
game_starter.start()