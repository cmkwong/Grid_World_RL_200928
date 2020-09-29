import numpy as np

class GridWorld:
    def __init__(self, grid_shape, step_cost, discount, start):
        self.rows = grid_shape[0] # 4
        self.cols = grid_shape[1] # 4
        self.i = start[0]
        self.j = start[1]
        self.step_cost = step_cost
        self.discount = discount
        self.action_table = np.zeros((self.rows, self.cols), dtype=list)
        self.reward_table = np.zeros((self.rows, self.cols), dtype=float)
        self.value_table = np.zeros((self.rows, self.cols), dtype=float)
        self.create_grid()

    def create_grid(self):
        """
        return arr(action_table), arr(reward_table), arr(value_table)
        """
        self.action_table[0, 0], self.reward_table[0,0] = ['r', 'd'], self.step_cost
        self.action_table[1, 0], self.reward_table[1,0] = ['u', 'd'], self.step_cost
        self.action_table[2, 0], self.reward_table[2,0] = ['u', 'r', 'd'], self.step_cost
        self.action_table[3, 0], self.reward_table[3,0] = ['u', 'r'], self.step_cost
        self.action_table[0, 1], self.reward_table[0,1] = ['r', 'l'], self.step_cost
        self.action_table[1, 1], self.reward_table[1,1] = [], 0
        self.action_table[2, 1], self.reward_table[2,1] = ['r', 'd', 'l'], self.step_cost
        self.action_table[3, 1], self.reward_table[3,1] = ['u', 'r', 'l'], self.step_cost
        self.action_table[0, 2], self.reward_table[0,2] = ['r', 'd', 'l'], self.step_cost
        self.action_table[1, 2], self.reward_table[1,2] = ['u', 'r', 'd'], self.step_cost
        self.action_table[2, 2], self.reward_table[2,2] = ['u', 'r', 'd', 'l'], self.step_cost
        self.action_table[3, 2], self.reward_table[3,2] = ['u', 'r', 'l'], self.step_cost
        self.action_table[0, 3], self.reward_table[0,3] = ['d', 'l'], self.step_cost
        self.action_table[1, 3], self.reward_table[1,3] = ['u', 'd', 'l'], self.step_cost
        self.action_table[2, 3], self.reward_table[2,3] = [], -1
        self.action_table[3, 3], self.reward_table[3,3] = [], 1

        print("Grid Created!")

    def get_pos(self, i, j, action):
        target_i, target_j = i, j
        if action in self.action_table[i, j]:
            if action == 'u':
                target_i = i - 1
            elif action == 'r':
                target_j = j + 1
            elif action == 'd':
                target_i = i + 1
            elif action == 'l':
                target_j = j - 1
        return target_i, target_j

class Agent:
    def __init__(self, grid_shape, discount=0.9, lr=0.1):
        self.lr = lr
        self.policy = np.zeros((grid_shape[0],grid_shape[1]), dtype=list)
        self.discount = discount
        self.init_policy()

    def init_policy_2(self, action_table):
        """
        According to the action table, init the policy
        return: policy
        """
        policy = np.zeros((action_table.shape), dtype=list)
        for i in range(policy.shape[0]):
            for j in range(policy.shape[1]):
                policy[i, j] = []
                total = len(action_table[i,j])
                for t in range(total):
                    policy[i,j].append(1/total)
        return policy

    def init_policy(self):
        """
        :parms: int = max_i, max_j
        :return: init policy
        """
        for i in range(self.policy.shape[0]):
            for j in range(self.policy.shape[1]):
                self.policy[i,j] = [1/4]*4 # u, r, d, l

    def get_action(self, position):
        """
        :params: curr_i, curr_j
        return str=action
        """
        action_labels = ['u','r','d','l']
        prob_dist = self.policy[position[0], position[1]]
        action_index = list(np.random.multinomial(1, prob_dist, size=1).reshape(-1,)).index(1)
        return action_labels[action_index]

def print_value_grid(value_table):
    for i in range(value_table.shape[0]):
        rows = ''
        for j in range(value_table.shape[1]):
            rows = rows + str(round(value_table[i,j], 4)) + '   '
        print(rows)

def roundList(element, precious=2):
    return round(element, precious)

def print_policy_grid(policy_table):
    for i in range(policy_table.shape[0]):
        rows = ''
        for j in range(policy_table.shape[1]):
            rounded_list = list(map(roundList, policy_table[i,j]))
            rows = rows + str(rounded_list) + ' '
        print(rows)

class Game_Starter:
    def __init__(self, env, agent, target_reward, print_every, playGame, clean_history=True):
        self.env = env
        self.agent = agent
        self.target_reward = target_reward
        self.print_every = print_every
        self.playGame = playGame
        self.clean_history = clean_history


    def play(self, pos):
        i, j = pos[0], pos[1]
        step = 0
        reward = 0
        goal = False
        while True:
            action = self.agent.get_action([i,j])
            i, j = self.env.get_pos(i, j, action)
            step += 1
            reward += self.env.reward_table[i, j]

            # finish the game when reach the final point
            if i == 2 and j == 3:
                break
            elif i == 3 and j ==3:
                goal = True
                break

        return step, reward, goal

    def update_state_value(self):
        actions_label = ['u', 'r', 'd', 'l']
        for i in range(self.env.rows):
            for j in range(self.env.cols):
                if not (i == 3 and j == 3) and not (i == 2 and j == 3):
                    value_state = 0
                    for ii, action_prob in enumerate(self.agent.policy[i,j]):
                        t_i, t_j = self.env.get_pos(i,j,actions_label[ii])
                        value_state += action_prob * (self.env.reward_table[t_i,t_j] + self.env.discount * self.env.value_table[t_i,t_j])
                    self.env.value_table[i,j] = value_state

    def update_policy(self):
        """

        """
        max_i, max_j = self.agent.policy.shape[0], self.agent.policy.shape[1]
        action_labels = ['u','r','d','l']
        for i in range(max_i):
            for j in range(max_j):
                state_values = []
                for action_label in action_labels:
                    t_i, t_j = self.env.get_pos(i,j, action_label)
                    # if t_i == 2 and t_j == 3:
                    #     print("")
                    state_values.append(self.env.reward_table[t_i,t_j] + self.env.discount * self.env.value_table[t_i,t_j])
                argmax_actions = list(np.argwhere(state_values == np.max(state_values)).reshape(-1,))
                # update the policy
                for argmax_action in argmax_actions:
                    self.agent.policy[i,j][argmax_action] += (self.agent.lr) / len(argmax_actions)
                self.agent.policy[i,j] = list(self.agent.policy[i,j] / np.sum(self.agent.policy[i,j]))

    def start(self):
        pos = [self.env.i, self.env.j]
        steps = []
        goal_count = 0
        total_reward = 0
        play_count = 0
        while True:
            # play the game
            if self.playGame:
                step, reward, goal = self.play(pos)
            else:
                step, reward, goal = 0.0,0.0,0.0
            total_reward += reward
            steps.append(step)
            if goal:
                goal_count += 1
            play_count += 1

            # cal the value of state
            self.update_state_value()
            # update the policy according to the updated value of state
            self.update_policy()

            # print every required steps
            if play_count % self.print_every == 0:
                print_value_grid(self.env.value_table)
                print_policy_grid(self.agent.policy)
                mean_steps = float(np.mean(steps[-self.print_every:]))
                completed_goal = (goal_count / play_count) * 100
                reward_per_play = total_reward / play_count
                print("%d plays - mean step: %.2f; completed goal: %.2f; reward per play: %.2f" %
                      (play_count, mean_steps, completed_goal, reward_per_play))

                if self.clean_history:
                    steps = []




