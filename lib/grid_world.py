import numpy as np

ACTION_LABELS = ['u', 'r', 'd', 'l']


def print_value_grid(value_table):
    print("------------Value Table---------------")
    for i in range(value_table.shape[0]):
        rows = ''
        for j in range(value_table.shape[1]):
            rows = rows + str(round(value_table[i, j], 5)) + '   '
        print(rows)

def roundList(element, precious=5):
    return round(element, precious)

def print_action_value_grid(action_value_table):
    print("-------------Action Value Table--------------")
    for i in range(action_value_table.shape[0]):
        rows = ''
        for j in range(action_value_table.shape[1]):
            rounded_list = list(map(roundList, action_value_table[i, j]))
            rows += str(rounded_list) + ' '
        print(rows)

def print_policy_grid(policy_table):
    print("--------------Policy Table-------------")
    for i in range(policy_table.shape[0]):
        rows = ''
        for j in range(policy_table.shape[1]):
            rounded_list = list(map(roundList, policy_table[i, j]))
            rows += str(rounded_list) + ' '
        print(rows)

class GridWorld:
    def __init__(self, grid_shape, step_cost, discount, start):
        self.rows = grid_shape[0]  # 4
        self.cols = grid_shape[1]  # 4
        self.i = start[0]
        self.j = start[1]
        self.step_cost = step_cost
        self.discount = discount
        self.action_table = np.zeros((self.rows, self.cols), dtype=list)
        self.reward_table = np.zeros((self.rows, self.cols), dtype=float)
        self.value_table = np.zeros((self.rows, self.cols), dtype=float)
        self.action_value_table = np.zeros(shape=(self.rows, self.cols), dtype=object)
        self.create_grid()

    def create_grid(self):
        """
        return arr(action_table), arr(reward_table), arr(value_table)
        """
        self.action_table[0, 0], self.reward_table[0, 0] = ['r', 'd'], self.step_cost
        self.action_table[1, 0], self.reward_table[1, 0] = ['u', 'd'], self.step_cost
        self.action_table[2, 0], self.reward_table[2, 0] = ['u', 'r', 'd'], self.step_cost
        self.action_table[3, 0], self.reward_table[3, 0] = ['u', 'r'], self.step_cost
        self.action_table[0, 1], self.reward_table[0, 1] = ['r', 'l'], self.step_cost
        self.action_table[1, 1], self.reward_table[1, 1] = [], 0
        self.action_table[2, 1], self.reward_table[2, 1] = ['r', 'd', 'l'], self.step_cost
        self.action_table[3, 1], self.reward_table[3, 1] = ['u', 'r', 'l'], self.step_cost
        self.action_table[0, 2], self.reward_table[0, 2] = ['r', 'd', 'l'], self.step_cost
        self.action_table[1, 2], self.reward_table[1, 2] = ['u', 'r', 'd'], self.step_cost
        self.action_table[2, 2], self.reward_table[2, 2] = ['u', 'r', 'd', 'l'], self.step_cost
        self.action_table[3, 2], self.reward_table[3, 2] = ['u', 'r', 'l'], self.step_cost
        self.action_table[0, 3], self.reward_table[0, 3] = ['d', 'l'], self.step_cost
        self.action_table[1, 3], self.reward_table[1, 3] = ['u', 'd', 'l'], self.step_cost
        self.action_table[2, 3], self.reward_table[2, 3] = [], -1
        self.action_table[3, 3], self.reward_table[3, 3] = [], 1
        # action value
        for i in range(self.rows):
            for j in range(self.cols):
                self.action_value_table[i,j] = [0] * 4

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
    def __init__(self, grid_shape, lr=0.1):
        self.lr = lr
        self.policy = np.zeros((grid_shape[0], grid_shape[1]), dtype=object)
        self.init_policy()

    def init_policy(self):
        """
        :parms: int = max_i, max_j
        :return: init policy
        """
        for i in range(self.policy.shape[0]):
            for j in range(self.policy.shape[1]):
                self.policy[i, j] = [1 / 4] * 4  # u, r, d, l

    def get_action(self, position):
        """
        :params: curr_i, curr_j
        return str=action
        """
        prob_dist = self.policy[position[0], position[1]]
        action_index = list(np.random.multinomial(1, prob_dist, size=1).reshape(-1, )).index(1)  # select the action based on distribution
        return ACTION_LABELS[action_index]

class Game_Starter:
    def __init__(self, env, agent, target_reward, print_every, clean_history=True):
        self.env = env
        self.agent = agent
        self.target_reward = target_reward
        self.print_every = print_every
        self.clean_history = clean_history

    def play(self, pos):
        i, j = pos[0], pos[1]
        step = 0
        reward = 0
        goal = False
        while True:
            action = self.agent.get_action([i, j])
            i, j = self.env.get_pos(i, j, action)
            step += 1
            reward += self.env.reward_table[i, j]

            # check if finish the game when reach the final point
            if i == 2 and j == 3:
                break
            elif i == 3 and j == 3:
                goal = True
                break

        return step, reward, goal

    def sampling_V(self, pos, sampling_times):

        samples = {"reward": [], "return": []}
        for _ in range(sampling_times):
            c_i, c_j = pos[0], pos[1]
            states_and_rewards = [[(c_i, c_j), 0]]
            while True:
                action = self.agent.get_action([c_i, c_j])
                t_i, t_j = self.env.get_pos(c_i, c_j, action)

                # record the experience
                states_and_rewards.append([(t_i, t_j), self.env.reward_table[t_i, t_j]])

                # check if finish the game anc calculate the returns
                if (t_i == 2 and t_j == 3) or (t_i == 3 and t_j == 3):
                    states_and_returns = []
                    G = 0
                    # calculate the return from rewards
                    for s, r in reversed(states_and_rewards):
                        states_and_returns.append([(s[0], s[1]), G])
                        G = r + self.env.discount * G
                    states_and_returns.reverse()
                    break

                c_i, c_j = t_i, t_j

            # append the samples
            samples["reward"].append(states_and_rewards)
            samples["return"].append(states_and_returns)

        return samples

    def sampling_Q(self, pos, sampling_times):
        samples = {"reward": [], "return": []}
        for _ in range(sampling_times):
            c_i, c_j = pos[0], pos[1]
            states_actions_and_rewards = []
            while True:
                action = self.agent.get_action([c_i, c_j])
                t_i, t_j = self.env.get_pos(c_i, c_j, action)

                # record the experience
                states_actions_and_rewards.append([(c_i, c_j), action, self.env.reward_table[c_i, c_j]])

                # break if the last experience
                if (c_i == 2 and c_j == 3) or (c_i == 3 and c_j == 3):
                    break

                c_i, c_j = t_i, t_j

            states_actions_and_returns = []
            G = 0
            # calculate the return from rewards
            for s, a, r in reversed(states_actions_and_rewards):
                states_actions_and_returns.append([s, a, G])
                G = r + self.env.discount * G
            states_actions_and_returns.reverse()

            # append the samples
            samples["reward"].append(states_actions_and_rewards)
            samples["return"].append(states_actions_and_returns)

        return samples

    def update_state_value(self):
        for i in range(self.env.rows):
            for j in range(self.env.cols):
                if not (i == 3 and j == 3) and not (i == 2 and j == 3):
                    V = 0
                    for ii, action_prob in enumerate(self.agent.policy[i, j]):
                        t_i, t_j = self.env.get_pos(i, j, ACTION_LABELS[ii])
                        # deterministic so the probability is always 1: https://www.udemy.com/course/artificial-intelligence-reinforcement-learning-in-python/learn/lecture/6417712?start=452#notes
                        V += action_prob * (1) * (self.env.reward_table[t_i, t_j] + self.env.discount * self.env.value_table[t_i, t_j])
                    self.env.value_table[i, j] = V

    def update_state_action_value(self):
        for i in range(self.env.rows):
            for j in range(self.env.cols):
                if not (i == 3 and j == 3) and not (i == 2 and j == 3):
                    for a_primary, action in enumerate(ACTION_LABELS):
                        t_i, t_j = self.env.get_pos(i, j, action)
                        G = (1) * self.env.reward_table[t_i, t_j]
                        for a_secondary, action_prob in enumerate(self.agent.policy[t_i, t_j]):
                            G += self.env.discount * action_prob * self.env.action_value_table[t_i, t_j][a_secondary]
                        self.env.action_value_table[i,j][a_primary] = G

    def update_state_value_with_exp(self, experience_samples, mode="MC"): # mode = "MC" / "TD0"
        """
        :params: experience_samples = [samples_reward, samples_return]
                        returns = [[ [s1, r1], [s2, r2], ...] ...]
                        rewards = [[ [s1, G1], [s2, G2], ...] ...]
        """
        if mode == "MC":
            state_and_return = experience_samples["return"]
            all_returns = np.zeros((self.env.rows, self.env.cols), dtype=float)
            all_counts = np.zeros((self.env.rows, self.env.cols), dtype=int)
            updated_states = set()
            for exp in range(len(state_and_return)):
                for s, G in state_and_return[exp]:
                    all_counts[s[0], s[1]] += 1
                    all_returns[s[0], s[1]] = ((all_counts[s[0], s[1]] - 1) * all_returns[(s[0], s[1])] + G) * (1 / all_counts[s[0], s[1]])
                    updated_states.add(s)

            # update the new value of state
            for s in updated_states:
                self.env.value_table[s[0], s[1]] = all_returns[s[0], s[1]]

        elif mode == "TD0":
            state_and_reward = experience_samples["reward"]
            for exp in state_and_reward:
                for t in range(len(exp) - 1):
                    s, _ = exp[t]
                    s2, r = exp[t+1]
                    self.env.value_table[s] = self.env.value_table[s] + self.agent.lr * (r + self.env.discount * self.env.value_table[s2] - self.env.value_table[s])


    def update_state_action_value_with_exp(self, experience_samples, mode="MC"): # mode = "MC" / "TD0"
        """
        :params: experience_samples = [samples_reward, samples_return]
                        samples_reward = [[ [s1, a1, r1], [s2, a2, r2], ...] ...]
                        samples_return = [[ [s1, a1, G1], [s2, a2, G2], ...] ...] where a = action character
        """
        if mode == "MC":
            state_and_return = experience_samples["return"]
            all_returns = np.zeros((self.env.rows, self.env.cols), dtype=object)
            all_counts = np.zeros((self.env.rows, self.env.cols), dtype=object)
            for i in range(self.env.rows):
                for j in range(self.env.cols):
                    all_returns[i,j] = [0] * 4
                    all_counts[i,j] = [0] * 4
            updated_state_actions = set()

            for exp in state_and_return:
                for s, action, G in exp:
                    a = ACTION_LABELS.index(action)
                    all_counts[s[0], s[1]][a] += 1
                    all_returns[s[0], s[1]][a] = ((all_counts[s[0], s[1]][a] - 1) * all_returns[(s[0], s[1])][a] + G) * (1 / all_counts[s[0], s[1]][a])
                    updated_state_actions.add((s,a))

            for s, a in updated_state_actions:
                self.env.action_value_table[s[0], s[1]][a] = all_returns[s[0], s[1]][a]

        elif mode == "TD0":
            state_and_reward = experience_samples["reward"]
            for exp in state_and_reward:
                for t in range(len(exp) - 1):
                    s, action, _ = exp[t]
                    s2, action2, r2 = exp[t+1]
                    self.env.action_value_table[s][ACTION_LABELS.index(action)] = self.env.action_value_table[s][ACTION_LABELS.index(action)] + self.agent.lr * (r2 + self.env.discount * self.env.action_value_table[s2][ACTION_LABELS.index(action2)] - self.env.action_value_table[s][ACTION_LABELS.index(action)])

    def update_policy(self, by='V'): # by='V' / 'Q'
        max_i, max_j = self.agent.policy.shape[0], self.agent.policy.shape[1]
        argmax_actions = None
        for i in range(max_i):
            for j in range(max_j):

                if by == 'V':
                    state_values = []
                    for action_label in ACTION_LABELS:
                        t_i, t_j = self.env.get_pos(i, j, action_label)
                        state_values.append(self.env.reward_table[t_i, t_j] + self.env.discount * self.env.value_table[t_i, t_j])
                    state_values = list(map(roundList, state_values))
                    argmax_actions = list(np.argwhere(state_values == np.max(state_values)).reshape(-1, ))  # there maybe more than one max value

                elif by == 'Q':
                    action_values = list(map(roundList, self.env.action_value_table[i,j]))
                    argmax_actions = list(np.argwhere(action_values == np.max(action_values)).reshape(-1, ))

                # update the policy
                for index in argmax_actions:
                    self.agent.policy[i, j][index] += (self.agent.lr) / len(argmax_actions)
                self.agent.policy[i, j] = list(self.agent.policy[i, j] / np.sum(self.agent.policy[i, j]))

    def start(self, play_game=True, state_mode='V', agent_mode='DP', sampling_times=10, policy_update_times=10): # state_mode = V, Q ; agent_mode = DP, MC, TD0
        pos = [self.env.i, self.env.j]
        steps = []
        goal_count = 0
        total_reward = 0
        play_count = 0
        while True:
            # play the game
            if play_game:
                step, reward, goal = self.play(pos)
            else:
                step, reward, goal = 0.0, 0.0, False
            total_reward += reward
            steps.append(step)
            if goal:
                goal_count += 1
            play_count += 1

            if agent_mode == "MC":
                if state_mode == 'V':
                    # sampling_V
                    experience_samples = self.sampling_V(pos=[0, 0], sampling_times=sampling_times)
                    # cal the value of state
                    self.update_state_value_with_exp(experience_samples=experience_samples, mode="MC")
                elif state_mode == 'Q':
                    for _ in range(policy_update_times):
                        # sampling_Q
                        experience_samples = self.sampling_Q(pos=[0, 0], sampling_times=sampling_times)
                        # cal the action-value of state
                        self.update_state_action_value_with_exp(experience_samples=experience_samples, mode="MC")

            elif agent_mode == "TD0":
                if state_mode == 'V':
                    # sampling_V
                    experience_samples = self.sampling_V(pos=[0, 0], sampling_times=sampling_times)
                    # cal the value of state
                    self.update_state_value_with_exp(experience_samples=experience_samples, mode="TD0")
                elif state_mode == 'Q':
                    # sampling_Q
                    experience_samples = self.sampling_Q(pos=[0, 0], sampling_times=sampling_times)
                    # cal the action-value of state
                    self.update_state_action_value_with_exp(experience_samples=experience_samples, mode="TD0")

            elif agent_mode == "DP":
                if state_mode == 'V':
                    # cal the value of state
                    self.update_state_value()
                elif state_mode == 'Q':
                    # cal the action-value of state
                    self.update_state_action_value()
            # update the policy according to the updated value of state
            self.update_policy(by=state_mode)

            # print every required steps
            if play_count % self.print_every == 0:
                if state_mode == 'V':
                    print_value_grid(self.env.value_table)
                elif state_mode == 'Q':
                    print_action_value_grid(self.env.action_value_table)
                print_policy_grid(self.agent.policy)
                mean_steps = float(np.mean(steps[-self.print_every:]))
                completed_goal = (goal_count / play_count) * 100
                reward_per_play = total_reward / play_count
                print("%d plays - mean step: %.2f; completed goal: %.2f%%; reward per play: %.2f" % (play_count, mean_steps, completed_goal, reward_per_play))
                print('\n= = = = = = = = = = = = = = = = = = = = = = = =')

                if self.clean_history:
                    steps = []

            if total_reward > self.target_reward:
                print('')
                print("-------------------------Goal Reached---------------------------")
                print("----------------------------------------------------------------")
                if state_mode == 'V':
                    print_value_grid(self.env.value_table)
                elif state_mode == 'Q':
                    print_action_value_grid(self.env.action_value_table)
                mean_steps = float(np.mean(steps[-self.print_every:]))
                completed_goal = (goal_count / play_count) * 100
                reward_per_play = total_reward / play_count
                print("%d plays - mean step: %.2f; completed goal: %.2f%%; reward per play: %.2f" % (play_count, mean_steps, completed_goal, reward_per_play))
                break

