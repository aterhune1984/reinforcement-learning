import numpy as np
import random
from environment import Env
from collections import defaultdict


class QLearningAgent:
    def __init__(self, actions):
        # actions = [0, 1, 2, 3]
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.9
        """ exploration facor ( random actions will be chosen if random number is 
         less than epsilon, so for instance if .2044 is generated and it is less than .9
         it will take a random action instead of from the q table"""
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        """create a queue table dictionary using defaultdict
        defaultdict will assume [0.0, 0.0, 0.0, 0.0] (in this case) 
        as the default value for a key
        if the key does not already exist"""

    # update q function with sample <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        q_1 = self.q_table[state][action]
        # using Bellman Optimality Equation to update q function
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_2 - q_1)

    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state):
        # decide if we are going to take a random action
        # based on the current value of eplison compared to a random number
        rand = np.random.rand()
        if rand > self.epsilon:
            # take random action
            action = np.random.choice(self.actions)
        else:
            # take action according to the q function table
            # create or lookup a key with value of state
            # if state does not have an entry in q_table, it will be
            # initialized with [0.0, 0.0, 0.0, 0.0]
            state_action = self.q_table[state]
            # get the best action to take
            action = self.arg_max(state_action)
        return action

    """ create a list of the largest indexes in state_action and return a random choice
    if there are multiple
    example:
    state_action = [-1.0, 0.0, 0.0, 0.0]
    max_index_list will be [1,2,3] as those are the highest values in the state_action
    a random index(1,2 or 3) will be returned from this function"""
    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        # set max_value to the 0'th item in the list
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            # if the current item number is larger than the
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

if __name__ == "__main__":
    env = Env()  # build game
    agent = QLearningAgent(actions=list(range(env.n_actions)))
    """Call QLearningAgent with actions=[0,1,2,3]"""

    for episode in range(1000):  # do this action 1000 times
        state = env.reset()  # reset the game

        while True:
            env.render()  # update the screen

            # take action and proceed one step in the environment
            action = agent.get_action(str(state))
            next_state, reward, done = env.step(action)

            # with sample <s,a,r,ns'>, agent learns new q function
            agent.learn(str(state), action, reward, str(next_state))

            state = next_state
            env.print_value_all(agent.q_table)

            # if episode ends, then break
            if done:
                break
