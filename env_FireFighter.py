"""
Retrieved from GitHub:
    https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment/blob/master/env_FireFighter/env_FireFighter.py
Modified by Frank Liu with MADDPG algorithm
"""

import numpy as np
import random

class EnvFireFighter(object):
    def __init__(self, house_num):
        self.house_num = house_num
        self.fighter_num = self.house_num - 1
        self.firelevel = []
        for i in range(self.house_num):
            self.firelevel.append(3)

    def step(self, target_list):    # 0 left, 1 right
        for i in range(self.house_num):
            if self.firelevel[i] > 0:
                if self.is_neighbour_on_fire(i):
                    if self.how_many_fighters(i, target_list) == 0:
                        if random.random() < 0.8:
                            self.firelevel[i] = self.firelevel[i] + 1
                    elif self.how_many_fighters(i, target_list) == 1:
                        if random.random() < 0.8:
                            self.firelevel[i] = self.firelevel[i] + 1
                        if random.random() < 0.6:
                            self.firelevel[i] = self.firelevel[i] - 1
                    else:
                        self.firelevel[i] = 0
                else:
                    if self.how_many_fighters(i, target_list) == 0:
                        if random.random() < 0.4:
                            self.firelevel[i] = self.firelevel[i] + 1
                    elif self.how_many_fighters(i, target_list) == 1:
                        if random.random() < 0.4:
                            self.firelevel[i] = self.firelevel[i] + 1
                        self.firelevel[i] = self.firelevel[i] - 1
                    else:
                        self.firelevel[i] = 0
            else:   # no fire
                if self.is_neighbour_on_fire(i):
                    if self.how_many_fighters(i, target_list) == 0:
                        if random.random() < 0.8:
                            self.firelevel[i] = self.firelevel[i] + 1
                    elif self.how_many_fighters(i, target_list) == 1:
                        if random.random() < 0.8:
                            self.firelevel[i] = self.firelevel[i] + 1
                        if random.random() < 0.6:
                            self.firelevel[i] = self.firelevel[i] - 1
                    else:
                        self.firelevel[i] = 0
                else:
                    if self.how_many_fighters(i, target_list) == 0:
                        self.firelevel[i] = 0
                    elif self.how_many_fighters(i, target_list) == 1:
                        self.firelevel[i] = 0
                    else:
                        self.firelevel[i] = 0
        self.regulate_fire()
        reward = 0
        for i in range(self.house_num):
            reward = reward - self.firelevel[i]
        return reward

    def is_neighbour_on_fire(self, index):
        is_on = False
        if index == 0:
            if self.firelevel[1] > 0:
                is_on = True
        elif index == self.house_num - 1:
            if self.firelevel[index - 1] > 0:
                is_on = True
        else:
            if self.firelevel[index - 1] > 0 or self.firelevel[index + 1] > 0:
                is_on = True
        return is_on

    def reset(self):
        self.firelevel = []
        for i in range(self.house_num):
            self.firelevel.append(3)

    def how_many_fighters(self, index, target_list):
        num = 0
        if index == 0:
            if target_list[0] == 0:
                num = num + 1
        elif index == self.house_num - 1:
            if target_list[index - 1] == 1:
                num = num + 1
        else:
            if target_list[index - 1] == 1:
                num = num + 1
            if target_list[index] == 0:
                num = num + 1
        return num

    def regulate_fire(self):
        for i in range(self.house_num):
            if self.firelevel[i] < 0:
                self.firelevel[i] = 0

    def get_obs(self):
        obs = []
        for i in range(self.fighter_num):
            temp = [0, 0]       # [left, right]
            if random.random() < 1 - np.exp(-self.firelevel[i]):
                temp[0] = 1

            if random.random() < 1 - np.exp(-self.firelevel[i + 1]):
                temp[1] = 1
            obs.append(temp)
        return obs
    def print_firelevel(self):
        print("The fire level of houses are" + str(self.firelevel))

if __name__ == "__main__":


    # test by Frank Liu
    num_houses = 8
    env = EnvFireFighter(num_houses)
    target_list = []

    print(env.get_obs())
    for i in range(1000):
        target_list = []
        for fire_fighter in range(num_houses - 1):
            action = np.random.randint(2)
            target_list.append(action)
        print("action taken: " + str(target_list))
        print("reward: " + str(env.step(target_list)))
        env.print_firelevel()
        print("action level seen by agents are: " + str(env.get_obs()) + "\n")


    # FIXME: it's almost always the case that middle houses are taken care of much more




    """
    # test by original document
    def generate_tgt_list(agt_num):
        tgt_list = []
        for i in range(agt_num):
            tgt_list.append(random.randint(0, 1))
        return tgt_list


    env = EnvFireFighter(4)

    max_iter = 100
    for i in range(max_iter):
        print("iter= ", i)
        print("actual fire level: ", env.firelevel)
        print("observed fire level: ", env.get_obs())
        tgt_list = generate_tgt_list(3)
        print("agent target: ", tgt_list)
        reward = env.step(tgt_list)
        print("reward: ", reward)
        print(" ")
    """