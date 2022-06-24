import numpy as np


class QLearning(object):
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim  # dimension of acgtion
        self.lr = cfg.lr  # learning rate
        self.gamma = cfg.gamma  # 衰减系数
        self.epsilon = 0.001
        self.sample_count = 0
        self.Q_table = np.zeros((state_dim, action_dim))  # Q表格

    def choose_action(self, state):
        ####################### 智能体的决策函数，需要完成Q表格方法（需要完成）#######################
        # 假设epsilon=0.1，下面的操作就是有0.9的概率按Q值表选择最优的，有0.1的概率随机选择动作
        # 随机选动作的意义就是去探索那些可能存在的之前没有发现但是更好的方案/动作/路径
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            Q_list = self.Q_table[state]  # 从Q表中选取状态(或观察值)对应的那一行
            maxQ = np.max(Q_list)  # 获取这一行最大的Q值，可能出现多个相同的最大值

            action_list = np.where(Q_list == maxQ)[0]  # np.where(条件)功能是筛选出满足条件的元素的坐标
            action = np.random.choice(action_list)  # 这里尤其如果最大值出现了多次，随机取一个最大值对应的动作就成

        else:
            action = np.random.choice(self.action_dim)  # e_greedy概率直接从动作空间中随机选取一个动作
        return action

    def update(self, state, action, reward, next_state, done):
        ############################ Q表格的更新方法（需要完成）##################################
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q_table[next_state])
        self.Q_table[state][action] += self.lr * (target_Q - self.Q_table[state][action])

    def save(self, path):
        np.save(path + "Q_table.npy", self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path + "Q_table.npy")
