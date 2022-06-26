import numpy as np


class Agent:
    def __init__(self, action_space, method="sample_average", alpha=0.1, Q=None, action="EGreedy", c=2, epsilon=0.1, baseline=False):
        self.baseline = baseline
        self.c = c
        self.epsilon = epsilon
        self.action = action
        self.action_space = action_space
        self.method = method
        self.Q = np.zeros(action_space) if Q is None else Q

        self.customInitialQ = list(self.Q)

        self.action_count = np.zeros(action_space)
        self.t = 0
        self.N_A = np.zeros(action_space)
        self.alpha = alpha

    def get_EGreedy(self, epsilon=None):
        if epsilon == None:
            epsilon = self.epsilon
        sample = np.random.rand()
        if sample > epsilon:
            return np.random.choice(np.where(self.Q == np.max(self.Q))[0])
        else:
            return np.random.choice(np.arange(self.action_space))

    def get_UCB(self, c=2):
        if np.where(self.action_count == 0.0)[0] != []:

            return np.random.choice(np.where(self.action_count == 0.0)[0])

        UCB_A = self.Q + c * \
            np.sqrt(np.log(self.t+1)/self.action_count+1e-5)
        # print(UCB_A)
        return np.argmax(UCB_A)

    def get_softmax_action(self):
        # ⚠⚠⚠USING Q as preference
        probs = np.exp(self.Q)/np.sum(np.exp(self.Q))
        return np.random.choice(np.arange(int(self.action_space)), p=probs)

    def get_softmax_probability(self, a):
        prob = np.exp(self.Q[a])/np.sum(np.exp(self.Q))
        return prob

    def get_action(self):

        actionTaken = 0
        if self.action == "EGreedy":
            actionTaken = self.get_EGreedy(self.epsilon)
        elif self.action == "UCB":
            actionTaken = self.get_UCB(self.c)
        elif self.action == "gradient":
            actionTaken = self.get_softmax_action()
        return actionTaken

    def train(self, action, reward, avg_reward=0):
        self.action_count[action] += 1
        self.N_A[action] += 1
        if self.method == "sample_average":
            self.apply_sample_average(action, reward)
        elif self.method == "exponential_recency_average":
            self.apply_exponential_average(action, reward)
        elif self.method == "gradient":
            self.apply_gradient_bandit(action, reward, avg_reward)

    def apply_sample_average(self, action, reward):
        self.t += 1
        self.Q[action] = self.Q[action] + \
            (1/self.action_count[action]) * (reward - self.Q[action])

    def apply_exponential_average(self, action, reward):
        self.t += 1
        self.Q[action] = self.Q[action] + \
            self.alpha * (reward - self.Q[action])

    def apply_gradient_bandit(self, action, reward, avg_reward=0):
        self.t += 1
        rewardBaseline = 0
        if self.baseline:
            rewardBaseline = avg_reward

        oneHotVector = np.zeros(self.action_space)
        oneHotVector[action] = 1
        probs = np.exp(self.Q)/np.sum(np.exp(self.Q))
        self.Q = self.Q + self.alpha * \
            (reward - rewardBaseline) * \
            (oneHotVector-probs)

    def resetQ(self):
        self.Q = np.zeros(
            self.action_space) if self.customInitialQ is None else self.customInitialQ
        self.t = 0
        self.action_count = np.zeros(self.action_space)

    def get_confidence_score(self, t, n, q, c):
        t = float(t)
        n = float(n)
        if n == 0 or t == 0 or np.log(n) == 0:
            return 0
        else:
            return c * (np.log(t)/np.log(n))**(0.5)
