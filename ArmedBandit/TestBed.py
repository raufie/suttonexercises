from Environment import ArmedBandit
from Agent import Agent
import time
import numpy as np


class TestBed:
    def __init__(self, n_tests=2000, n_steps=1000):
        self.n_steps = n_steps
        self.n_tests = n_tests
        self.agent = Agent(4, "sample_average")
        self.env = ArmedBandit(4)

    def run_experiment(self, k=4, epsilon=0):

        self.agent = Agent(k, "sample_average")
        self.env = ArmedBandit(k)

        rewards = []
        optimalActions = np.zeros(self.n_steps)

        t = time.time()
        for i in range(self.n_tests):
            rewards.append([])

            self.env.reset_environment()
            self.agent.resetQ()
            print("Run # " + str(i)+"\r", end="")
            for j in range(self.n_steps):
                a = self.agent.get_EGreedy(epsilon)
                r = self.env.step(a)
                rewards[i].append(r)
                optimalActions[j] += 1 if a == np.argmax(
                    self.env.Q_Optimal) else 0
                self.agent.train(a, r)
        print("Experiment Finished - time elapsed: ", time.time() - t)
        return rewards, optimalActions

    def run_experiment2_5(self, k=4, epsilon=0):

        self.agent = Agent(k, "sample_average")
        self.env = ArmedBandit(k)

        rewards = []
        optimalActions = np.zeros(self.n_steps)

        t = time.time()
        for i in range(self.n_tests):
            rewards.append([])

            self.env.reset_environment()
            self.agent.resetQ()
            print("Run # " + str(i)+"\r", end="")
            for j in range(self.n_steps):
                a = self.agent.get_EGreedy(epsilon)
                r = self.env.step(a)
                rewards[i].append(r)
                optimalActions[j] += 1 if a == np.argmax(
                    self.env.Q_Optimal) else 0
                self.agent.train(a, r)

                # random walk
                self.env.Q_Optimal += np.random.normal(0, 0.01, k)
        print("Experiment Finished - time elapsed: ", time.time() - t)
        return rewards, optimalActions

    def run_experiment2_5b(self, k=4, epsilon=0, alpha=0.1):

        self.agent = Agent(k, "exponential_recency_average", alpha=alpha)
        self.env = ArmedBandit(k)

        rewards = []
        optimalActions = np.zeros(self.n_steps)

        t = time.time()
        for i in range(self.n_tests):
            rewards.append([])

            self.env.reset_environment()
            self.agent.resetQ()
            print("Run # " + str(i)+"\r", end="")
            for j in range(self.n_steps):
                a = self.agent.get_EGreedy(epsilon)
                r = self.env.step(a)
                rewards[i].append(r)
                optimalActions[j] += 1 if a == np.argmax(
                    self.env.Q_Optimal) else 0
                self.agent.train(a, r)

                # random walk
                self.env.Q_Optimal += np.random.normal(0, 0.01, k)
        print("Experiment Finished - time elapsed: ", time.time() - t)
        return rewards, optimalActions

    def run_experiment2_6a(self, k=4, epsilon=0, alpha=0.1, Q=None, action="EGreedy"):
        #
        self.agent = Agent(k, "exponential_recency_average", alpha=alpha, Q=Q)
        self.env = ArmedBandit(k)

        rewards = []
        optimalActions = np.zeros(self.n_steps)

        t = time.time()
        for i in range(self.n_tests):
            rewards.append([])

            self.env.reset_environment()
            self.agent.resetQ()
            self.agent.Q = list(Q)
            print("Run # " + str(i)+"\r", end="")
            for j in range(self.n_steps):
                a = self.agent.get_EGreedy(epsilon)
                r = self.env.step(a)
                rewards[i].append(r)
                optimalActions[j] += 1 if a == np.argmax(
                    self.env.Q_Optimal) else 0
                self.agent.train(a, r)

                # random walk
        print("Experiment Finished - time elapsed: ", time.time() - t)
        return rewards, optimalActions

    def run_experiment2_7a(self, k=4, epsilon=0, alpha=0.1, Q=None, action="UCB", constant=2.0):
        #
        self.agent = Agent(k, "sample_average",
                           alpha=alpha, action=action, c=constant, epsilon=constant)
        self.env = ArmedBandit(k)

        rewards = []
        optimalActions = np.zeros(self.n_steps)

        t = time.time()
        for i in range(self.n_tests):
            rewards.append([])

            self.env.reset_environment()
            self.agent.resetQ()

            print("Run # " + str(i)+"\r", end="")
            for j in range(self.n_steps):
                a = self.agent.get_action()
                r = self.env.step(a)
                rewards[i].append(r)
                optimalActions[j] += 1 if a == np.argmax(
                    self.env.Q_Optimal) else 0
                self.agent.train(a, r)

        print("Experiment Finished - time elapsed: ", time.time() - t)
        return rewards, optimalActions

    def run_experiment2_8a(self, k=4, epsilon=0, alpha=0.4, Q=None, action="EGreedy", baseline=False):
        #
        self.agent = Agent(k, method="gradient", alpha=alpha, Q=Q,
                           action="gradient", baseline=baseline)
        self.env = ArmedBandit(k, true_reward=4)

        rewards = []
        optimalActions = np.zeros(self.n_steps)

        t = time.time()
        for i in range(self.n_tests):
            rewards.append([])

            self.env.reset_environment()
            self.agent.resetQ()
            self.agent.Q = list(Q)
            print("Run # " + str(i)+"\r", end="")
            avg_reward = 0
            for j in range(self.n_steps):
                a = self.agent.get_action()
                r = self.env.step(a)
                avg_reward += (1/(j+1))*(r - avg_reward)
                rewards[i].append(r)
                optimalActions[j] += 1 if a == np.argmax(
                    self.env.Q_Optimal) else 0
                self.agent.train(a, r, avg_reward=avg_reward)

                # random walk
        print("Experiment Finished - time elapsed: ", time.time() - t)
        return rewards, optimalActions

    def run_experiment2_9(self, k=4, epsilon=0, alpha=0.4, method="sample_average", Q=None, action="EGreedy", baseline=False, constant=2.0, true_reward=4):
        #
        if method == "sample_average":

            self.agent = Agent(k, method=method, alpha=alpha, Q=Q,
                               action=action, baseline=baseline, epsilon=constant, c=constant
                               )
        self.env = ArmedBandit(k, true_reward=true_reward)

        rewards = []
        optimalActions = np.zeros(self.n_steps)

        t = time.time()
        for i in range(self.n_tests):
            rewards.append([])

            self.env.reset_environment()
            self.agent.resetQ()
            self.agent.Q = list(Q)
            print("Run # " + str(i)+"\r", end="")
            avg_reward = 0
            for j in range(self.n_steps):
                a = self.agent.get_action()
                r = self.env.step(a)
                avg_reward += (1/(j+1))*(r - avg_reward)
                rewards[i].append(r)
                optimalActions[j] += 1 if a == np.argmax(
                    self.env.Q_Optimal) else 0
                self.agent.train(a, r, avg_reward=avg_reward)

                # random walk
        print("Experiment Finished - time elapsed: ", time.time() - t)
        return rewards, optimalActions
