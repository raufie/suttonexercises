from TestBed import TestBed
import matplotlib.pyplot as plt
import numpy as np
import pickle


def run10ArmedBandits():
    n_tests = 2000
    n_steps = 1000
    testBed = TestBed(n_tests, n_steps)
    rewards1, optimalActions1 = testBed.run_experiment(k=10, epsilon=0.1)
    rewards2, optimalActions2 = testBed.run_experiment(k=10, epsilon=0.01)
    rewards3, optimalActions3 = testBed.run_experiment(k=10, epsilon=0)

    avg1 = getAvgRewards(rewards1, n_tests, n_steps)
    avg2 = getAvgRewards(rewards2, n_tests, n_steps)
    avg3 = getAvgRewards(rewards3, n_tests, n_steps)

    # save_data(avg1, "epsilon0")
    # save_data(avg1, "epsilon0_01")
    # save_data(avg1, "epsilon0_1")
    plt.plot(avg1, 'b', label="$\epsilon = 0.1$")
    plt.plot(avg2, 'r', label="$\epsilon = 0.01$")
    plt.plot(avg3, 'g', label="$\epsilon = 0$")
    plt.xlabel("n steps")
    plt.ylabel("average reward")
    plt.title('Sample Average Method - avg reward')
    plt.legend()
    plt.savefig("2_2 avg.png")
    # plt.show()
    plt.clf()
    # plt.close()
    # plotting avg reward wrt true r vals in percentage

    plt.plot(100*np.array(optimalActions1) / (n_tests),
             'b', label="$\epsilon = 0.1$")
    plt.plot(100*np.array(optimalActions2) / (n_tests),
             'r', label="$\epsilon = 0.01$")
    plt.plot(100*np.array(optimalActions3) / (n_tests),
             'g', label="$\epsilon = 0$")
    plt.xlabel("n steps")
    plt.ylabel("% optimal values")
    plt.title('Sample Average Method - avg reward')
    plt.legend()
    plt.savefig("2_2 optimal actions percentage.png")


def run2_5a():
    n_tests = 2000
    n_steps = 1000
    testBed = TestBed(n_tests, n_steps)
    rewards1, optimalActions1 = testBed.run_experiment2_5(k=10, epsilon=0.1)
    rewards2, optimalActions2 = testBed.run_experiment2_5(k=10, epsilon=0.01)
    rewards3, optimalActions3 = testBed.run_experiment2_5(k=10, epsilon=0)

    avg1 = getAvgRewards(rewards1, n_tests, n_steps)
    avg2 = getAvgRewards(rewards2, n_tests, n_steps)
    avg3 = getAvgRewards(rewards3, n_tests, n_steps)

    # save_data(avg1, "epsilon0")
    # save_data(avg1, "epsilon0_01")
    # save_data(avg1, "epsilon0_1")
    plt.plot(avg1, 'b', label="$\epsilon = 0.1$")
    plt.plot(avg2, 'r', label="$\epsilon = 0.01$")
    plt.plot(avg3, 'g', label="$\epsilon = 0$")
    plt.xlabel("n steps")
    plt.ylabel("average reward")
    plt.title('Sample Average Method - avg reward')
    plt.legend()
    plt.savefig("2_5 avg (sample_avg).png")
    # plt.show()
    plt.clf()
    # plt.close()
    # plotting avg reward wrt true r vals in percentage

    plt.plot(100*np.array(optimalActions1) / (n_tests),
             'b', label="$\epsilon = 0.1$")
    plt.plot(100*np.array(optimalActions2) / (n_tests),
             'r', label="$\epsilon = 0.01$")
    plt.plot(100*np.array(optimalActions3) / (n_tests),
             'g', label="$\epsilon = 0$")
    plt.xlabel("n steps")
    plt.ylabel("% optimal values")
    plt.title('Sample Average Method - avg reward')
    plt.legend()
    plt.savefig("2_5 optimal actions percentage (sample avg).png")


def run2_5b():
    n_tests = 2000
    n_steps = 10000
    testBed = TestBed(n_tests, n_steps)
    rewards1, optimalActions1 = testBed.run_experiment2_5b(
        k=10, epsilon=0.1, alpha=0.1)

    avg1 = getAvgRewards(rewards1, n_tests, n_steps)

    # save_data(avg1, "epsilon0")
    # save_data(avg1, "epsilon0_01")
    # save_data(avg1, "epsilon0_1")
    plt.plot(avg1, 'b', label="$\epsilon = 0.1 \\alpha = 0.1$")

    plt.xlabel("n steps")
    plt.ylabel("average reward")
    plt.title('Exponential Recency Method - avg reward')
    plt.legend()
    plt.savefig("2_5B avg (exponential avg).png")
    # plt.show()
    plt.clf()
    # plt.close()
    # plotting avg reward wrt true r vals in percentage

    plt.plot(100*np.array(optimalActions1) / (n_tests),
             'b', label="$\epsilon = 0.1$")

    plt.xlabel("n steps")
    plt.ylabel("%optimal values")
    plt.title('Exponential Recency Method - avg reward')
    plt.legend()
    plt.savefig("2_5B optimal actions percentage (Expo. Rec).png")


def run2_6a():
    n_tests = 2000
    n_steps = 1000
    testBed = TestBed(n_tests, n_steps)
    rewards1, optimalActions1 = testBed.run_experiment2_6a(
        k=10, epsilon=0.0, alpha=0.1, Q=np.full(10, 5.0))
    rewards2, optimalActions2 = testBed.run_experiment2_6a(
        k=10, epsilon=0.1, alpha=0.1, Q=np.full(10, 0.0))
    avg1 = getAvgRewards(rewards1, n_tests, n_steps)
    avg2 = getAvgRewards(rewards2, n_tests, n_steps)

    # save_data(avg1, "epsilon0")
    # save_data(avg1, "epsilon0_01")
    # save_data(avg1, "epsilon0_1")

    plt.plot(100*np.array(optimalActions1) / (n_tests),
             'b', label="$\epsilon = 0.0 Q_t = 5.0$")

    plt.plot(100*np.array(optimalActions2) / (n_tests),
             'r', label="$\epsilon = 0.1 Q_t = 0.0$")
    plt.xlabel("n steps")
    plt.ylabel("%optimal values")
    plt.title('Exponential Recency Method - avg reward')
    plt.legend()
    plt.savefig("2.6a optimal actions percentage.png")


def run2_7a():
    n_tests = 2000
    n_steps = 1000
    testBed = TestBed(n_tests, n_steps)
    rewards1, optimalActions1 = testBed.run_experiment2_7a(
        k=10, epsilon=0.0, alpha=0.1, Q=None, action="UCB", constant=2.0)
    rewards2, optimalActions2 = testBed.run_experiment2_7a(
        k=10, epsilon=0.1, alpha=0.1, Q=None, action="EGreedy", constant=0.1)
    avg1 = getAvgRewards(rewards1, n_tests, n_steps)
    avg2 = getAvgRewards(rewards2, n_tests, n_steps)

    # save_data(avg1, "epsilon0")
    # save_data(avg1, "epsilon0_01")
    # save_data(avg1, "epsilon0_1")

    plt.plot(avg1,
             'b', label="$c=2 UCB$")

    plt.plot(avg2,
             'r', label="$\epsilon = 0.1 \epsilon - Greedy$")
    plt.xlabel("n steps")
    plt.ylabel("Average reward")
    plt.title('UCB vs EGreedy')
    plt.legend()
    plt.savefig("2.7a average reward.png")


def run2_8a():
    n_tests = 2000
    n_steps = 1000
    testBed = TestBed(n_tests, n_steps)
    rewards1, optimalActions1 = testBed.run_experiment2_8a(
        k=10, epsilon=0.1, alpha=0.1, Q=np.full(10, 0.0), baseline=True)
    rewards2, optimalActions2 = testBed.run_experiment2_8a(
        k=10, epsilon=0.1, alpha=0.4, Q=np.full(10, 0.0), baseline=True)

    # without baseline

    rewards3, optimalActions3 = testBed.run_experiment2_8a(
        k=10, epsilon=0.1, alpha=0.1, Q=np.full(10, 0.0), baseline=False)
    rewards4, optimalActions4 = testBed.run_experiment2_8a(
        k=10, epsilon=0.1, alpha=0.4, Q=np.full(10, 0.0), baseline=False)

    # save_data(avg1, "epsilon0")
    # save_data(avg1, "epsilon0_01")
    # save_data(avg1, "epsilon0_1")

    plt.plot(100*np.array(optimalActions1) / (n_tests),
             'b', label="$\\alpha = 0.1 with baseline$")

    plt.plot(100*np.array(optimalActions2) / (n_tests),
             'r', label="$\\alpha = 0.4 with baseline$")

    plt.plot(100*np.array(optimalActions3) / (n_tests),
             'g', label="$\\alpha = 0.1 without baseline$")

    plt.plot(100*np.array(optimalActions4) / (n_tests),
             'orange', label="$\\alpha = 0.4 without baseline$")

    plt.xlabel("n steps")
    plt.ylabel("%optimal values")
    plt.title('Gradient bandit with/without baseline')
    plt.legend()
    plt.savefig("2.8a gradient bandi.png")


def run2_9():
    n_tests = 2000
    n_steps = 1000
    params = [2**(i-7) for i in range(10)]
    # EGREEDY
    testbed = TestBed(n_tests=n_tests, n_steps=n_steps)
    EGreedyRewards = []
    for param in params[:6]:

        rewards1, optimalActions1 = testbed.run_experiment2_9(
            k=10, epsilon=param, alpha=0.1, Q=np.full(10, 0.0), baseline=True, action="EGreedy", method="sample_average")
        avg_rewards = getAvgRewards(rewards1, n_tests, n_steps)
        EGreedyRewards.append(np.mean(avg_rewards))
    plt.plot(EGreedyRewards,
             'b', label="$EGreedy$")
    plt.savefig("2.9 parametric study.png")


def getAvgRewards(rewards, n_tests, n_steps):
    avg_rewards = np.zeros(n_steps)
    for i, run in enumerate(rewards):

        for j, reward in enumerate(run):
            avg_rewards[j] += reward

    avg_rewards /= n_tests
    return avg_rewards


def displayResults(results):
    plt.plot(results)
    plt.show()


def save_data(results, name):
    f = open(f"{name}.pkl", "wb")
    pickle.dump(results, f)


if __name__ == "__main__":
    run2_9()
