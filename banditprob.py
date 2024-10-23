import numpy as np
import matplotlib.pyplot as plt

# Problem 1 code (Binary Bandit)
def run_problem_1():
    Q = np.zeros(2)
    N = np.zeros(2)
    e = 0.1
    avg = []
    
    def binary_bandit_B(action):
        p = [0.8, 0.9]
        return 1 if np.random.rand() < p[action] else 0
    
    for i in range(1000):
        if np.random.rand() > e:
            A = np.argmax(Q)
        else:
            A = np.random.choice([0, 1])
        R = binary_bandit_B(A)  # reward
        N[A] += 1
        Q[A] = Q[A] + (R - Q[A]) / N[A]
        if i == 0:
            avg.append(R)
        else:
            avg.append(((i - 1) * avg[-1] + R) / i)

    plt.plot(avg, color="red")
    plt.ylim([0, 1])
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Problem 1: Binary Bandit - Epsilon Greedy")
    plt.show()

# Problem 2 code (Non-Stationary 10-Armed Bandit)
def run_problem_2():
    Q = np.zeros(10)
    N = np.zeros(10)
    R = np.zeros(10000)
    epsilon = 0.1
    m = np.ones(10)
    
    def bandit_nonstat(action, m):
        v = np.random.normal(0, 0.01, 10)
        m = m + v
        return m[action], m
    
    for i in range(10000):
        if np.random.rand() > epsilon:
            A = np.argmax(Q)
        else:
            A = np.random.choice(range(10))
        RR, m = bandit_nonstat(A, m)
        N[A] += 1
        Q[A] = Q[A] + (RR - Q[A]) / N[A]
        if i == 0:
            R[i] = RR
        else:
            R[i] = ((i - 1) * R[i - 1] + RR) / i

    plt.plot(R, color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Problem 2: Non-Stationary 10-Armed Bandit - Epsilon Greedy")
    plt.show()

# Problem 3 code (Modified Epsilon-Greedy with Alpha)
def run_problem_3():
    Q = np.zeros(10)
    N = np.zeros(10)
    R = np.zeros(10000)
    epsilon = 0.1
    alpha = 0.7
    m = np.ones(10)
    
    def bandit_nonstat(action, m):
        v = np.random.normal(0, 0.01, 10)
        m = m + v
        return m[action], m
    
    for i in range(10000):
        if np.random.rand() > epsilon:
            A = np.argmax(Q)
        else:
            A = np.random.choice(range(10))
        RR, m = bandit_nonstat(A, m)
        N[A] += 1
        Q[A] = Q[A] + (RR - Q[A]) * alpha
        if i == 0:
            R[i] = RR
        else:
            R[i] = ((i - 1) * R[i - 1] + RR) / i

    plt.plot(R, color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Problem 3: Non-Stationary Bandit - Modified Epsilon Greedy with Alpha")
    plt.show()

# Main function to call each problem's code
if _name_ == "_main_":
    # Uncomment the function you want to run
    # run_problem_1()  # Run code for Problem 1
    # run_problem_2()  # Run code for Problem 2
    # run_problem_3()  # Run code for Problem 3
