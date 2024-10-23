# Non-Stationary Bandit Function
def non_stat_bandit(action, m):
    v = np.random.normal(0, 0.01, 10)
    m += v
    return m[action], m

# Epsilon-Greedy Algorithm for Non-Stationary Bandit
Q = np.zeros(10)
N = np.zeros(10)
R = np.zeros(10000)
epsilon = 0.1
m = np.ones(10)

for i in range(10000):
    if np.random.rand() > epsilon:
        action = np.argmax(Q)
    else:
        action = np.random.randint(0, 10)
    
    reward, m = non_stat_bandit(action, m)
    N[action] += 1
    Q[action] += (reward - Q[action]) / N[action]
    R[i] = reward if i == 0 else ((i * R[i - 1]) + reward) / (i + 1)

# Plot the Average Reward over Time
plt.figure(figsize=(10, 6))
plt.plot(R, color='red')
plt.title('Average Reward - Non-Stationary 10-Armed Bandit')
plt.xlabel('Iterations')
plt.ylabel('Average Reward')
plt.show()
