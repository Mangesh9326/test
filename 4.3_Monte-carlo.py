import numpy as np
import gymnasium as gym
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# environment
custom_map = [
    'SFFF',
    'FHFH',
    'FFFH',
    'HFFG'
]

size = 4
env = gym.make(
    'FrozenLake-v1',
    desc=custom_map,
    is_slippery=False,
    render_mode=None
)

# Monte Carlo parameters
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.997
min_epsilon = 0.01
episodes = 5000
max_steps = 100

state_space_size = env.observation_space.n
action_space_size = env.action_space.n

# Q-table and returns for averaging
Q_table = np.zeros((state_space_size, action_space_size))
returns_sum = defaultdict(float)
returns_count = defaultdict(float)

# Monte Carlo Training
total_rewards = []

for episode in range(episodes):
    # Generate an episode
    state, _ = env.reset()
    episode_data = []
    for step in range(max_steps):
        # ε-greedy action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_data.append((state, action, reward))
        state = next_state
        if terminated or truncated:
            break

    # First-visit Monte Carlo update
    visited = set()
    G = 0
    for t in reversed(range(len(episode_data))):
        s, a, r = episode_data[t]
        G = gamma * G + r
        if (s, a) not in visited:
            visited.add((s, a))
            returns_sum[(s, a)] += G
            returns_count[(s, a)] += 1
            Q_table[s, a] = returns_sum[(s, a)] / returns_count[(s, a)]

    total_rewards.append(sum(r for _, _, r in episode_data))

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        avg_last = np.mean(total_rewards[-100:])
        print(f"Episode {episode+1}/{episodes} - Avg Reward (last 100 eps): {avg_last:.3f} - Epsilon: {epsilon:.3f}")

print("\nTraining finished.")
print(f"Average reward over all episodes: {np.mean(total_rewards):.4f}")

# Plot Rewards
plt.figure(figsize=(10,5))
plt.plot(total_rewards, label="Reward per Episode", alpha=0.6)
plt.plot(
    np.convolve(total_rewards, np.ones(50)/50, mode='valid'),
    label="Moving Average (50 eps)", linewidth=2
)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Episodes vs Rewards - FrozenLake Monte Carlo Control")
plt.legend()
plt.grid(True)
plt.show()

# Visualization function
action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

def state_to_position(state, size):
    return (state // size, state % size)

def visualize_agent(Q_table, size, episodes=5):
    env_test = gym.make(
        'FrozenLake-v1',
        desc=custom_map,
        is_slippery=False,
        render_mode="human"
    )

    for episode in range(episodes):
        state, _ = env_test.reset()
        done = False
        path = [state_to_position(state, size)]
        actions_taken = []
        total_reward = 0
        step = 0

        while not done and step < max_steps:
            action = np.argmax(Q_table[state])
            actions_taken.append(action_names[action])
            next_state, reward, terminated, truncated, _ = env_test.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
            path.append(state_to_position(state, size))
            step += 1

        if total_reward >= 1.0:
            print(f"\nEpisode {episode + 1}:  Reached goal!")
            print(f"Path taken: {path}")
            print(f"Actions taken: {actions_taken}")
        else:
            print(f"\nEpisode {episode + 1}:  Did not reach goal.")

    env_test.close()
visualize_agent(Q_table, size)
