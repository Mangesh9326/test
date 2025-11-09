import numpy as np
import gymnasium as gym
import warnings
import matplotlib.pyplot as plt

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

# SARSA parameters
alpha = 0.8            # Learning rate
gamma = 0.95           # Discount factor
epsilon = 1.0          # Initial exploration rate
epsilon_decay = 0.997  # Slower decay for better exploration
min_epsilon = 0.01
episodes = 5000         # Training episodes
max_steps = 100

# Initialize Q-table
state_space_size = env.observation_space.n
action_space_size = env.action_space.n
Q_table = np.zeros((state_space_size, action_space_size))

# Training loop (SARSA)
total_rewards = []

for episode in range(episodes):
    state, _ = env.reset()

    # ε-greedy policy for first action
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q_table[state])

    total_reward = 0

    for step in range(max_steps):
        new_state, reward, terminated, truncated, _ = env.step(action)

        # Choose next action using ε-greedy (on-policy)
        if np.random.rand() < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(Q_table[new_state])

        # SARSA update
        Q_table[state, action] += alpha * (
            reward + gamma * Q_table[new_state, next_action] - Q_table[state, action]
        )

        state, action = new_state, next_action
        total_reward += reward

        if terminated or truncated:
            break

    # Decay epsilon per episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    total_rewards.append(total_reward)

    if (episode + 1) % 100 == 0:
        avg_last = np.mean(total_rewards[-100:])
        print(f"Episode {episode+1}/{episodes} - Avg Reward (last 100 eps): {avg_last:.3f} - Epsilon: {epsilon:.3f}")

print("\nTraining finished.")
print(f"Average reward over all episodes: {np.mean(total_rewards):.4f}")

# Step 5: Plot Rewards
plt.figure(figsize=(10,5))
plt.plot(total_rewards, label="Reward per Episode", alpha=0.6)
plt.plot(
    np.convolve(total_rewards, np.ones(50)/50, mode='valid'),
    label="Moving Average (50 eps)", linewidth=2
)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Episodes vs Rewards - FrozenLake SARSA")
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
            print(f"\nEpisode {episode + 1}: ✅ Reached goal!")
            print(f"Path taken: {path}")
            print(f"Actions taken: {actions_taken}")
        else:
            print(f"\nEpisode {episode + 1}: ❌ Did not reach goal.")

    env_test.close()

visualize_agent(Q_table, size)
