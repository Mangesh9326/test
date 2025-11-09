import numpy as np
import gymnasium as gym
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

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

state_space_size = env.observation_space.n
action_space_size = env.action_space.n

# Neural Network Model
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# One-hot encode states for NN input
def one_hot_state(state, state_space_size):
    vec = np.zeros(state_space_size, dtype=np.float32)
    vec[state] = 1.0
    return vec

q_net = QNetwork(state_space_size, action_space_size)
optimizer = optim.Adam(q_net.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training parameters
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.997
min_epsilon = 0.01
episodes = 5000
max_steps = 100

total_rewards = []

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(one_hot_state(state, state_space_size)).unsqueeze(0)
                q_values = q_net(state_tensor)
                action = torch.argmax(q_values).item()

        # Take action
        new_state, reward, terminated, truncated, _ = env.step(action)

        # Q-learning target
        with torch.no_grad():
            new_state_tensor = torch.tensor(one_hot_state(new_state, state_space_size)).unsqueeze(0)
            max_next_q = torch.max(q_net(new_state_tensor)).item()
            target_q = reward + gamma * max_next_q * (1 - int(terminated or truncated))

        # Current Q prediction
        state_tensor = torch.tensor(one_hot_state(state, state_space_size)).unsqueeze(0)
        q_values = q_net(state_tensor)
        target = q_values.clone()
        target[0, action] = target_q

        # Optimize
        loss = loss_fn(q_values, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = new_state
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    total_rewards.append(total_reward)

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
plt.title("Episodes vs Rewards - FrozenLake NN Q-Learning")
plt.legend()
plt.grid(True)
plt.show()

# Visualization function
action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

def state_to_position(state, size):
    return (state // size, state % size)

def visualize_agent(q_net, size, episodes=5):
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
            with torch.no_grad():
                state_tensor = torch.tensor(one_hot_state(state, state_space_size)).unsqueeze(0)
                action = torch.argmax(q_net(state_tensor)).item()

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
visualize_agent(q_net, size)
