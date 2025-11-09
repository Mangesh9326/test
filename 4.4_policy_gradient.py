import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings

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

# Define Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# One-hot encoding for discrete states
def one_hot_state(state, state_size):
    vec = np.zeros(state_size, dtype=np.float32)
    vec[state] = 1.0
    return vec

# Training parameters
gamma = 0.99
learning_rate = 0.01
episodes = 2000
max_steps = 100

policy_net = PolicyNetwork(state_space_size, action_space_size)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

total_rewards = []

# REINFORCE Training Loop
for episode in range(episodes):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    total_reward = 0

    for step in range(max_steps):
        state_tensor = torch.tensor(one_hot_state(state, state_space_size)).unsqueeze(0)
        action_probs = policy_net(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        total_reward += reward

        state = next_state
        if terminated or truncated:
            break

    # Compute returns (discounted rewards)
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    # Policy gradient update
    loss = []
    for log_prob, G in zip(log_probs, returns):
        loss.append(-log_prob * G)
    loss = torch.cat(loss).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_rewards.append(total_reward)

    if (episode + 1) % 100 == 0:
        avg_last = np.mean(total_rewards[-100:])
        print(f"Episode {episode+1}/{episodes} - Avg Reward (last 100 eps): {avg_last:.3f}")

print("\nTraining finished.")
print(f"Average reward over all episodes: {np.mean(total_rewards):.4f}")

# Plot Rewards
plt.figure(figsize=(10, 5))
plt.plot(total_rewards, label="Reward per Episode", alpha=0.6)
plt.plot(
    np.convolve(total_rewards, np.ones(50)/50, mode='valid'),
    label="Moving Average (50 eps)", linewidth=2
)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Episodes vs Rewards - FrozenLake Policy Gradient")
plt.legend()
plt.grid(True)
plt.show()

# Visualization function
action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

def state_to_position(state, size):
    return (state // size, state % size)

def visualize_agent(policy_net, size, episodes=5):
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
            state_tensor = torch.tensor(one_hot_state(state, state_space_size)).unsqueeze(0)
            action_probs = policy_net(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()
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
visualize_agent(policy_net, size)
