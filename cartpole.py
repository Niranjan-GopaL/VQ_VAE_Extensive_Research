"""
Created on Sun Jun 8 01:04:14 2025

@author: harsh
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# --- Setup Environment ---
env = gym.make("CartPole-v1")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Define the Policy Network ---

class PolicyNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        # This final layer outputs "logits" - raw scores for each action.
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        Called when we pass a state to the network.
        Returns the action probabilities.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        # We use softmax to convert the raw scores (logits) into a
        # probability distribution that sums to 1.
        return F.softmax(x, dim=1)


# --- Hyperparameters ---
LR = 1e-4            # Learning rate
GAMMA = 0.99          # Discount factor for future rewards
NUM_EPISODES = 1500   # Total number of episodes to run
PRINT_EVERY = 100     # How often to print the average score

# --- Get Environment Info ---
state, info = env.reset()
n_observations = len(state)
n_actions = env.action_space.n

# --- Initialize Network and Optimizer ---
policy_net = PolicyNetwork(n_observations, n_actions).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

# We also keep a running list of the last 100 episode scores
# to see how the agent is improving.
scores_deque = deque(maxlen=100)
all_scores = []

# --- Main Training Loop ---
print("Starting training with REINFORCE...")

for i_episode in range(NUM_EPISODES):
    
    # --- Run one full episode ---
    
    # These lists will store the log-probabilities and rewards
    # for *every step* in this single episode.
    saved_log_probs = []
    rewards = []
    
    state, info = env.reset()
    done = False
    
    while not done:
        # 1. Get Action Probabilities
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action_probs = policy_net(state_tensor)
        
        # 2. Select an Action
        # We create a categorical distribution from our probabilities
        # and sample from it. This is how the agent explores.
        m = Categorical(action_probs)
        action = m.sample()
        
        # 3. Store the log-probability of the action we took
        # We need this for the loss calculation later.
        saved_log_probs.append(m.log_prob(action))
        
        # 4. Take the action in the environment
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated

    # --- Episode finished ---
    
    total_reward = sum(rewards)
    scores_deque.append(total_reward)
    all_scores.append(total_reward)
    
    # --- Calculate Discounted Future Rewards (Returns) ---
    
    # `returns` will be a list of the discounted reward-to-go for
    # each timestep.
    # e.g., G_t = r_t + GAMMA * r_{t+1} + GAMMA^2 * r_{t+2} + ...
    returns = []
    discounted_reward = 0
    
    # We iterate *backwards* through the rewards
    for r in reversed(rewards):
        discounted_reward = r + GAMMA * discounted_reward
        returns.insert(0, discounted_reward) # Prepend to list
        
    # Convert returns to a tensor
    returns = torch.tensor(returns, device=device)
    
    # --- Normalize the returns (optional, but helps stability) ---
    # This scales the returns to have a mean of 0 and std dev of 1.
    # It stops very high or low rewards from creating massive gradients.
    returns = (returns - returns.mean()) / (returns.std() + 1e-6) # Add 1e-6 to avoid div by zero
    
    # --- Calculate the Loss ---
    
    policy_loss = []
    # We iterate through the log-probs and the corresponding returns
    for log_prob, R in zip(saved_log_probs, returns):
        # The loss for each step is - (log_prob * Return)
        # We use a negative sign because optimizers *minimize* loss,
        # but we want to *maximize* (log_prob * Return).
        # A high positive Return will make the loss negative, pushing
        # the optimizer to make that log_prob *more* likely.
        policy_loss.append(-log_prob * R)
        
    # --- Optimize the Model (Backpropagation) ---
    
    optimizer.zero_grad()
    
    # Sum up all the step losses into one total loss
    loss = torch.cat(policy_loss).sum()
    loss.backward()
    
    optimizer.step()
    
    # --- Print Progress ---
    if i_episode % PRINT_EVERY == 0 and i_episode > 0:
        avg_score = np.mean(scores_deque)
        print(f"Episode {i_episode}\tAverage Score (last 100): {avg_score:.2f}")
        
    # Check for solving
    if np.mean(scores_deque) >= 195.0 and i_episode >= 100:
        print(f"Solved in {i_episode} episodes! Average score is {np.mean(scores_deque):.2f}")
        break

print('Complete')
env.close()

# --- Plot the results ---
plt.figure(1)
plt.title('REINFORCE Result')
plt.xlabel('Episode')
plt.ylabel('Duration (Total Reward)')
plt.plot(all_scores)
# Take a 100-episode rolling average
if len(all_scores) >= 100:
    means = np.convolve(all_scores, np.ones(100)/100, mode='valid')
    plt.plot(np.arange(99, len(all_scores)), means, label='100-episode average', color='red')
plt.legend()
plt.show()
