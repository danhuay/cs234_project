import logging
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import LazyFrames
from torch.utils.tensorboard import SummaryWriter

from final_project.code.src.policy.base import CNNPolicy

logger = logging.getLogger(__name__)


class ReplayBuffer:
    def __init__(self, capacity=None, demonstration=None):
        """demonstration should be a list of (s,a,r,s',done) tuples"""
        if demonstration is None:
            self.fixed_episodes = list()
            self.buffer = deque(maxlen=capacity)
        else:
            self.fixed_episodes = demonstration
            n_fixed_episodes = len(demonstration)
            self.buffer = deque(maxlen=capacity - n_fixed_episodes)

    def push(self, state, action, reward, next_state, done):
        # convert lazyframes to tensors
        state = state.__array__()
        next_state = next_state.__array__()

        # will overwrite the buffer if it is full
        self.buffer.append((state, action, reward, next_state, done))

    def get_buffer(self):
        return self.fixed_episodes + list(self.buffer)

    def sample(self, batch_size):
        buffer = self.get_buffer()
        samples = random.choices(buffer, k=batch_size)
        state, action, reward, next_state, done = zip(*samples)

        state = torch.tensor(np.array(state), dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
        done = torch.tensor(np.array(done), dtype=torch.long)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.get_buffer())


# DQN Agent that ties everything together
class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        buffer_capacity=10000,
        batch_size=64,
        demonstration=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Main and target networks
        self.policy_net = CNNPolicy(state_dim, action_dim).to(self.device)
        self.target_net = CNNPolicy(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # set target network to evaluation mode

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity, demonstration)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return (
                torch.tensor(random.randrange(self.action_dim)).to(self.device).item()
            )
        else:
            if type(state) == LazyFrames:
                state = torch.tensor(state.__array__(), dtype=torch.float32)
            state = state.to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    def update(self, global_step):
        if len(self.replay_buffer) < self.batch_size:
            return None  # not enough samples in buffer

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q values
        current_q_values = (
            self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# Example training loop with TensorBoard logging
def train(
    agent,
    env,
    num_episodes,
    epsilon_start=1.0,
    epsilon_final=0.01,
    log_dir="runs/dqn_experiment",
    model_save_path="checkpoints/dqn_model.pt",
):
    writer = SummaryWriter(log_dir=log_dir)
    episode_rewards = []
    global_step = 0
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Update epsilon
            epsilon_decay = num_episodes
            epsilon = max(epsilon_final, epsilon_start - episode / epsilon_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            global_step += 1

            # Update agent and log loss if available
            loss = agent.update(global_step)
            if loss is not None and global_step % 1000 == 0:
                writer.add_scalar("Loss/train", loss, global_step)

        # Optionally update the target network every few episodes
        if episode % 5 == 0:
            agent.update_target_network()

        episode_rewards.append(episode_reward)
        writer.add_scalar("Reward/episode", episode_reward, episode)
        writer.add_scalar("Epsilon", epsilon, episode)
        print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}")

        writer.add_text("Info/episode", str(info), episode)

    writer.close()

    # Save the model
    torch.save(agent.policy_net.state_dict(), model_save_path)
    return episode_rewards
