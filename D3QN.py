import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from collections import Counter
from collections import deque

USE_CUDA = torch.cuda.is_available()

# 移除旧的Variable定义，直接使用Tensor.to(device)
device = torch.device("cuda" if USE_CUDA else "cpu")


class DuelingDQN(nn.Module):
    def __init__(self, observation_space, action_sapce):
        super(DuelingDQN, self).__init__()
        self.observation_space = observation_space
        self.action_sapce = action_sapce

        self.feature = nn.Sequential(
            nn.Linear(observation_space, 128),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_sapce),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def act(self, state, epsilon):
        if random.random() > epsilon:
            # 使用torch.no_grad()代替volatile=True
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():  # 禁用梯度计算
                q_value = self.forward(state_tensor)

            action = q_value.max(1)[1].item()  # 直接使用.item()获取标量值
        else:
            action = random.randrange(self.action_sapce)
        return action


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done


def compute_td_loss(current_model, target_model, optimizer, replay_buffer, gamma, batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    # 转换为Tensor并移动到设备
    state = torch.FloatTensor(np.float32(state)).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)

    # 当前网络计算Q值（启用梯度）
    q_values = current_model(state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    # 目标网络计算Q值（禁用梯度）
    with torch.no_grad():  # 代替volatile=True
        next_q_values = target_model(next_state)
        next_q_value = next_q_values.max(1)[0]

    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = F.mse_loss(q_value, expected_q_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())
