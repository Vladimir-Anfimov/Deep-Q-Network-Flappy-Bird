import random
import flappy_bird_gymnasium
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array")

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(NeuralNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions)
        )

    def forward(self, state):
        return self.model(state)
    


class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []
        self.index = 0

    def add(self, experience):
        if len(self.memory) < self.max_size:
            self.memory.append(None)
        self.memory[self.index] = experience
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


GAMMA = 0.99
LEARNING_RATE = 10e-4
MEMORY_SIZE = 100000
BATCH_SIZE = 64
EPSILON_START = 5e-2
EPSILON_END = 1e-4
EPSILON_DECAY = 0.9995


NUMBER_OF_GAMES = 100000

def train():
    model = NeuralNetwork(env.observation_space.shape[0], env.action_space.n)
    target_model = NeuralNetwork(env.observation_space.shape[0], env.action_space.n)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = EPSILON_START

    for episode in range(NUMBER_OF_GAMES):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state).float().unsqueeze(0)
                action = model(state_tensor).argmax(dim=1).item()

            next_state, reward, done, _, _ = env.step(action)

            if reward == 1:
                reward = 10
            elif reward == -1:
                reward = -5

            memory.add((state, action, reward, next_state, done))

            if len(memory.memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.tensor(np.array(states)).float()
                actions_tensor = torch.tensor(np.array(actions)).long()
                rewards_tensor = torch.tensor(np.array(rewards)).float()
                next_states_tensor = torch.tensor(np.array(next_states)).float()
                dones_tensor = torch.tensor(np.array(dones)).float()

                current_q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                next_q_values = target_model(next_states_tensor).max(dim=1).values
                target_q_values = rewards_tensor + (1 - dones_tensor) * next_q_values * GAMMA

                loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epsilon > EPSILON_END:
                    epsilon *= EPSILON_DECAY

                if episode % 10 == 0:
                    target_model.load_state_dict(model.state_dict())

            state = next_state
            total_reward += reward

        
        
        
        print(f"Episode: {episode}, total reward: {total_reward}")

        if episode % 1000 == 0:
            torch.save(model.state_dict(), f"./models/model_{episode}.pth")



if __name__ == "__main__":
    train()

