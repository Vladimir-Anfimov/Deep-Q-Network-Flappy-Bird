import random
import flappy_bird_gymnasium
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
import torch
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array")

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_actions):
        super(NeuralNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(), 
            torch.nn.Linear(32 * 9 * 9, 128),
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


def img_preprocess(img):
    img = img[:400,:]
    img = cv2.resize(img, (84, 84))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = torch.tensor(img)
    return img


GAMMA = 0.99
LEARNING_RATE = 10e-4
MEMORY_SIZE = 100000
BATCH_SIZE = 32
EPSILON_START = 5e-2
EPSILON_END = 1e-4
EPSILON_DECAY = 0.9995


NUMBER_OF_GAMES = 100_000

def train():
    model = NeuralNetwork(env.action_space.n).to(device)
    target_model = NeuralNetwork(env.action_space.n).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = EPSILON_START

    for episode in range(NUMBER_OF_GAMES):
        _ = env.reset()

        image = env.render()
        state = img_preprocess(image).to(device)
        
        done = False
        total_reward = 0
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = state.clone().detach().float().unsqueeze(0).unsqueeze(0).to(device)
                action = model(state_tensor).argmax(dim=1).item()

            _, reward, done, _, _ = env.step(action)
            next_image = env.render()
            next_state = img_preprocess(next_image).to(device)

            if reward == 1:
                reward = 10
            elif reward == -1:
                reward = -5

            memory.add((state, action, reward, next_state, done))

            if len(memory.memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.stack(states).float().unsqueeze(1).to(device)
                actions = torch.tensor(actions).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards).unsqueeze(1).to(device)
                next_states = torch.stack(next_states).float().unsqueeze(1).to(device)
                dones = torch.tensor(dones).unsqueeze(1).to(device)


                current_q_values = model(states).gather(1, actions)
                next_q_values = (target_model(next_states).max(dim=1)[0].unsqueeze(1))
                target_q_values = (rewards + GAMMA * next_q_values * (1 - dones.float()))


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

        if episode % 250 == 0:
            torch.save(model.state_dict(), f"./models/model_{episode}.pth")



if __name__ == "__main__":
    train()

