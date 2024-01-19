import flappy_bird_gymnasium
import gymnasium
from matplotlib import pyplot as plt
import torch

from main import NeuralNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = gymnasium.make("FlappyBird-v0", render_mode="human")

def test(model_number, number_of_games=10):
    model = NeuralNetwork(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(f"./models/model_{model_number}.pth"))
    model.eval()

    for _ in range(number_of_games):

        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            state_tensor = torch.tensor(state).float().unsqueeze(0)
            action = model(state_tensor).argmax(dim=1).item()

            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward

        print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    input_data = input(f"Input [model_number] [number_of_games]: ")
    model_number, number_of_games = input_data.split()
    test(int(model_number), int(number_of_games))

