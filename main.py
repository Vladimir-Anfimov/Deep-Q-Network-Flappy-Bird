import flappy_bird_gymnasium
import gymnasium
from matplotlib import pyplot as plt


env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array")

obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    
    rgb_array = env.render()

    print(f"RgB array shape: {rgb_array / 255.0}")

    # Save rgb_array as an image
    plt.imsave(f"rgb_array.png", rgb_array)
    
    exit()
    # Checking if the player is still alive
    if terminated:
        break


