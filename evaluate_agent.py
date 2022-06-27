import itertools
import random
import time

import flappy_bird_gym
import numpy as np
from IPython.display import clear_output


def transform_state(state):
    x = (state[0] * h_dist).clip(min=0, max=h_dist).round(0)
    y = (state[1] * v_dist).clip(min=-v_dist, max=v_dist).round(0)
    transformed_state = np.array([x, y])
    return state_permutations.index(
        (transformed_state[0], transformed_state[1]))


DATA_DIVIDER = 10

if __name__ == "__main__":
    env = flappy_bird_gym.make("FlappyBird-v0")

    screen_size = env._screen_size
    h_dist = int(screen_size[0] / DATA_DIVIDER)
    v_dist = int(screen_size[1] / DATA_DIVIDER)

    q_table = np.load(f"q_table_{DATA_DIVIDER}.npy")
    print("q_table loaded")

    y_values = np.arange(-v_dist, v_dist + 1, 1)
    x_values = np.arange(0, h_dist + 1, 1)
    state_permutations = list(itertools.product(x_values, y_values))

    """Evaluate agent's performance after Q-learning"""

    total_epochs, total_penalties = 0, 0
    episodes = 100

    for _ in range(episodes):
        state = env.reset()
        state_index = transform_state(state)
        epochs, reward = 0, 0

        done = False

        while not done:
            action = np.argmax(q_table[state_index])
            state, reward, done, info = env.step(action)

            epochs += 1

            # env.render()
            # time.sleep(1 / 30)  # FPS

        env.close()

        total_epochs += epochs

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
