import itertools
import random
import time

import flappy_bird_gym
import numpy as np
from IPython.display import clear_output

DATA_DIVIDER = 10

# Hyperparameters
ALPHA = 0.001  # learning rate
GAMMA = 0.2  # discount factor
EPSILON = 0.1  # exploration rate


def transform_state(state):
    x = (state[0] * h_dist).clip(min=0, max=h_dist).round(0)
    y = (state[1] * v_dist).clip(min=-v_dist, max=v_dist).round(0)
    transformed_state = np.array([x, y])
    return state_permutations.index(
        (transformed_state[0], transformed_state[1]))


if __name__ == "__main__":
    env = flappy_bird_gym.make("FlappyBird-v0")

    screen_size = env._screen_size
    h_dist = int(screen_size[0] / DATA_DIVIDER)
    v_dist = int(screen_size[1] / DATA_DIVIDER)
    try:
        q_table = np.load(f"q_table_{DATA_DIVIDER}.npy")
        print("q_table loaded")
    except FileNotFoundError:
        q_table = np.zeros([(h_dist + 1) * (v_dist + 1) * 2, 2])
        print("q_table created")

    y_values = np.arange(-v_dist, v_dist + 1, 1)
    x_values = np.arange(0, h_dist + 1, 1)
    state_permutations = list(itertools.product(x_values, y_values))

    epoch_history = []

    for i in range(1, 10001):
        state = env.reset()
        state_index = transform_state(state)

        epochs, reward, = 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()  # Explore action space
            else:
                # Exploit learned values
                action = np.argmax(q_table[state_index])

            next_state, reward, done, info = env.step(action)
            next_state_index = transform_state(next_state)

            old_value = q_table[state_index][action]
            next_max = np.max(q_table[next_state_index])

            new_value = (1 - ALPHA) * old_value + ALPHA * \
                (reward + GAMMA * next_max)
            q_table[state_index][action] = new_value

            state = next_state
            epochs += 1

            # Rendering the game:
            # (remove this two lines during training)
            # env.render()
            # time.sleep(1 / 300)  # FPS

        epoch_history.append(epochs)

        if i % 10 == 0:
            clear_output(wait=True)
            # print(f"Episode: {i}")
            print(
                f"(saved) | Episode: {i} | Epochs: min: {min(epoch_history)}, avg: {sum(epoch_history) / len(epoch_history)}, max: {max(epoch_history)}")
            epoch_history = []
            np.save(f'q_table_{DATA_DIVIDER}', q_table)

    print("Training finished.\n")
