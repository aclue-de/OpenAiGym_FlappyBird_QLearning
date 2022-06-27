import itertools
import random
import time

import flappy_bird_gym
import numpy as np
from IPython.display import clear_output

DATA_DIVIDER = 10

# Hyperparameters
ALPHA = 0.01  # learning rate
GAMMA = 0.1  # discount factor
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

    top_epochs = 0

    for i in range(1, 10001):
        state = env.reset()
        state_index = transform_state(state)

        epochs, reward, = 0, 0
        done = False

        while not done:
            # action = env.action_space.sample()  # Explore action space
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

            if epochs > top_epochs:
                top_epochs = epochs

            # Rendering the game:
            # (remove this two lines during training)
            # env.render()
            # time.sleep(1 / 60)  # FPS

        if i % 10 == 0:
            clear_output(wait=True)
            # print(f"Episode: {i}")
            print(
                f"(saved) | Episode: {i} | Top Epochs: {top_epochs}")
            np.save(f'q_table_{DATA_DIVIDER}', q_table)

    print("Training finished.\n")
