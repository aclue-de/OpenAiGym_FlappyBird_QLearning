import itertools
import random
import time

import flappy_bird_gym
import numpy as np
from IPython.display import clear_output

DATA_DIVIDER = 10

# Hyperparameters
ALPHA = 0.7  # learning rate (0.7 to start, reduce over time)
GAMMA = 0.2  # discount factor
EPSILON = 0.0  # exploration rate (0 to start, increase over time)


def h_v_dist():
    screen_size = env._screen_size
    h_dist = int(screen_size[0] / DATA_DIVIDER)
    v_dist = int(screen_size[1] / DATA_DIVIDER)
    return h_dist, v_dist


def load_data(h_dist, v_dist):
    try:
        q_table = np.load(f"q_table_custom_{DATA_DIVIDER}.npy")
        print("q_table loaded")
    except FileNotFoundError:
        q_table = np.zeros([(h_dist + 1) * (v_dist + 1) * 2, 2])
        print("q_table created")
    return q_table


def transform_state(h_dist, v_dist, state):
    y_values = np.arange(-v_dist, v_dist + 1, 1)
    x_values = np.arange(0, h_dist + 1, 1)

    state_permutations = list(itertools.product(x_values, y_values))

    x = (state[0] * h_dist).clip(min=0, max=h_dist).round(0)
    y = (state[1] * v_dist).clip(min=-v_dist, max=v_dist).round(0)

    transformed_state = np.array([x, y])
    return state_permutations.index(
        (transformed_state[0], transformed_state[1]))


def render_game(enabled=False):
    if enabled:
        env.render()
        time.sleep(1 / 60)  # FPS


if __name__ == "__main__":
    # make gym environment
    env = flappy_bird_gym.make("FlappyBird-v0")

    # init state with reduced complexity
    h_dist, v_dist = h_v_dist()

    # load or create q-table
    q_table = load_data(h_dist, v_dist)

    # for logging
    epoch_history = []

    for i in range(1, 10001):
        # reset environment
        state = env.reset()
        # get state index for q-table
        state_index = transform_state(h_dist, v_dist, state)

        epochs, reward, = 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < EPSILON:
                # Explore action space
                action = env.action_space.sample()
            else:
                # Exploit learned values
                action = np.argmax(q_table[state_index])

            # perform chosen action
            next_state, _, done, info = env.step(
                action)
            # get next state index for q-table
            next_state_index = transform_state(h_dist, v_dist, next_state)

            # calculate custom reward
            reward = 0
            if done:
                reward = -1000

            # get old value in q-table
            old_value = q_table[state_index][action]
            # determine best action for next state
            next_max = np.max(q_table[next_state_index])

            # calculate q-function
            new_value = (1 - ALPHA) * old_value + ALPHA * \
                (reward + GAMMA * next_max)
            # update q-table
            q_table[state_index][action] = new_value

            # advance state for next loop
            state = next_state
            epochs += 1

            # Render the game (should be disabled in training)
            render_game(True)

        # for logging
        epoch_history.append(epochs)

        # print result and save current q-table to file every 10 episodes
        if i % 10 == 0:
            clear_output(wait=True)
            print(
                f"(saved) | Episode: {i} | Epochs: min: {min(epoch_history)}, avg: {sum(epoch_history) / len(epoch_history)}, max: {max(epoch_history)}")
            epoch_history = []
            np.save(f'q_table_{DATA_DIVIDER}', q_table)

    print("Training finished.\n")
