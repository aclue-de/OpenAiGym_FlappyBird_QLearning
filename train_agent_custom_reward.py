from contextlib import nullcontext
import itertools
import random
import time

import flappy_bird_gym
import numpy as np
from IPython.display import clear_output

# Hyperparameters
ALPHA = 0.01  # learning rate (0.7 to start, reduce over time)
GAMMA = 0.1  # discount factor
EPSILON = 0.1  # exploration rate (0.1 to start, adjust over time)


def load_data():
    try:
        q_table = np.load(f"q_table_custom.npy")
        print("q_table loaded")
    except FileNotFoundError:
        q_table = np.array([])
        print("q_table created")
    return q_table


def transform_state(state):
    screen_size = env._screen_size

    x = int((state[0] * screen_size[0]).round(0))
    y = int((state[1] * screen_size[1]).round(0))

    return np.array([x, y])


def get_state_index(state, q_table):
    if len(q_table) == 0:
        return None

    transformed_state = transform_state(state)
    return np.where(np.all(q_table[:, 0] == transformed_state, axis=1))


def add_or_get_state_in_q_table(state, q_table):
    state_index = get_state_index(state, q_table)

    if state_index is not None and len(q_table[state_index]) > 0:
        return np.argmax(q_table[state_index, 1]), state_index, q_table
    else:
        q_values = [0, 0]
        if len(q_table) == 0:
            q_table = np.array([[transform_state(state), q_values]])
        else:
            q_table = np.append(
                q_table, [[transform_state(state), q_values]], axis=0)
        # default action for new state
        return 0, len(q_table) - 1, q_table


def render_game(enabled=False):
    if enabled:
        env.render()
        time.sleep(1 / 60)  # FPS


if __name__ == "__main__":
    # make gym environment
    env = flappy_bird_gym.make("FlappyBird-v0")

    # load or create q-table
    q_table = load_data()

    # for logging
    epoch_history = []

    for i in range(1, 100001):
        # reset environment
        state = env.reset()

        state_index = None
        epochs, reward, = 0, 0
        done = False

        while not done:

            # Exploit learned values
            action, state_index, q_table = add_or_get_state_in_q_table(
                state, q_table)
            # Explore action space
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()

            # perform chosen action
            next_state, _, done, info = env.step(
                action)

            # calculate custom reward
            reward = 0
            if done:
                reward = -1000

            # get old value in q-table
            old_value = float(q_table[state_index, 1, action])

            # determine best action for next state
            next_state_index = get_state_index(next_state, q_table)
            next_state_actions = q_table[next_state_index]
            next_max = np.max(
                next_state_actions[:, 1]) if len(next_state_actions) > 0 else 0

            # calculate q-function
            new_value = float((1 - ALPHA) * old_value + ALPHA *
                              (reward + GAMMA * next_max))
            # update q-table
            q_table[state_index, 1, action] = new_value

            # advance state for next loop
            state = next_state
            epochs += 1

            # Render the game (should be disabled in training)
            render_game(False)

        # for logging
        epoch_history.append(epochs)

        # print result and save current q-table to file every 10 episodes
        if i % 100 == 0:
            clear_output(wait=True)
            print(
                f"(saved) | Episode: {i} | Epochs: min: {min(epoch_history)}, avg: {sum(epoch_history) / len(epoch_history)}, max: {max(epoch_history)}")
            epoch_history = []
            np.save(f'q_table_custom', q_table)

    print("Training finished.\n")
