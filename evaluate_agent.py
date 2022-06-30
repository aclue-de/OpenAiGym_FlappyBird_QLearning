import itertools
import random
import time

import flappy_bird_gym
import numpy as np
from IPython.display import clear_output


EPISODES = 10001  # how many attempts for the agent


# load existing q-table from file or create a new one
def load_data():
    try:
        q_table = np.load(f"q_table.npy")
        print("q_table loaded")
    except FileNotFoundError:
        q_table = np.array([])
        print("q_table created")
    return q_table


# transform the states to match pixel values of the game
def transform_state(state):
    screen_size = env._screen_size

    x = (state[0] * screen_size[0])
    y = (state[1] * screen_size[1])

    return np.around(np.array([x, y]), decimals=0)


def get_state_index(state, q_table):
    if len(q_table) == 0:
        return None

    transformed_state = transform_state(state)
    return np.where(np.all(q_table[:, 0] == transformed_state, axis=1))


def get_state_action_in_q_table(state, q_table):
    state_index = get_state_index(state, q_table)

    if state_index is not None and len(q_table[state_index]) > 0:
        return q_table[state_index, 1][0, 0], state_index, q_table
    else:
        q_values = [0, 0]
        if len(q_table) == 0:
            q_table = np.array([[transform_state(state), q_values]])
        else:
            q_table = np.append(
                q_table, [[transform_state(state), q_values]], axis=0)
        # default action for new state
        return q_values, len(q_table) - 1, q_table


def render_game(enabled=False):
    if enabled:
        env.render()
        time.sleep(1 / 30)  # FPS


if __name__ == "__main__":
    env = flappy_bird_gym.make("FlappyBird-v0")

    q_table = load_data()

    epoch_history = []

    for _ in range(1, EPISODES):
        state = env.reset()

        state_index = None
        epochs, reward, = 0, 0
        done = False

        while not done:
            action = np.argmax(q_table[state_index])
            state, reward, done, info = env.step(action)

            actions, state_index, q_table = get_state_action_in_q_table(
                state, q_table)
            action = np.argmax(actions)

            next_state, _, done, info = env.step(
                action)

            state = next_state

            epochs += 1
            render_game(True)

        epoch_history.append(epochs)

    print(f"Results after {EPISODES} episodes:")
    print(
        f"Epochs - min: {min(epoch_history)}, avg: {sum(epoch_history) / len(epoch_history)}, max: {max(epoch_history)}")
