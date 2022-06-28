from contextlib import nullcontext
import itertools
import random
import time

import flappy_bird_gym
import numpy as np
from IPython.display import clear_output

# Hyperparameters
ALPHA = 0.5  # learning rate
GAMMA = 0.5  # discount factor
EPSILON = 0  # exploration rate

EPISODES = 2001  # how many attempts for the agent


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
        time.sleep(1 / 120)  # FPS


if __name__ == "__main__":
    # make gym environment
    env = flappy_bird_gym.make("FlappyBird-v0")

    # load or create q-table
    q_table = load_data()

    # for logging
    epoch_history = []

    for i in range(1, EPISODES):
        # reset environment
        state = env.reset()

        state_index = None
        epochs, reward, = 0, 0
        done = False

        while not done:

            # Exploit learned values
            actions, state_index, q_table = get_state_action_in_q_table(
                state, q_table)
            # Explore action space
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(actions)

            # perform chosen action
            next_state, _, done, info = env.step(
                action)

            # calculate custom reward
            reward = 0
            if done:
                reward = -10

            # get old value in q-table
            old_value = actions[action]

            # determine best action for next state
            next_actions, _, q_table = get_state_action_in_q_table(
                next_state, q_table)
            next_max = np.max(next_actions)

            # calculate q-function
            new_value = (1 - ALPHA) * old_value + ALPHA * \
                (reward + GAMMA * next_max)
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
                f"(saved) | Episode: {i} | Epochs - min: {min(epoch_history)}, avg: {sum(epoch_history) / len(epoch_history)}, max: {max(epoch_history)}")
            epoch_history = []
            np.save(f'q_table', q_table)

    print("Training finished.\n")
