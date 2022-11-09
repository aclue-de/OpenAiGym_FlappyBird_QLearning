import random
import time
from ast import literal_eval
from datetime import datetime

import flappy_bird_gym
import numpy as np
from IPython.display import clear_output
from pandas import DataFrame, concat, read_csv

# Hyperparameters
ALPHA = 0.7  # learning rate
GAMMA = 0.5  # discount factor
EPSILON = 0  # exploration rate

TRAIN_AGENT = False  # toggle updating of rewards and new states
RENDER_GAME = True  # should be disabled in training

EPISODES = 1000001  # how many attempts for the agent
EPISODE_REPORT = 10  # after how many episodes you receive performance infos
LIMIT_EPOCHS = None  # increase training on earlier steps (None for unlimited)
DATA_REDUCTION = 8  # how much the state values are divided by


# load existing q-table from file or create a new one
def load_data():
    try:
        q_table = read_csv("q_table.csv", delimiter=",", converters={
                           "actions": lambda x: literal_eval(x)})
        print("q_table loaded")
    except FileNotFoundError:
        q_table = DataFrame({
            "state_history": [],
            "actions": []
        })
        print("q_table created")
    return q_table


# initialze the state history with None values
def init_state_history():
    initial_state = [None, None]
    return [initial_state, initial_state,
            initial_state, initial_state]


# transform the states to match pixel values of the game
def transform_state(state):
    screen_size = [288, 512]

    x = (state[0] * screen_size[0] / DATA_REDUCTION)
    y = (state[1] * screen_size[1] / DATA_REDUCTION)

    return list(np.around([x, y], decimals=0))


# append to the state history. return the latest 4 states
def add_state_to_history(state, state_history):
    transformed_state = transform_state(state)

    new_history = state_history + [transformed_state]
    return new_history[-4:]


# find a state in the q table and return it's index
def get_state_index(state_history, q_table):
    state_index = q_table.index[q_table["state_history"].apply(
        lambda s: str(s) == str(state_history))]
    return state_index[0] if len(state_index) > 0 else None


# find a state in the q table and add it if it didn't exist yet
def get_state_action_in_q_table(state_history, q_table):
    state_index = get_state_index(state_history, q_table)

    if state_index != None:
        return q_table.at[state_index, "actions"], state_index, q_table
    else:
        q_values = [0.0, 0.0]
        new_entry = DataFrame({
            "state_history": [
                state_history
            ],
            "actions": [
                q_values
            ]
        })

        q_table = concat([q_table, new_entry], ignore_index=True)
        return q_values, len(q_table) - 1, q_table


# optionally render the game
def render_game(enabled=False):
    if enabled:
        env.render()
        time.sleep(1 / 45)  # FPS


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

        # history of latest states for velocity information
        state_history = init_state_history()

        # initialize training values
        epochs, reward, = 0, 0
        done = False

        while not done:
            # add states for each episode
            state_history = add_state_to_history(state, state_history)

            # Exploit learned values
            actions, state_index, q_table = get_state_action_in_q_table(
                state_history, q_table)
            # Explore action space
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(actions)

            # perform chosen action
            next_state, _, done, info = env.step(
                action)

            if TRAIN_AGENT:
                # get potential rewards for next state
                future_state_outlook = add_state_to_history(
                    next_state, state_history)

                # penalty for crashing
                if done:
                    reward = -100

                # get current value in q-table
                current_value = actions[action]

                # determine future value estimate
                next_actions, _, q_table = get_state_action_in_q_table(
                    future_state_outlook, q_table)
                future_value_extimate = np.max(next_actions)

                # calculate q-function
                new_value = (1 - ALPHA) * current_value + ALPHA * \
                    (reward + GAMMA * future_value_extimate)

                # update value for chosen action in q-table
                q_table.at[state_index, "actions"][action] = new_value

            # advance state for next loop
            state = next_state
            epochs += 1

            render_game(RENDER_GAME)

            if (LIMIT_EPOCHS != None and epochs >= LIMIT_EPOCHS):
                break

        # for logging
        epoch_history.append(epochs)

        if TRAIN_AGENT:
            q_table.to_csv("q_table.csv", index=False)

        # print result and save current q-table to file every 10 episodes
        if i % EPISODE_REPORT == 0:
            clear_output(wait=True)
            print(
                f"({datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}) States: {len(q_table)} | Episode: {i} | Epochs - min: {min(epoch_history)}, avg: {sum(epoch_history) / len(epoch_history)}, max: {max(epoch_history)}")
            epoch_history = []

    print("Training finished.\n")
