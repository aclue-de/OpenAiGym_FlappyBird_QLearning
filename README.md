# Setup
- this project uses python 3.9
- create a virtual env for the project: `python -m venv .venv`
- to use the virtual env, execute: `.\.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux) as well as `deactivate` (both) to stop
- install dependencies: `pip install -r requirements.txt`
- run `python train_agent.py` to train your agent

# Evaluate a trained agent
- in `train_agent.py`: switch `TRAIN_AGENT` to `False` and `RENDER_GAME` to `True`
- run `python train_agent.py`

## note to Windows users
- OpenAI Gym doesn't fully support graphical interfaces for Windows
- Follow [this tutorial](https://research.wmz.ninja/articles/2017/11/setting-up-wsl-with-graphics-and-audio.html) if you have issues with opening a game window

### for repeated startup
- start pulseaudio under Windows
- start XLaunch under Windows
- execute `export DISPLAY=:0.0` and `export PULSE_SERVER=tcp:localhost` under Linux

## note to Mac (M1) users
You need to install the local version of the `flappy-bird-gym` library, because of version incompabilities.

1. `pip install --upgrade pip`
2. `cd flappy-bird-gym`
3. `pip install .`

After that install the remaining requirements with `pip install autopep8 IPython pandas`.
