# Reinforcement Learning

This repository provides a reinforcement learning (RL) pipeline for autonomous driving research using the CARLA simulator.


## Getting Started

### Prerequisites

- Python 3.10 (and higher?)
- CARLA 0.9.15 (Download from [here](https://github.com/carla-simulator/carla/releases/tag/0.9.15/))
- See `requirements.txt` for Python dependencies.

### Installation

1. Download and install [CARLA 0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15/). Set up the CARLA_ROOT variable to point to your CARLA directory, and add it to the PYTHONPATH:
    ```bash
    echo 'export CARLA_ROOT="/path/to/my/CARLA_0.9.15"' >> ~/.bashrc
    echo 'export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg"' >> ~/.bashrc
    source ~/.bashrc
    ```
2. Using poetry: install poetry globally, then create venv and install dependencies inside the project folder:
    ```bash
    sudo apt install python3-poetry
    cd <project folder>
    poetry config --local virtualenvs.in-project true
    poetry install
    pip install carla==0.9.15
    ```
    Alternatively: create your virtual env manually and install dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```
3. Link the CARLA 'agents' lib by running:

    ```bash
    ln -s $CARLA_ROOT/PythonAPI/carla/agents/ .venv/lib/python3.10/site-packages/
    ```

    replace `.venv` with the name of your virtual env dir.


### Running the RL Pipeline

To run the project, follow these steps:

1. **Start the CARLA Simulator**  
    Open a terminal and run:
    ```bash
    $CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -quality-level=Low -prefernvidia
    ```
    - `-RenderOffScreen`: Runs CARLA without rendering to a display (useful for headless servers).
    - `-quality-level=Low`: Sets the graphics quality to low for better performance.

2. **Train or Evaluate the RL Agent**  
    In a separate terminal, use the following commands:
    - To train a model:
      ```bash
      python train.py
      ```
    - To evaluate a trained model:
      ```bash
      python eval.py
      ```

3. **Configuration**  
    Adjust environment and training settings in `envs/configs/config.yaml` as needed.

## Features

- **CARLA Environment Integration:** Custom wrappers and rendering for RL agents (`envs/carla_env.py`, `envs/carla_env_render.py`).
- **Observation Processing:** Includes vectorized BEV observers and traffic rule decision modules (`envs/observation/`).
- **Traffic Rule Handling:** Supports STOP, Yield, Pedestrian Crossing, Lane Relations, Speed Limit, and Lane Priority (`envs/observation/decision_traffic_rules/README.md`).
- **Modeling:** Preprocessing and DIPP predictor utilities (`models/`).
- **Logging:** Training logs and results are saved in `saved_rl_models/`.

## Directory Overview
- `train.py`: Main script to train the reinforcement learning agent in the CARLA environment.
- `eval.py`: Script to evaluate a trained RL agent and report performance metrics.
- `envs/`: Environment wrappers, configs, and observation modules.
- `models/`: Model definitions and utilities.
- `saved_rl_models/`: Logs and saved model results.

## Results

- While creating the 

## TODOs

- Update Action space (currently a dummy action space is being used)
- Update configs
- Add Welf's vis as rendering
- ...