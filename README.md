# Reinforcement Learning

This repository provides a reinforcement learning (RL) pipeline for autonomous driving research using the CARLA and GPUDrive simulators.

## Getting Started

### Prerequisites

- Python 3.11 (3.12 should also work)
- CARLA 0.9.16 (Download from [here](https://github.com/carla-simulator/carla/releases/tag/0.9.16/))

### Installation

1. Clone the repo recursively:
    ```bash
    git clone --recursive https://github.com/Motor-Ai/reinforcement_learning_pipeline.git
    ```
    you can update GPUDrive later using:
    ```bash
    git submodule update --init --recursive
    ```


2. Install uv:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    then install the virtual env:
    ```bash
    cd ./reinforcement_learning_pipeline
    uv sync
    ```


2. **Build gpudrive and the docker**  
   Build the gpudrive library and install its python dependencies inside a virtual environment:
    ```bash
    make gpudrive_build
    ```
   Build the docker image:
    ```bash
    make docker_build
    ```

### Install CARLA

1. Download and install [CARLA 0.9.16](https://github.com/carla-simulator/carla/releases/tag/0.9.16/). Set up the CARLA_ROOT variable to point to your CARLA directory, and install carla on python:
    ```bash
    echo 'export CARLA_ROOT="/path/to/my/CARLA_0.9.16"' >> ~/.bashrc
    source ~/.bashrc
    uv pip install ${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.16-cp311-cp311-manylinux_2_31_x86_64.whl
    ```

2. Link the CARLA 'agents' lib by running:

    ```bash
    ln -s $CARLA_ROOT/PythonAPI/carla/agents/ .venv/lib/python3.11/site-packages/
    ```

    replace `.venv` with the name of your virtual env dir.


### Running the RL Pipeline with CARLA

To run the project, follow these steps:

1. **Start the CARLA Simulator**  
    Open a terminal and run:
    ```bash
    $CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -quality-level=Low -prefernvidia
    ```
    - `-RenderOffScreen`: Runs CARLA without rendering to a display (useful for headless servers).
    - `-quality-level=Low`: Sets the graphics quality to low for better performance.

2. **Train or Evaluate the RL Agent**
    Run the docker:
    ```bash
    make docker_run
    ```
    Run the training script:
    ```bash
    uv run python3 train.py
    ```
    Or the evaluation script:
    ```bash
    uv run python3 eval.py
    ```

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