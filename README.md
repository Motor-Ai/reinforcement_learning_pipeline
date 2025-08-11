# Reinforcement Learning

This repository provides a reinforcement learning (RL) pipeline for autonomous driving research using the CARLA simulator.

## Project Structure

```
.
├── main.py
├── requirements.txt
├── envs/
│   ├── callbacks.py
│   ├── carla_env.py
│   ├── carla_env_render.py
│   ├── configs/
│   │   └── config.yaml
│   ├── observation/
│   │   ├── __init__.py
│   │   ├── tr_costs.py
│   │   ├── vector_BEV_observer.py
│   │   └── decision_traffic_rules/
│   │       ├── feature_indices.py
│   │       ├── lanelet_data_extractor.py
│   │       ├── lanelet_traffic_rules.py
│   │       └── README.md
├── models/
│   ├── __init__.py
│   ├── preprocess.py
│   └── dipp_predictor_py/
│       ├── __init__.py
│       ├── dipp_carla.py
│       └── dipp_predictor_utils.py
├── saved_rl_models/
│   ├── log.txt
│   └── results.txt
```

## Getting Started

### Prerequisites

- Python 3.7+
- See `requirements.txt` for Python dependencies.

### Installation

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Configure environment variables and settings as needed in `envs/configs/config.yaml`.

### Running the RL Pipeline

Start training or evaluation by running:
```bash
python main.py
```

## Features

- **CARLA Environment Integration:** Custom wrappers and rendering for RL agents (`envs/carla_env.py`, `envs/carla_env_render.py`).
- **Observation Processing:** Includes vectorized BEV observers and traffic rule decision modules (`envs/observation/`).
- **Traffic Rule Handling:** Supports STOP, Yield, Pedestrian Crossing, Lane Relations, Speed Limit, and Lane Priority (`envs/observation/decision_traffic_rules/README.md`).
- **Modeling:** Preprocessing and DIPP predictor utilities (`models/`).
- **Logging:** Training logs and results are saved in `saved_rl_models/`.

## Directory Overview

- `main.py`: Entry point for training/evaluation.
- `envs/`: Environment wrappers, configs, and observation modules.
- `models/`: Model definitions and utilities.
- `saved_rl_models/`: Logs and saved model results.

## License

Specify your license here.

## Acknowledgements

- [CARLA Simulator](https://carla.org/)