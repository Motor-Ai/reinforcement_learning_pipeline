"""Dataclasses for environment configurations."""

from dataclasses import dataclass


@dataclass
class CarlaEnvConfig:
    """
    Attributes:
        _target_ (str): Target class to instantiate.
        render_camera (bool): If True, spawns a camera sensor and renders its output via Pygame.
        scene_duration (int): Duration of an episode in seconds.
        slowdown_percentage (float): Percentage reduction of the vehicles' speed limit.
        ego_autopilot (bool): Whether the ego vehicle uses autopilot.
        frequency (float): Simulation tick time (in seconds).
        use_custom_map (bool): Whether to load a custom OpenDrive map.
        show_route (bool): Whether to visualize the global route.
        preprocess_observation (bool): Whether to preprocess observations.
        display_width (int): Screen width (if camera rendering is enabled).
        display_height (int): Screen Height (if camera rendering is enabled).
        num_actions (int): Number of maneuvers (if `mai_action_space` is False) 
            or number of actions otherwise.
        n_vehicles (int): Number of non-ego vehicles in the simulation.
        mai_action_space (bool): If True, uses the Motor.AI action space.
        n_actions_per_maneuver (int): Number of actions per maneuver.
    """
    _target_: str = "src.envs.carla_env.CarlaGymEnv"
    render_camera: bool = False
    scene_duration: int = 30
    slowdown_percentage: float = 10.0
    ego_autopilot: bool = False
    frequency: float = 0.1
    use_custom_map: bool = False
    show_route: bool = True
    preprocess_observation: bool = True
    display_width: int = 1080
    display_height: int = 720
    num_actions: int = 3
    n_vehicles: int = 1
    mai_action_space: bool = True
    n_actions_per_maneuver: int = 5