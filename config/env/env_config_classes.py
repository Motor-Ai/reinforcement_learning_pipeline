"""Dataclasses for environment configurations."""

from dataclasses import dataclass


@dataclass
class CarlaEnvConfig:
    """
    Attributes:
        render_camera (bool): If True, spawns a camera sensor and renders its output via Pygame.
        scene_duration (int): Duration of an episode in seconds.
        slowdown_percentage (float): Percentage reduction of the vehicles’ speed limit.
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
    render_camera: bool
    scene_duration: int
    slowdown_percentage: float
    ego_autopilot: bool
    frequency: float
    use_custom_map: bool
    show_route: bool
    preprocess_observation: bool
    display_width: int
    display_height: int
    num_actions: int
    n_vehicles: int
    mai_action_space: bool
    n_actions_per_maneuver: int
