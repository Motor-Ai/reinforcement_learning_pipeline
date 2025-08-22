import carla
import numpy as np
import math

def detect_lanelet_priority(turnleft=False, turnright=False, yieldsign=False, stopsign=False, rightofwaysign=False, prioritysign=False):
    """
    Returns a priority number between 1 and 9. The larger the number, the earlier the vehicle is allowed to drive
    Only includes lanelet priority and trajectory (no right before left rule)
    """

    if yieldsign or stopsign: # low priority
        if turnleft:
            return 1
        elif turnright:
            return 2
        else:
            return 3
    elif rightofwaysign or prioritysign: # high priority
        if turnleft:
            return 7
        elif turnright:
            return 8
        else:
            return 9
    else: # medium priority
        if turnleft:
            return 4
        elif turnright:
            return 5
        else:
            return 6

def get_turning_intention(vehicle: carla.Vehicle) -> tuple[bool, bool]:
    """
    Get the turning intention of vehicle at intersection
    """
    turnleft = False
    turnRight = False
    if vehicle.get_light_state() == carla.VehicleLightState.LeftBlinker:
        turnleft = True
    if vehicle.get_light_state() == carla.VehicleLightState.RightBlinker:
        turnRight = True
    return turnleft, turnRight

def make_valid_orientation(angle):
    TWO_PI = 2.0 * np.pi
    angle = angle % TWO_PI
    if np.pi <= angle <= TWO_PI:
        angle = angle - TWO_PI
    assert -np.pi <= angle <= np.pi
    return angle

def intersection_relative_priority(ego, other, carla_map, ego_priority=0, other_priority=0):
    """
    Relative priority of the other car wrt ego {-1, 0, 1} for {yield, same, priority}
    """
    other_rel_priority = 0
    other_wp = carla_map.get_waypoint(other.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
    ego_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))

    # if other is in intersection, it has priority
    if other_wp.is_junction and not ego_wp.is_junction: # TODO handle case when both already in junction?
        # print("other is in intersection and has priority")
        other_rel_priority = 1
        return other_rel_priority

    # similar priorities: right before left rule
    if ego_priority == ego_priority or ((ego_priority in [4,5,6]) and (other_priority in [4,5,6])):
        ego_heading = (ego.get_transform().rotation.yaw * math.pi / 180)
        other_heading = (other.get_transform().rotation.yaw * math.pi / 180)
        rel_orientation = -make_valid_orientation(ego_heading - other_heading)
        # print(ego_heading, other_heading, rel_orientation, np.pi / 2, 1.5 * np.pi)
        # TODO the following 4 cases can be summarized, just a matter of sign checking
        if np.isclose(rel_orientation, np.pi / 2, atol=np.pi * 1 / 6):
            other_rel_priority = 1
            # print("similar priority, other has right-before-left priority 1")
            return other_rel_priority
        elif np.isclose(rel_orientation, 1.5 * np.pi, atol=np.pi * 1 / 6):
            other_rel_priority = -1
            # print("similar priority, ego has right-before-left priority 2")
            return other_rel_priority
        elif np.isclose(-rel_orientation, np.pi / 2, atol=np.pi * 1 / 6):
            other_rel_priority = -1
            # print("similar priority, ego has right-before-left priority 3")
            return other_rel_priority
        elif np.isclose(-rel_orientation, 1.5 * np.pi, atol=np.pi * 1 / 6):
            other_rel_priority = 1
            # print("similar priority, other has right-before-left priority 4")
            return other_rel_priority
        else:
            # print("similar priority. right-before-left does not apply")
            pass

    # different priorities
    if other_priority < ego_priority:
         other_rel_priority = -1
        #  print("ego has higher priority")
    elif other_priority > ego_priority:
         other_rel_priority = 1
        #  print("other has higher priority")

    return other_rel_priority


def get_rel_prio_intersection(ego, other, carla_map):
    # TODO check if at same intersection and close enough to intersection
    # world modelling approach: check for interaction of the trajectories after it happened

    # turning intention
    ego_turnleft, ego_turnright = get_turning_intention(ego)
    other_turnleft, other_turnright = get_turning_intention(other)

    # print("Ego left turn:", ego_turnleft, "right turn:", ego_turnright)
    # print("Other left turn:", other_turnleft, "right turn:", other_turnright)

    # lane priority
    ego_priority = detect_lanelet_priority(
        turnleft=ego_turnleft,
        turnright=ego_turnright,
        yieldsign=False, # todo check if there is a sign in the map_lane in front of the car
        stopsign=False, # todo check if there is a sign in the map_lane in front of the car
        rightofwaysign=False, # todo check if there is a sign in the map_lane in front of the car
        prioritysign=False, # todo check if there is a sign in the map_lane in front of the car
    )
    other_priority = detect_lanelet_priority(
        turnleft=other_turnleft,
        turnright=other_turnright,
        yieldsign=False, # todo check if there is a sign in the map_lane in front of the car
        stopsign=False, # todo check if there is a sign in the map_lane in front of the car
        rightofwaysign=False, # todo check if there is a sign in the map_lane in front of the car
        prioritysign=False, # todo check if there is a sign in the map_lane in front of the car
    )

    # print("ego_priority", ego_priority)
    # print("other_priority", other_priority)

    # relative priority of other at intersection
    other_relative_priority = intersection_relative_priority(ego, other, carla_map, ego_priority, other_priority)
    return other_relative_priority