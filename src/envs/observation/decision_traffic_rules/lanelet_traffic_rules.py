import stlcg
import numpy as np
from stlcg import Expression
import torch
import lanelet2

from lanelet2.core import (
    AllWayStop,
    AttributeMap,
    BasicPoint2d,
    BoundingBox2d,
    Lanelet,
    LaneletMap,
    LaneletWithStopLine,
    LineString3d,
    Point2d,
    Point3d,
    RightOfWay,
    TrafficLight,
    getId,
)

from lanelet2.projection import (
    UtmProjector,
    GeocentricProjector,
    LocalCartesianProjector,
    MercatorProjector,
)

from collections import namedtuple

from dipp_planner_py.train_utils import project_to_frenet_frame, bicycle_model

# Note for STLCG
# the interval in STLCG is describing the INDICES of the desired time interval.
# The user is responsible for converting the time interval (in time units) into
# indices (integers) using knowledge of the time step size

# Class template of Vehicle State
VehState = namedtuple("VehState", ("x", "y", "theta", "v", "a", "D"))

# Parameters
V_ERR = Expression("V_ERR", torch.Tensor([00.1]).requires_grad_(False))  # m/s
D_SL = Expression("D_SL", torch.Tensor([5.0]).requires_grad_(False))  # m
D_BR = Expression("D_BR", torch.Tensor([5.0]).requires_grad_(False))  # m
T_SLW = 3.0  # indexes Time is pause at an intersection stop sign
TRUE_expr = Expression("TRUE", torch.Tensor([True]).requires_grad_(False))
FALSE_expr = Expression("FALSE", torch.Tensor([False]).requires_grad_(False))


# def speed_predicate(velocity_estimate, velocity_groundtruth):
#     # STL Logic
#     speed_lower_limit = torch.tensor(10.0, requires_grad=False)
#     speed_upper_limit = torch.tensor(100.0, requires_grad=False)

#     velocity_estimate = Expression("velocity_estimate", velocity_estimate)
#     speed_lower_limit = Expression("speed_lower_limit", speed_lower_limit)
#     speed_upper_limit = Expression("speed_upper_limit", speed_upper_limit)

#     lt = speed_lower_limit <= velocity_estimate
#     gt = velocity_estimate <= speed_upper_limit

#     speed_control_formula = stlcg.Always(lt & gt)
#     speed_error = (
#         speed_control_formula.robustness(velocity_estimate, scale=1).mean() ** 2
#     )
#     return speed_error


# Implementation of Lane Info extractions from "Formalization of Intersection Traffic Rules in Temporal Logic"
# https://mediatum.ub.tum.de/doc/1664592/uw2i3i5kwjh3w4ezek0qov5og.Maierhofer-2022-IV.pdf


def successor(graph, current_lane):
    following_lanes = graph.following(current_lane)
    return following_lanes


def predessor(graph, current_lane):
    preceding_lanes = graph.previous(current_lane)
    return preceding_lanes


def adjacent_left(graph, current_lane):
    left_lanes = graph.adjacent_left(current_lane)
    return left_lanes


def adjacent_right(graph, current_lane):
    right_lanes = graph.adjacent_right(current_lane)
    return right_lanes


# Lanes: Series of LaneLets involving lane changes to reach a destination
def predessor_lanes(graph, current_lane):
    # TODO : Unable to find the right implementation
    predessor_lanes = []
    previous = predessor(graph, current_lane)
    recorded_lane = [current_lane]
    count = 0
    while len(previous) > 0 or count < 10:
        for lane in previous:
            if lane not in recorded_lane:
                recorded_lane.append(lane)
                predessor_lanes.append(lane)

        count += 1
        previous = predessor(graph, lane)

    return predessor_lanes


def successor_lanes():
    # TODO : Unable to find the right implementation
    pass


def lanes():
    lanes = predessor_lanes() + successor_lanes()
    return lanes


def reach(lanelet, maxRoutingCost=300, allowLaneChanges=False):
    # Determines which lanelets can be reached from a give start lanelets within a given amount of routing cost
    return graph.reachableSet(
        lanelet, maxRoutingCost, 0, allowLaneChanges=allowLaneChanges
    )


def stop_line(route_map, lane):
    # TODO: Stop line must be in the Lanelet layer too! Can be done by adding relations
    for elem in route_map.lineStringLayer:
        if elem.attributes["type"] == "stop_line":
            if elem.id in [lane.rightBound.id, lane.leftBound.id]:
                p1 = (elem[0].x, elem[0].y)
                p2 = (elem[1].x, elem[1].y)
                return p1, p2, elem
        else:
            return None


def min_dist(pose: lanelet2.core.Point3d, linestring: lanelet2.core.LineString3d):
    return lanelet2.geometry.distance(linestring, pose)


def traffic_sign(lanelet, sign_type):
    for reg_elem in lanelet.regulatoryElements:
        if reg_elem.attributes["type"] == sign_type:
            return TRUE_expr
        else:
            return FALSE_expr


def traffic_sign_type(lanelet):
    if len(lanelet.regulatoryElements) > 0:
        return lanelet.regulatoryElements
    else:
        return None


def argmin_s(traffic_sign_ID):
    pass


def get_priority(eval_idx, direction):
    pass


# Non-Learnable Function
def get_traffic_light(lanelet):
    for reg_elem in lanelet.regulatoryElements:
        if reg_elem.attributes["subtype"] == "traffic_light":
            return reg_elem
        else:
            return None


def direction(traffic_light_elem):
    # TODO: use a direction tag to model traffic lights that have arrows
    pass


def traffic_light_state(traffic_light):
    # TODO: query traffic light state from SU?
    # "Red", "Yellow", "Green", "Inactive"
    pass


def traffic_light_direction(traffic_light):
    # TODO: query traffic light direction from SU?
    # "left", "right", "straight", "leftStraight", "rightStraight", "leftRight", "all"
    return "right"


def intersection(current_lane, graph):
    # TODO: Encode the lane connectivity tyoe ad subtype in the map
    pass


def incoming_lane(current_lane, graph):
    return graph.previous(current_lane)[0]


def oncoming_lane(current_lane, graph):
    oncoming_lanes = []
    if len(current_lane.Lefts(current_lane)) > 0:
        for lane in current_lane.Lefts(current_lane):
            oncoming_lanes.append(graph.previous(lane))
        return oncoming_lanes
    else:
        return None


def inc_la_left_of(current_lane, graph):
    pass


def line_in_front(pose: VehState, line: lanelet2.core.LineString3d):
    # TODO: Reimplemetation required
    sp = projector.forward(
        lanelet2.core.GPSPoint(pose.x, pose.y, 0)
    )  # required based on the coordinate system
    plocal = lanelet2.geometry.to2D(sp)
    # nearest = lanelet2.geometry.findNearest(map.laneletLayer, plocal, 1)
    dist = lanelet2.geometry.distance(line, sp)
    if dist > 0:
        return True
    else:
        return False


def lanelets_dir(pose: VehState, graph: lanelet2.routing.Route):
    lanelets_in_dir = []
    sp = projector.forward(
        lanelet2.core.GPSPoint(pose.x, pose.y, 0)
    )  # required based on the coordinate system
    plocal = lanelet2.geometry.to2D(sp)
    nearest = lanelet2.geometry.findNearest(map.laneletLayer, plocal, 1)
    inside = lanelet2.geometry.inside(nearest[0][1], plocal)
    if inside:
        current_lane = nearest[0][1]
        lanelets_in_dir.append(current_lane)
        for succeeding_lane in graph.following(current_lane):
            lanelets_in_dir.append(succeeding_lane)
        return lanelets_in_dir
    else:
        raise ValueError("One of the poses is outside the lanelets")


# TBD
def lanelets_dir(pose: VehState, path: lanelet2.routing.LaneletPath):
    # occupied lanelets based on the driving direction of the vehicle

    lanelets_in_dir = []
    # sp = projector.forward(
    #     lanelet2.core.GPSPoint(pose.x, pose.y, 0)
    # )  # required based on the coordinate system
    # plocal = lanelet2.geometry.to2D(sp)
    # nearest = lanelet2.geometry.findNearest(map.laneletLayer, plocal, 1)
    # inside = lanelet2.geometry.inside(nearest[0][1], plocal)
    # if inside:
    for lane in path:
        lanelets_in_dir.append(lane)
    return lanelets_in_dir
    # else:
    #     raise ValueError("One of the poses is outside the lanelets")


def active_tls_by_lanelet(lane):
    traffic_light = traffic_light(lane)
    status = traffic_light_state(traffic_light)
    if status == "inactive":
        return False
    else:
        return True


def in_standstill(pose: VehState):
    vel = Expression("vel", torch.abs(torch.Tensor([pose.v])))
    return stlcg.LessThan(lhs=vel, val=V_ERR)


def relevant_traffic_light(ego_pose):
    for lane in lanelets_dir(ego_pose, path):
        for l in reach(lane):
            if active_tls_by_lanelet(l):
                return TRUE_expr
            else:
                continue
    return FALSE_expr


def at_traffic_sign(pose: VehState, sign_type: str):
    for lane in lanelets_dir(pose, path):  # How does the gradient work here?
        if traffic_sign(lane, sign_type).value > 0.0:
            return TRUE_expr
        else:
            continue
    return FALSE_expr


def active_tl(pose: VehState):
    active_tl = []
    for lane in lanelets_dir(pose, path):
        if active_tls_by_lanelet(lane):
            active_tl.append(lane)

    return active_tl


def going_straight(pose: VehState):
    # Is the vehicle going straight
    for lane in path:
        if lane.attributes["type"] == "straight":
            return TRUE_expr
        else:
            continue
    return FALSE_expr

    sp = projector.forward(
        lanelet2.core.GPSPoint(pose.x, pose.y, 0)
    )  # required based on the coordinate system
    plocal = lanelet2.geometry.to2D(sp)
    nearest = lanelet2.geometry.findNearest(map.laneletLayer, plocal, 1)
    inside = lanelet2.geometry.inside(nearest[0][1], plocal)
    if inside:
        current_lane = nearest[0][1]
        if current_lane.attributes["type"] == "straight":
            return True
        else:
            return False


def turning_left(pose: VehState, path: lanelet2.routing.LaneletPath):
    # Is the vehicle turning left
    for lane in path:
        if lane.attributes["type"] == "left":
            return TRUE_expr
        else:
            continue
    return FALSE_expr
    # sp = projector.forward(
    #     lanelet2.core.GPSPoint(pose.x, pose.y, 0)
    # )  # required based on the coordinate system
    # plocal = lanelet2.geometry.to2D(sp)
    # nearest = lanelet2.geometry.findNearest(map.laneletLayer, plocal, 1)
    # inside = lanelet2.geometry.inside(nearest[0][1], plocal)
    # if inside:
    #     current_lane = nearest[0][1]
    #     if current_lane.attributes["type"] == "left":
    #         return True
    #     else:
    #         return False


def turning_right(pose: VehState):
    # Is the vehicle turning right
    for lane in path:
        if lane.attributes["type"] == "right":
            return TRUE_expr
        else:
            continue
    return FALSE_expr
    sp = projector.forward(
        lanelet2.core.GPSPoint(pose.x, pose.y, 0)
    )  # required based on the coordinate system
    plocal = lanelet2.geometry.to2D(sp)
    nearest = lanelet2.geometry.findNearest(map.laneletLayer, plocal, 1)
    inside = lanelet2.geometry.inside(nearest[0][1], plocal)
    if inside:
        current_lane = nearest[0][1]
        if current_lane.attributes["type"] == "right":
            return True
        else:
            return False


# Non-Learnable property
def at_traffic_light(pose: VehState, direction: str, color: str):
    for lane in active_tl(pose):
        lane_traffic_light = get_traffic_light(lane)
        if (
            traffic_light_state(lane_traffic_light) == color
            and traffic_light_direction(lane_traffic_light)
            == direction  # Is it required to check the belong to a subset check here?
        ):
            return TRUE_expr
        else:
            continue
    return FALSE_expr


def type(lane):
    # Intersection type and its labels
    # "incoming", "intersection", "oncoming", "outgoingLeft", "outgoingRight", "outgoingStraight", "leftTurn", "rightTurn", "goingStraight"
    return lane.attributes[
        "type"
    ]  # TODO: This cannot be used direct. Need more types and subtypes


def trans(path: lanelet2.core.LineString3d, pose: VehState):
    # Compute the longitudinal position along the reference path
    # TBD: path must be in Frenet Frame?

    interpolated_path = []
    distance_to_pose = []

    for i in range(0, 100, 1):
        # Interpolate the path
        basic_point = lanelet2.geometry.interpolatedPointAtDistance(path, i * 0.1)
        interpolated_path.append(
            lanelet2.core.Point3d(0, basic_point.x, basic_point.y, 0)
        )
        # Distance of the interpolated point to the intended pose
        distance_to_pose.append(lanelet2.geometry.distance(interpolated_path[i], pose))

    min_dist_to_pose = np.argmin(distance_to_pose)
    lane_chunk = lanelet2.core.LineString3d(points=interpolated_path[:min_dist_to_pose])
    distance_to_pose = lanelet2.geometry.length(lane_chunk)
    return distance_to_pose


def ref_path_lanelets(pose: VehState, path: lanelet2.routing.LaneletPath):
    sp = projector.forward(
        lanelet2.core.GPSPoint(pose.x, pose.y, 0)
    )  # required based on the coordinate system
    plocal = lanelet2.geometry.to2D(sp)
    nearest = lanelet2.geometry.findNearest(map.laneletLayer, plocal, 1)
    inside = lanelet2.geometry.inside(nearest[0][1], plocal)
    if inside:
        current_lane = nearest[0][1]

    remaining_lane = path.remaining_lane(current_lane)
    return remaining_lane


# TBD https://mediatum.ub.tum.de/doc/1575358/2n3lb03tprh28cbg1n19faj8w.Koschi_TIV2020_final.pdf
def overappr_braking_pos(pose: VehState, min_acc: float = 0.5):
    # frontmost point in the Cartesian coordinate frame of the over-approximated reachable set
    lanes = ref_path_lanelets(pose, path)
    # sp = projector.forward(
    #     lanelet2.core.GPSPoint(pose.x, pose.y, 0)
    # )  # required based on the coordinate system
    # plocal = lanelet2.geometry.to2D(sp)
    # nearest = lanelet2.geometry.findNearest(map.laneletLayer, plocal, 1)
    # inside = lanelet2.geometry.inside(nearest[0][1], plocal)
    # if inside:
    #     current_lane = nearest[0][1]
    trans(lanes[0].centerline, pose)


def front(path: lanelet2.routing.LaneletPath, pose: VehState):
    point = lanelet2.core.Point3d(0, pose.x, pose.y, 0)
    pp = []
    for lane in path:
        for c in lane.centerline:
            pp.append(lanelet2.core.Point3d(0, c.x, c.y, 0))
    path_centerpoints = lanelet2.core.LineString3d(id=1, points=pp)
    dist = lanelet2.geometry.distance(path_centerpoints, point)
    return lanelet2.geometry.nearestPointAtDistance(path_centerpoints, dist)


def rear(path: lanelet2.routing.LaneletPath, pose: VehState):
    return front(path, pose)


# Learnable property
def causes_braking_intersection(pose_k: VehState, pose_p: VehState):
    rear_k = rear(path, pose_k)
    front_p = front(path, pose_p)
    dist = np.hypot(rear_k.x - front_p.x, rear_k.y - front_p.y)
    stlcg.LessThan("dist", val=D_BR)

    pass


def braking_intersection_possible(pose: VehState):
    for lane in lanelets_dir(pose, path):
        for l in reach(lane):
            if not (
                "incoming" in type(l)
                and (
                    trans(path, lane.leftBound[-1])
                    > trans(path, overappr_braking_pos(pose, 0.5))
                )
            ):
                return False

    return True


def on_lanelet_with_type(pose: VehState, lanelet_type: str):
    sp = projector.forward(
        lanelet2.core.GPSPoint(pose.x, pose.y, 0)
    )  # required based on the coordinate system
    plocal = lanelet2.geometry.to2D(sp)
    nearest = lanelet2.geometry.findNearest(map.laneletLayer, plocal, 1)
    inside = lanelet2.geometry.inside(nearest[0][1], plocal)
    if inside:
        current_lane = nearest[0][1]
        if current_lane.attributes["type"] == lanelet_type:
            return True
        else:
            return False


# Implementation of STL predicates for Intersection rules
def stop_sign_rule_robustness(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)  # batch x Ts x [acc, steer]
    ref_line = aux_vars[0].tensor
    current_state = aux_vars[1].tensor[:, 0]
    traj = bicycle_model(
        control, current_state
    )  # [x, y, heading, v_x, v_y, length, width, height]
    rule = stop_sign_rule(current_state, ref_line)
    rho = rule.robustness(control)
    return torch.relu(-rho).mean()


# Rule R-IN1
# %%
def stop_sign_rule(current_state, ref_line):
    lhs1 = stlcg.And(
        subformula1=passing_stop_line(current_state),
        subformula2=at_traffic_sign(current_state, "stop_sign"),
    )
    lhs = stlcg.And(
        subformula1=lhs1,
        subformula2=stlcg.Negation(relevant_traffic_light(current_state)),
    )

    rhs1 = stlcg.And(
        subformula1=stop_line_in_front(current_state),
        subformula2=in_standstill(current_state),
    )
    rhs2 = stlcg.Always(subformula=rhs1, interval=[0, T_SLW])
    rhs = stlcg.Eventually(subformula=rhs2, interval=[-2.0, 0.0])
    return stlcg.Implies(subformula1=lhs, subformula2=rhs)


def passing_stop_line(pose: VehState):
    stop_line_in_front_expr = stop_line_in_front(pose)
    neg_stop_line_in_front = stlcg.Negation(subformula=stop_line_in_front_expr)
    not_crossed = stlcg.Eventually(
        subformula=neg_stop_line_in_front, interval=[t, t + 2.0]
    )
    return stlcg.And(subformula1=stop_line_in_front_expr, subformula2=not_crossed)


def stop_line_in_front(pose: VehState):
    # stop_line = stop_line(lane)
    # min_dist = Expression("min_dist", value=min_dist(pose, stop_line))
    # D_SL = Expression("D_SL", value=5.0)
    # distance_within_threshold = min_dist <= D_SL

    # line_in_front = Expression("line_in_front", value=line_in_front(pose, stop_line))

    # stlcg.And(subformula1=distance_within_threshold, subformula2=line_in_front)

    # for lane in lanelets_dir(pose, path):
    #     stop_line = stop_line(lane)
    #     if min_dist(pose, stop_line) < D_SL and line_in_front(pose, stop_line):
    #         return True
    #     else:
    #         continue
    # return None

    all_lanes = lanelets_dir(pose, path)
    for lane in all_lanes:
        stop_line_expr = stop_line(lane)
        lhs = stlcg.LessThan(
            subformula1=min_dist(pose, stop_line_expr), subformula2=D_SL
        )
        rhs = line_in_front(pose, stop_line_expr)
        if lhs.value > 0.0 and rhs.value > 0.0:
            return stlcg.And(subformula1=lhs, subformula2=rhs)
        else:
            continue
    return FALSE_expr


# %%
# Rule R-IN2
def waiting_at_traffic_light(pose_ego: VehState):
    pass


# %%
#
example_file = "tegel_map/Decision_tegel_map_18_04_2024_with_centerlines_processed.osm"
projector = UtmProjector(lanelet2.io.Origin(52.55754329939843, 13.281288759978164))
# gc_projector = GeocentricProjector(lanelet2.io.Origin(52.55644, 13.2751107))
map = lanelet2.io.load(example_file, projector)
traffic_rules = lanelet2.traffic_rules.create(
    lanelet2.traffic_rules.Locations.Germany,
    lanelet2.traffic_rules.Participants.Vehicle,
)
graph = lanelet2.routing.RoutingGraph(map, traffic_rules)

sp = projector.forward(lanelet2.core.GPSPoint(52.5571716, 13.2796366, 0))
plocal = lanelet2.geometry.to2D(sp)
sample_point = lanelet2.core.Point3d(0, sp.x, sp.y, 0)

min_distance = 10000000000000

sp = projector.forward(lanelet2.core.GPSPoint(52.5571716, 13.2796366, 0))
start_point = lanelet2.core.Point3d(0, sp.x, sp.y, 0)

ep = projector.forward(lanelet2.core.GPSPoint(52.5579939, 13.2813646, 0))
end_point = lanelet2.core.Point3d(1, ep.x, ep.y, 0)


# Find the start lane
for lanelet in map.laneletLayer:
    distance = lanelet2.geometry.distance(lanelet.centerline, start_point)
    if distance < min_distance:
        min_distance = distance
        start_lanelet = lanelet

print("Start lane")
print(start_lanelet.attributes["name"])

min_distance = 10000000000000
# Find the end lane
for lanelet in map.laneletLayer:
    distance = lanelet2.geometry.distance(lanelet.centerline, end_point)
    if distance < min_distance:
        min_distance = distance
        end_lanelet = lanelet

print("Goal lane")
print(end_lanelet.attributes["name"])

path = graph.shortestPath(start_lanelet, end_lanelet)
