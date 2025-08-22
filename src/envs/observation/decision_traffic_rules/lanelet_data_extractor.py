import lanelet2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from lanelet2.core import BasicPoint2d, BasicPoint3d, LineString3d, GPSPoint, RightOfWay
import torch
import os
import sys
import lanelet2.core
from lanelet2.projection import (
    UtmProjector,
    GeocentricProjector,
    LocalCartesianProjector,
    MercatorProjector,
)

import lanelet2.routing

PATH = os.path.dirname(os.path.realpath(__file__)).split("decision_traffic_rules")[0]
sys.path.append(PATH)

from decision_traffic_rules.traffic_sign_db import (
    traffic_feat_idx,
    lane_type,
    warning_ts_encoding,
    regulatory_ts_encoding,
    directional_ts_encoding,
    installations_ts_encoding,
)


class LaneletDataExtractor:
    def __init__(self, file_path, origin, proj_type) -> None:
        self.lane_dist_threshold = 0.15  # 20
        self.file = file_path
        self.origin = origin
        self.proj = proj_type
        self.has_map = bool(self.file and self.origin)
        if self.has_map:
            self.map, self.proj = self._get_lanelet_map(
                file_path, origin, proj_type="utm"
            )
        self.traffic_rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Vehicle,
        )
        self.graph = lanelet2.routing.RoutingGraph(self.map, self.traffic_rules)

    def _get_lanelet_map(self, filepath: str, origin: tuple[float], proj_type=None):
        # Origin needs to be passed in lon-lat order
        if proj_type == "utm":
            proj = UtmProjector(lanelet2.io.Origin(*origin))
        elif proj_type == "merc":
            proj = MercatorProjector(lanelet2.io.Origin(*origin))
        else:
            proj = lanelet2.io.Origin(*origin)
        map = lanelet2.io.load(filepath, proj)
        self.filepath = filepath
        return map, proj

    def set_map_in_ego_pos(self, ego_pos: list[float]):
        x, y = ego_pos[:2]
        pos2d = BasicPoint3d(float(x), float(y), 0.0)
        GPS_loc = self.proj.reverse(pos2d)
        proj = lanelet2.io.Origin(GPS_loc)
        self.map = lanelet2.io.load(self.filepath, proj)
        return self.map

    def get_lanelet_at_map_pos(self, pos: list[float]) -> lanelet2.core.Lanelet:
        try:
            x, y = pos[:2]
            pos2d = BasicPoint2d(float(x), float(y))
        except TypeError:  # pos is already in lanelet2 format
            pos2d = pos

        try:
            nearest_lanelet = lanelet2.geometry.findWithin2d(
                self.map.laneletLayer, pos2d, self.lane_dist_threshold
            )[0][1]
        except IndexError:  # point is outside of any lanelet
            nearest_lanelet = self.map.laneletLayer.nearest(pos2d, 1)[
                0
            ]  # this sometimes produces weird results, but it returns a lanelet when the other method can't be used

        return nearest_lanelet

    def get_traffic_rules_at_pos(self, point: torch.tensor, keys):
        lanelet = self.get_lanelet_at_map_pos(point[:2])

        # LaneID
        point[traffic_feat_idx["lane_id"]] = lanelet.id

        # Map is always interpolated by SU
        point[traffic_feat_idx["interpolating"]] = 1

        # try:  # TODO: Remove try...catch...
        # Regulations over the whole lanelet
        for regelem in lanelet.regulatoryElements:
            for key in keys:
                if key == "yield_right_of_way" and key in regelem.attributes.values():
                    if regelem.type() == "de205":
                        point[traffic_feat_idx["priority"]] = regulatory_ts_encoding[
                            regelem.attributes["name"]
                        ]
                        yield_to = (
                            regelem.parameters["refers"][0]
                            .attributes["yield"]
                            .split(",")
                        )
                        # ID = 0 #TODO: Use this after mapping team implements the relevant attributes
                        # for prio_lane in yield_to:
                        #     point[traffic_feat_idx["yield_to_1"] + ID] = int(prio_lane)
                        #     ID += 1

                    else:
                        point[traffic_feat_idx["priority"]] = 1

                if key == "speed_limit" and key in regelem.attributes.values():
                    try:
                        point[traffic_feat_idx[key]] = regulatory_ts_encoding[
                            regelem.attributes["name"]
                        ]
                    except KeyError:
                        print(
                            f"Key {regelem.attributes['name']} not found in regulatory_ts_encoding. Skipping..."
                        )
                if (
                    key == "traffic_sign_directional"
                    and "traffic_sign" in regelem.attributes.values()
                ):
                    if regelem.type() == "de209" or regelem.type() == "de209-10":
                        point[traffic_feat_idx["traffic_sign_directional"]] = (
                            directional_ts_encoding[regelem.attributes["name"]]
                        )

        # Lane Boundary types
        # removed centerline because it should not have a marking
        try:
            point[traffic_feat_idx["ll_type"]] = lane_type[
                lanelet.leftBound.attributes["lane_marking"]
            ]
            point[traffic_feat_idx["rl_type"]] = lane_type[
                lanelet.rightBound.attributes["lane_marking"]
            ]
        except KeyError:  # lane_marking not present in attributes
            pass

        # Lane connectivity
        # Successors
        ID = 0
        for relation in self.graph.followingRelations(lanelet):
            point[traffic_feat_idx["successor_laneID_1"] + ID] = relation.lanelet.id
            ID += 1

        return point

    def find_closest_idx(self, route, point):
        diff = route[:, :2] - point
        idx = np.hypot(diff[:, 0], diff[:, 1]).argmin()
        return idx

    def get_point_attributes(self, route, keys):
        # Regulatory Element per point
        for reg_elem in self.map.regulatoryElementLayer:
            for elem in reg_elem.parameters[
                "refers"
            ]:  # TODO: USe reglem.type() instead and avoid looping over all elements
                for key in keys:
                    if (
                        key == "stop_sign" and elem.attributes["subtype"] == "de206"
                    ):  # german stop sign id
                        stop_sign_pose = self.center_linestring(elem)
                        idx = self.find_closest_idx(route, stop_sign_pose)
                        route[idx, traffic_feat_idx["stop_sign"]] = 1
                        try:
                            route[idx, traffic_feat_idx["traffic_sign_regulatory"]] = (
                                regulatory_ts_encoding[elem.attributes["name"]]
                            )
                        except KeyError:
                            route[idx, traffic_feat_idx["traffic_sign_regulatory"]] = (
                                regulatory_ts_encoding["stop_sign"]
                            )
                        route[idx, traffic_feat_idx["stop_point"]] = 1
                        route[idx, traffic_feat_idx["stop_sign_x"]] = stop_sign_pose[0]
                        route[idx, traffic_feat_idx["stop_sign_y"]] = stop_sign_pose[1]
                    if (
                        key == "yield_right_of_way"
                        and elem.attributes["subtype"] == "de205"
                    ):
                        yield_sign_pose = self.center_linestring(elem)
                        idx = self.find_closest_idx(route, yield_sign_pose)
                        try:
                            route[idx, traffic_feat_idx["traffic_sign_regulatory"]] = (
                                regulatory_ts_encoding[elem.attributes["name"]]
                            )
                        except KeyError:
                            route[idx, traffic_feat_idx["traffic_sign_regulatory"]] = (
                                regulatory_ts_encoding["yield_right_of_way"]
                            )
                        route[idx, traffic_feat_idx["yield_sign"]] = 1
                        route[idx, traffic_feat_idx["yield_sign_x"]] = yield_sign_pose[
                            0
                        ]
                        route[idx, traffic_feat_idx["yield_sign_y"]] = yield_sign_pose[
                            1
                        ]
                    if (
                        key == "pedestrian_crossing"
                        and elem.attributes["subtype"] == "de101-11"
                    ):
                        pedestrian_crossing_pose = self.center_linestring(elem)
                        idx = self.find_closest_idx(route, pedestrian_crossing_pose)
                        route[idx, traffic_feat_idx["traffic_sign_warning"]] = (
                            warning_ts_encoding[elem.attributes["name"]]
                        )
                        p_c_x = np.mean([elem[0].x, elem[1].x, elem[2].x])
                        p_c_y = np.mean([elem[0].y, elem[1].y, elem[2].y])
                        route[idx, traffic_feat_idx["pedestrian_crossing_x"]] = p_c_x
                        route[idx, traffic_feat_idx["pedestrian_crossing_y"]] = p_c_y

                    if (
                        key == "traffic_light"
                        and elem.attributes["subtype"] == "traffic_light"
                    ):  # TODO: THe right subtype
                        traffic_light_pose = self.center_linestring(elem)
                        idx = self.find_closest_idx(route, traffic_light_pose)
                        route[idx, traffic_feat_idx["traffic_light"]] = 1
                        route[idx, traffic_feat_idx["stop_point"]] = 1
        return route

    def get_traffic_rules_in_route(self, route: torch.tensor, keys):
        # add rules that apply to every point
        rules = torch.stack(
            [self.get_traffic_rules_at_pos(point, keys) for point in route]
        )
        rules = self.get_point_attributes(rules, keys)
        return rules

    def get_lanelet_priority(self, lanelet):
        for reg_elem in lanelet.regulatoryElements:
            if type(reg_elem) == RightOfWay:
                if reg_elem.getManeuver(lanelet) == "RightOfWay":
                    return 1
                elif type(reg_elem) == "Yield":
                    return 2
        return 0

    def get_stop_signs(self, lanelet=None):
        """
        Find and return all stop signs in the given lanelet map.
        If lanelet is given, only return those that are associated with that lanelet.
        """
        stop_sign_list = []
        for reg_elem in self.map.regulatoryElementLayer:
            if type(reg_elem) == RightOfWay:
                for elem in reg_elem.parameters["refers"]:
                    if (
                        elem.attributes["type"] == "traffic_sign"
                        and elem.attributes["subtype"] == "de206"
                    ):  # german stop sign id
                        elem = self.center_stop_sign(elem)
                        if lanelet is not None:
                            pos2d = lanelet2.core.BasicPoint2d(elem[0], elem[1])
                            nearest_lanelet = self.map.laneletLayer.nearest(pos2d, 1)[0]
                            if nearest_lanelet != lanelet:
                                break
                        stop_sign_list.append(elem)
        return stop_sign_list

    @staticmethod
    def center_linestring(linestring):
        return np.array(
            [
                linestring[0].x + (linestring[1].x - linestring[0].x) / 2,
                linestring[0].y + (linestring[1].y - linestring[0].y) / 2,
            ]
        )

    def rereference_lanelet_on_pos(self, lanelet, pos):

        left_boundary = lanelet.leftBound
        lanelet.leftBound = lanelet2.core.LineString3d(
            points=[
                lanelet2.core.BasicPoint3d(point.x - pos.x, point.y - pos.y, 0.0)
                for point in left_boundary
            ]
        )

        right_boundary = lanelet.rightBound
        lanelet.rightBound = lanelet2.core.LineString3d(
            [
                lanelet2.core.BasicPoint2d(point.x - pos.x, point.y - pos.y)
                for point in right_boundary
            ]
        )

        centerline = lanelet.centerline
        lanelet.centerline = lanelet2.core.LineString3d(
            [
                lanelet2.core.BasicPoint2d(point.x - pos.x, point.y - pos.y)
                for point in centerline
            ]
        )

        return lanelet

    def extract_map_lanes(self, ego_trajectory: torch.tensor):
        map_lanes = torch.zeros([1, 2, 2, len(traffic_feat_idx)], dtype=torch.float)

        prev_lanelet = None
        point_idx = 0
        for ego_pos in ego_trajectory:
            lanelet = self.get_lanelet_at_map_pos(ego_pos[:2])
            # Primary Lane
            if lanelet != prev_lanelet:
                prev_lanelet = lanelet
                point_idx2 = int(point_idx)
                for point in lanelet.centerline:
                    map_lanes[0, 0, point_idx2, traffic_feat_idx["cl_x"]] = point.x
                    map_lanes[0, 0, point_idx2, traffic_feat_idx["cl_y"]] = point.y
                    map_lanes[0, 0, point_idx2, traffic_feat_idx["cl_yaw"]] = float(
                        point.attributes["yaw"]
                    )
                    point_idx2 += 1

                point_idx2 = int(point_idx)
                for point in lanelet.leftBound:
                    map_lanes[0, 0, point_idx2, traffic_feat_idx["ll_x"]] = point.x
                    map_lanes[0, 0, point_idx2, traffic_feat_idx["ll_y"]] = point.y
                    map_lanes[0, 0, point_idx2, traffic_feat_idx["ll_yaw"]] = float(
                        point.attributes["yaw"]
                    )
                    point_idx2 += 1

                point_idx2 = int(point_idx)
                for point in lanelet.rightBound:
                    map_lanes[0, 0, point_idx2, traffic_feat_idx["rl_x"]] = point.x
                    map_lanes[0, 0, point_idx2, traffic_feat_idx["rl_y"]] = point.y
                    map_lanes[0, 0, point_idx2, traffic_feat_idx["rl_yaw"]] = float(
                        point.attributes["yaw"]
                    )
                    point_idx2 += 1

                point_idx = int(point_idx2)

                # # Secondary Lane
                # for point in lanelet.leftBound:
                #     map_lanes[0, 1, 0, traffic_feat_idx["rl_x"]] = point.x
                #     map_lanes[0, 1, 0, traffic_feat_idx["rl_y"]] = point.y
                #     map_lanes[0, 1, 0, traffic_feat_idx["rl_yaw"]] = float(
                #         point.attributes["yaw"]
                #     )

                # for point in lanelet.rightBound:
                #     map_lanes[0, 1, 0, traffic_feat_idx["ll_x"]] = point.x
                #     map_lanes[0, 1, 0, traffic_feat_idx["ll_y"]] = point.y
                #     map_lanes[0, 1, 0, traffic_feat_idx["ll_yaw"]] = float(
                #         point.attributes["yaw"]
                #     )

                # for point in lanelet.centerline:
                #     map_lanes[0, 1, 0, traffic_feat_idx["cl_x"]] = point.x
                #     map_lanes[0, 1, 0, traffic_feat_idx["cl_y"]] = point.y
                #     map_lanes[0, 1, 0, traffic_feat_idx["cl_yaw"]] = float(
                #         point.attributes["yaw"]
                #     )

        return map_lanes

    def extract_route_rules(self, map_lanes, ego_pos, keys):
        maplane_features = torch.zeros(
            [
                1,
                map_lanes.shape[-3],
                map_lanes.shape[-2],
                max(list(traffic_feat_idx.values())) + 1,
            ]
        ).to(device=map_lanes.device)
        maplane_features[:, :, :, : map_lanes.shape[-1]] = map_lanes

        if self.has_map:
            for idx, lanes in enumerate(maplane_features[0, :]):
                maplane_features[0, idx, :, :] = self.get_traffic_rules_in_route(
                    lanes, keys
                )

        # return traffic_rules
        return maplane_features


if __name__ == "__main__":
    # Usage
    FILENAME = (
        "tegel_map/Decision_tegel_map_18_04_2024_with_centerlines_2_processed.osm"
    )
    ORIGIN = (52.55754329939843, 13.281288759978164)

    base_path = (
        Path.cwd() / "train" / "LimSim_DIPP" / "networkFiles"
    )  # cwd should be mtdm_poc
    file_path = str(base_path / FILENAME)

    lane_extractor = LaneletDataExtractor(file_path, ORIGIN, proj_type="utm")
