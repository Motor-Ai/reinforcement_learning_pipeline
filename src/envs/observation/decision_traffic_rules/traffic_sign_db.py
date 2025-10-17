# Source: https://routetogermany.com/drivingingermany/road-signs#supplementary-signs

warning_ts_encoding = {"traffic_sign_ahead": 1, 
                       "pedestrian_crossing": 2}

supplementary_ts_encoding = {"no_entry": 1}

regulatory_ts_encoding = {
    "stop_sign": 1,
    "yield_right_of_way": 2,
    "no_entry": 3,
    "round_about": 4,
    "no_overtaking": 5,
    "bus_stop" : 6,
    "speed_limit_20": 20,
    "speed_limit_30": 30,
    "speed_limit_40": 40,
    "speed_limit_50": 50,
    "speed_limit_60": 60,
    "speed_limit_70": 70,
    "speed_limit_80": 80,
    "speed_limit_90": 90,
    "speed_limit_100": 100,
}

directional_ts_encoding = {
    "priority_at_intersection": 1,
    "right_ahead": 2,
    "left_ahead": 3,
    "priority_road":4,
    "autobahn_start":5,
    "autobahn_end":6,
}

installations_ts_encoding = {
    "right_obstruction_marker": 1,
    "left_obstruction_marker": 2,
}

lane_marking_type = {
    "dashed": 1,
    "yellow_dashed": 2,
    "yellow_solid": 3,
    "double_dashed": 4,
    "other": 5,
    "solid": 6,
    "line_thin": 7,
}

left_passable_types = {
    "dashed": 16,           # LABEL_LINE_MARKING_DASHED,
    "solid_dashed": 18,     # LABEL_LINE_MARKING_SOLID_DASHED,
}

right_passable_types = {
    "dashed": 16,           # LABEL_LINE_MARKING_DASHED,
    "dashed_solid": 17,     # LABEL_LINE_MARKING_DASHED_SOLID,
}

traffic_sign_type_labels = {
    -205: "yield_right_of_way",     # LABEL_TRAFFIC_SIGN_205
    -206: "stop_sign",              # LABEL_TRAFFIC_SIGN_206
    -27430: "speed_limit_30",       # LABEL_TRAFFIC_SIGN_27430
    -27450: "speed_limit_50",       # LABEL_TRAFFIC_SIGN_27450
    -27470: "speed_limit_70",       # LABEL_TRAFFIC_SIGN_27470
}

traffic_light_type = {
    "pedestrian": 1,
    "directional": 2,
    "construction": 3,
}

traffic_light_state = {
    "red": 1,
    "yellow": 2,
    "green": 3,
    "inactive": 4,
}

lane_surface_type = {
    "Concrete":1, 
    "asphalt":2,
    "Gravel":3,
    "Pavement":4,
}

lane_type = {"road":1,
          "crosswalk":2,
          "motorway":3,
          "bicycle_lane":4,
          "pedestrian_lane":5,
          "bus_lane":6,
          }

lane_morphology_type = {
    "straight": 1,
    "curve": 2,
    "sloped": 3,
    "None": 4,
    "roundabout": 5,
    "merge":6,
    "split":7,
    "turn":8,
    "bus_lane":9,
    "intersection":10,
}

# influenced by: https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/RegulatoryElementTagging.md
traffic_feat_idx = {
    "cl_x": 0,
    "cl_y": 1,
    "cl_yaw": 2,
    "ll_x": 3,
    "ll_y": 4,
    "ll_yaw": 5,
    "rl_x": 6,
    "rl_y": 7,
    "rl_yaw": 8,
    "speed_limit": 9,  # or maxspeed
    "maxspeed": 9,
    "cl_type": 10,  # under attribute lanelet.centerline.attributes["lane_marking"]
    "ll_type": 11,  # similar as above
    "rl_type": 12,  # similar as above
    "tl_type": 13,
    "stop_point": 14,  # Index of the stop line info
    "interpolating": 15,
    "stop_sign": 16,
    "lane_id": 17,  # Index of the lane id
    "successor_laneID_1": 18,  # Index of the successor lane
    "successor_laneID_2": 19,  # Index of the successor lane
    "successor_laneID_3": 20,  # Index of the successor lane
    "traffic_sign_warning": 21,  # Index of the warning traffic signs info
    "traffic_sign_regulatory": 22,  # Index of the regulatory traffic signs info
    "traffic_sign_directional": 23,  # Index of the directional traffic signs info
    "traffic_sign_installations": 24,  # Index of the installations traffic signs info
    "traffic_sign_supplementary": 25,  # Index of the supplementary traffic signs info
    "priority": 26,  # or right_of_way
    "right_of_way": 26,  # 0: no assigned priority, 1: right of way, 2: yield
    "all_way_stop": 27,
    "dynamic": 28,
    "fallback": 29,
    "traffic_light": 30,
    "stop_sign_x": 31,
    "stop_sign_y": 32,
    "yield_to_1": 33,
    "yield_to_2": 34,
    "yield_to_3": 35,
    "yield_to_4": 36,
    "yield_sign": 37,
    "yield_sign_x": 38,
    "yield_sign_y": 39,
    "pedestrian_crossing_x": 40,  # assuming pedestrian crossings are defined by 3 points
    "pedestrian_crossing_y": 41,
    "s": 42,
    "road_id": 43, 
    "lane_id": 44,
    "lane_morphology": 45, # the type of lane morphology [lane_morphology_type]
    "lane_type": 46, # the type of lane [lane_type]
    "lane_surface": 47, # the type of lane surface [lane_surface_type]
    "tl_state": 48,  # Index of the traffic light state
    "is_route": 49, # whether the lane is part of the planned route
}