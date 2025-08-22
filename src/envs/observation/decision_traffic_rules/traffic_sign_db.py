# DIPP Lane Features
# centerline_x
# centerline_y
# centerline_yaw
# leftline_x
# leftline_y
# leftline_yaw
# rightline_x
# rightline_y
# rightline_yaw
# speed_limit
# centerline_type
# leftline_type
# rightline_type
# traffic_light_type
# stop_point
# interpolating
# stop_sign

# Source: https://routetogermany.com/drivingingermany/road-signs#supplementary-signs

warning_ts_encoding = {"traffic_sign_ahead": 1, "pedestrian_crossing": 2}

regulatory_ts_encoding = {
    "stop_sign": 1,
    "yield_right_of_way": 2,
    "no_entry": 3,
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
}

installations_ts_encoding = {
    "right_obstruction_marker": 1,
    "left_obstruction_marker": 2,
}

lane_type = {
    "dashed": 1,
    "yellow_dashed": 2,
    "yellow_solid": 3,
    "double_dashed": 4,
    "other": 5,
    "solid": 6,
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
    "is_route": 45, # 0: not on route, 1: on route
}
