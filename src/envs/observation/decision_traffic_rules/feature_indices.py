agent_feat_id = {
    "id" : 0, # tracking id [int]
    "x": 1,  # location [m]
    "y": 2,  # location [m]
    "yaw": 3,  # heading [rad]
    "vx": 4,  # velocity [m/s]
    "vy": 5,  # velocity [m/s]
    "length": 6,  # bounding box size [m]
    "width": 7,  # bounding box size [m]
    "height": 8,  # bounding box size [m]
    "class": 9,  # object class [categorical]
    "is_dynamic": 10,
    "road_id": 11,  # road identifier [int]
    "lane_id": 12,  # lane identifier (within road) [int] sign indicates direction of travel
    "is_junction": 13,  # if agent is in junction --- currently being  scoped ---
    "s" : 14, # distance towards end of lane in frenet frame --- currently being  scoped ---
    "rss_obj_id": 15, # int --- currently being  scoped ---
    "rss_status": 16, # TODO: unit --- currently being  scoped ---
    "rss_long_current_dist": 17, # TODO: unit --- currently being  scoped ---
    "rss_long_safe_dist": 18, # TODO: unit --- currently being  scoped ---
    "rss_lat_current_right_dist": 19, # TODO: unit --- currently being  scoped ---
    "rss_lat_safe_right_dist": 20, # TODO: unit --- currently being  scoped ---
    "rss_lat_current_left_dist": 21, # TODO: unit --- currently being  scoped ---
    "rss_lat_safe_left_dist": 22, # TODO: unit --- currently being  scoped ---
    "relative_priority": 23, # TODO: unit --- currently being  scoped ---
}