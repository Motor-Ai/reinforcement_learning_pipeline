# Unset=0, Vehicle=1, Pedestrian=2, Cyclist=3, Other=4
agent_classes = {
    "default":0,
    "vehicle":1,
    "pedestrian":2,
    "cyclist":3,
    "other":4,
}

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
    "turning_intention" :15, # what direction is the agent manuever changing [categorical]
}

rss_feat_id = {
    "rss_obj_id": 0, # int --- currently being  scoped ---
    "rss_status": 1, # TODO: unit --- currently being  scoped ---
    "rss_long_current_dist": 2, # TODO: unit --- currently being  scoped ---
    "rss_long_safe_dist": 3, # TODO: unit --- currently being  scoped ---
    "rss_lat_current_right_dist": 4, # TODO: unit --- currently being  scoped ---
    "rss_lat_safe_right_dist": 5, # TODO: unit --- currently being  scoped ---
    "rss_lat_current_left_dist": 6, # TODO: unit --- currently being  scoped ---
    "rss_lat_safe_left_dist": 7, # TODO: unit --- currently being  scoped ---
}

turning_intention = {
    "left":-1,
    "none":0,
    "right":1
}