import stlcg
import numpy as np
import stlcg
from stlcg import Expression
import torch
import lanelet2

STOP_LINE_FEAT_IDX = 10  # Index of the stop line info in the centerline route
LANE_ID_FEAT_IDX = 9  # Index of the lane id in the centerline route
TRAFFIC_LIGHT_FEAT_IDX = 8  # Index of the traffic light info in the centerline route
TRAFFIC_SIGN_FEAT_IDX = 7  # Index of the traffic sign info in the centerline route

SIGN_TYPE = {
    "STOP": torch.tensor([1]).requires_grad_(False).to(device="cuda"),
    "YIELD": torch.tensor([2]).requires_grad_(False).to(device="cuda"),
}

TRAFFIC_LIGHT_STATUS = {
    "RED": torch.tensor([1]).requires_grad_(False).to(device="cuda"),
    "GREEN": torch.tensor([2]).requires_grad_(False).to(device="cuda"),
    "YELLOW": torch.tensor([3]).requires_grad_(False).to(device="cuda"),
    "INACTIVE": torch.tensor([4]).requires_grad_(False).to(device="cuda"),
}

EGO_PRED_HORIZON = 50


def min_dist(ego_pred, route_clipped):
    route_to_goal_dist = torch.norm(
        route_clipped[..., :2] - ego_pred[..., :1, :2], dim=-1
    )

    # Clip the route to ego pred length # TODO: Make the time space relative
    route_to_goal_dist = route_to_goal_dist[:, : ego_pred.shape[-2]]

    # route_to_goal_dist = torch.norm(route_clipped[..., :2] - ego_pred, dim=-1)

    # Considering ego cnetric route
    # route_to_goal_dist = torch.norm(route_clipped[:, :2], dim=-1)
    # dist_to_goal, idx = route_to_goal_dist.min(dim=-1)
    return route_to_goal_dist


def calc_velocity(ego_pred_x, ego_pred_y):
    vel = torch.zeros_like(
        ego_pred_x
    )  # First element is set to 0, as the size reduces by 1 with differntiation
    v_x = torch.diff(ego_pred_x, dim=-2)
    v_y = torch.diff(ego_pred_y, dim=-2)
    vel[:, 1:, :] = torch.hypot(v_x, v_y)
    return vel


def relevant_traffic_light(route):
    # TODO: Traffic Light State only from the predetermined route given by the local router
    mask = route[..., TRAFFIC_LIGHT_FEAT_IDX] != 0
    route_shape = route.shape
    route_till_tl = torch.masked_select(
        route[..., TRAFFIC_LIGHT_FEAT_IDX], mask
    ).reshape(
        route_shape[0], -1
    )  # All the traffic lights in the route

    # Get only the first lights in the route
    route_till_tl[..., :-1] = 0
    route_till_tl = route_till_tl[
        ..., :EGO_PRED_HORIZON
    ]  # TODO: Should we consider all traffic lights in the route?

    route_till_tl = route_till_tl.view(route_shape[0], -1).unsqueeze(-1)
    return route_till_tl


class TrafficRules:
    def __init__(self, route):
        self.route = route
        self.ego_pose = self.route[0, :]
        self.ego_pred_horizon = EGO_PRED_HORIZON
        # Parameters
        self.V_ERR = torch.Tensor([00.1]).requires_grad_(False).to(device="cuda")  # m/s
        self.D_SL = torch.Tensor([5.0]).requires_grad_(False).to(device="cuda")  # m
        self.D_BR = "D_BR", torch.Tensor([5.0]).requires_grad_(False).to(
            device="cuda"
        )  # m
        self.T_SLW = 3  # indexes Time is pause at an intersection stop sign

        self.TRUE_expr = Expression(
            "TRUE", torch.Tensor([True]).requires_grad_(False).to(device="cuda")
        )
        self.FALSE_expr = Expression(
            "FALSE", torch.Tensor([False]).requires_grad_(False).to(device="cuda")
        )

    def line_in_front(self, route):
        if self.route.shape[-2] == route.shape[-2]:
            return (
                torch.Tensor([0])
                .unsqueeze(-1)
                .repeat(self.route.shape[0], self.ego_pred_horizon)
                .requires_grad_(False)
                .to(device="cuda")
            )
        else:
            return (
                torch.Tensor([1])
                .unsqueeze(-1)
                .repeat(self.route.shape[0], self.ego_pred_horizon)
                .requires_grad_(False)
                .to(device="cuda")
            )

    def stop_line(self):
        id = torch.where(self.route[..., STOP_LINE_FEAT_IDX] == 1)[0]
        if id.shape[0] != 0:  # If STOP LINE found
            route_until_stop_line = self.route[..., :id]
            return route_until_stop_line
        else:  # else return the whole route
            return self.route

    def at_traffic_sign(self, sign_type):
        identified_sign = torch.where(
            self.route[..., TRAFFIC_SIGN_FEAT_IDX] == SIGN_TYPE[sign_type]
        )
        rhs = SIGN_TYPE[sign_type]
        lhs = Expression(
            "route_traffic_sign_info",
            self.route[..., TRAFFIC_SIGN_FEAT_IDX],
        )
        return stlcg.Equal(lhs=lhs, val=rhs)

    def in_standstill(self, vel):
        return stlcg.LessThan(lhs=vel, val=self.V_ERR)

    def passing_stop_line(self):
        t = 0
        # ego_pred_x = Expression("ego_pred_x", torch.randn((2)).to(device="cuda"))
        # ego_pred_y = Expression("ego_pred_y", torch.randn((2)).to(device="cuda"))
        # ego_pred_yaw = Expression("# "Red", "Yellow", "Green", "Inactive"ego_pred_yaw", torch.randn((2)).to(device="cuda"))
        stop_line_in_front_expr = self.stop_line_in_front()
        neg_stop_line_in_front = stlcg.Negation(subformula=stop_line_in_front_expr)
        not_crossed = stlcg.Eventually(
            subformula=neg_stop_line_in_front  # , interval=[t, t + 2.0]
        )
        return stlcg.And(subformula1=stop_line_in_front_expr, subformula2=not_crossed)

    def traffic_light_state(self, traffic_light_info):
        # TODO: query traffic light state from SU?
        # "Red", "Yellow", "Green", "Inactive"
        return traffic_light_info

    def active_tls(self, traffic_light_info):
        status = self.traffic_light_state(traffic_light_info)
        return stlcg.Negation(
            status
            == Expression(
                "active_tls",
                torch.zeros_like(traffic_light_info.value).requires_grad_(False),
            )
        )

    def stop_line_in_front(self):
        # Look for the stop line info in the route
        min_dist_expr = stlcg.Expression(
            "min_dist",
            torch.randn((15)).requires_grad_(False).to(device="cuda"),
        )
        lhs = stlcg.LessThan(lhs=min_dist_expr, val=self.D_SL)
        line_in_front_expr = stlcg.Expression(
            "line_in_front",
            torch.randn((15)).requires_grad_(False).to(device="cuda"),
        )
        rhs = stlcg.Equal(
            lhs=line_in_front_expr,
            val=torch.tensor([1]).requires_grad_(False).to(device="cuda"),
        )
        return stlcg.And(subformula1=lhs, subformula2=rhs)

    def stop_sign_rule(self):
        ego_pred_x = Expression(
            "ego_pred_x", torch.randn([15, 50, 1]).to(device="cuda")
        )
        ego_pred_y = Expression(
            "ego_pred_y", torch.randn([15, 50, 1]).to(device="cuda")
        )
        # ego_pred_yaw = Expression("ego_pred_yaw", torch.randn([15, 50, 1]).to(device="cuda"))

        lhs1 = stlcg.And(
            subformula1=self.passing_stop_line(),
            subformula2=self.at_traffic_sign("STOP"),
        )

        traffic_sign_info = Expression(
            "traffic_sign_info", torch.randn([15, 1, 1]).to(device="cuda")
        )

        lhs = stlcg.Always(
            subformula=stlcg.And(
                subformula1=lhs1,
                subformula2=stlcg.Negation(self.active_tls(traffic_sign_info)),
            )
        )

        vel = Expression("vel", torch.randn([15, 50, 1]).to(device="cuda"))
        rhs1 = stlcg.And(
            subformula1=self.stop_line_in_front(),
            subformula2=self.in_standstill(vel),
        )
        rhs2 = stlcg.Always(subformula=rhs1, interval=[0, self.T_SLW])
        rhs = stlcg.Eventually(
            subformula=rhs2  # , interval=[-2.0, 0.0]
        )  # TODO : once in the previous state
        return stlcg.Implies(subformula1=lhs, subformula2=rhs1)
        # return lhs

    def _stl_signal(self, ego_pred_x, ego_pred_y):
        ego_pred = torch.cat([ego_pred_x, ego_pred_y], dim=-1)
        min_dist_val = min_dist(ego_pred, self.stop_line()).unsqueeze(-1)
        line_in_front_val = self.line_in_front(self.stop_line()).unsqueeze(-1)
        vel = calc_velocity(ego_pred_x, ego_pred_y)
        relevant_traffic_light_ip = relevant_traffic_light(self.route)

        sign_type = "STOP"
        sign = SIGN_TYPE[sign_type].repeat(ego_pred_x.shape[0], ego_pred_x.shape[1], 1)

        stop_line_in_front_ip = (min_dist_val, line_in_front_val)
        passing_stop_line_ip = (stop_line_in_front_ip, stop_line_in_front_ip)
        lhs1_ip = (passing_stop_line_ip, sign)
        lhs_ip = (lhs1_ip, relevant_traffic_light_ip)
        rhs1 = (stop_line_in_front_ip, vel)

        return (lhs_ip, rhs1)

    def _shape_signal_batch(self, ego_pred):
        ego_pred = ego_pred.view(
            [
                ego_pred.shape[0] * ego_pred.shape[1],
                ego_pred.shape[2],
                ego_pred.shape[3],
            ]
        )
        ego_pred_x = ego_pred[:, :, 0].unsqueeze(-1)
        ego_pred_y = ego_pred[:, :, 1].unsqueeze(-1)
        ego_pred_yaw = ego_pred[:, :, 2].unsqueeze(-1)
        return self._stl_signal(ego_pred_x, ego_pred_y)

    def set_route(self, route):
        self.route = route

    def get_robustness(self, route, ego_pred, scale=-1):
        # Modify shape of route according to ego_pred
        route = (
            (
                route.unsqueeze(0)
                .unsqueeze(0)
                .repeat(ego_pred.shape[0], ego_pred.shape[1], 1, 1)
            )
            .to(device="cuda")
            .requires_grad_(False)
        ).view(ego_pred.shape[0] * ego_pred.shape[1], route.shape[0], -1)
        # TODO: set ego lane ID
        self.ego_lane_id = torch.tensor([0]).to(device="cuda")

        # Set relevant route
        self.set_route(route)

        stop_sign_rule = self.stop_sign_rule()

        robustness = (
            stop_sign_rule.robustness(self._shape_signal_batch(ego_pred), scale=scale)
            .squeeze()
            .reshape(ego_pred.shape[0], ego_pred.shape[1])
        )
        return robustness


if __name__ == "__main__":
    ego_pred = torch.zeros([5, 3, 50, 3]).requires_grad_(True).to(device="cuda")
    ego_pred[:, :, :, 0] = torch.arange(50)
    ego_pred[:, :, :, 1] = torch.arange(50) * 2
    ego_pred[:, :, :, 2] = torch.arange(50)

    route = torch.zeros([93, 17]).requires_grad_(False).to(device="cuda")
    route[:, 0] = torch.arange(93)
    route[:, 1] = torch.arange(93)
    route[:, TRAFFIC_LIGHT_FEAT_IDX] = 1

    traffic_rule = TrafficRules(route)
    traffic_rule.get_robustness(route, ego_pred)
