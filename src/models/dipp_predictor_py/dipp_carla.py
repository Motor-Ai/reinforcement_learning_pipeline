import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.models.dipp_predictor_py.dipp_predictor_utils import AgentEncoder, LaneEncoder, Agent2Map, Agent2Agent, AgentDecoder, Score


# Build predictor
class Predictor(nn.Module):
    def __init__(self, future_steps):
        super(Predictor, self).__init__()
        self._future_steps = future_steps
        self.modes = 3

        # agent layer
        self.vehicle_net = AgentEncoder()
        self.vehicle_net_future = AgentEncoder()
        # self.pedestrian_net = AgentEncoder()
        # self.cyclist_net = AgentEncoder()

        # map layer
        self.lane_net = LaneEncoder()
        # self.crosswalk_net = CrosswalkEncoder()

        # attention layers
        self.agent_map = Agent2Map()
        self.agent_agent = Agent2Agent()

        # decode layers
        # self.plan = AVDecoder(self._future_steps)
        self.predict_ego = AgentDecoder(self._future_steps, n_agents=1)
        self.predict = AgentDecoder(self._future_steps)
        self.score = Score()

    def trainable_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Trainable parameters", params)
        return params

    def forward(
        self,
        ego,
        neighbors,
        map_lanes,
        preprocess=True,
    ):
        if preprocess:
            # remove map features not available in the stack
            EPS = 0.0000001  # not using this gives Nan outputs

            """
            Lanes from limsim are currently all zero except x (index 0) y (index 1) and traffic light status (index 13)
            traffic light status from limsim is 0) other 1) green 2) red
            """
            # remove agent features not available in stack
            ego[..., 3:] = (
                ego[..., 3:] * 0 + EPS
            )  # + 0.000001 # this can be set to zero if training from scratch
            neighbors[..., 3:-1] = (
                neighbors[..., 3:-1] * 0 + EPS
            )  # + 0.000001 # this can set to zero if training from scratch

        # actors
        ego_actor = self.vehicle_net(ego)
        vehicles = torch.stack(
            [self.vehicle_net(neighbors[:, i]) for i in range(10)], dim=1
        )

        actors = torch.cat([ego_actor.unsqueeze(1), vehicles], dim=1)

        # maps
        lane_feature = self.lane_net(map_lanes)
        # crosswalk_feature = self.crosswalk_net(map_crosswalks)
        lane_mask = torch.eq(map_lanes, 0)[:, :, :, 0, 0]
        # crosswalk_mask = torch.eq(map_crosswalks, 0)[:, :, :, 0, 0]
        map_mask = lane_mask  # torch.cat([lane_mask, crosswalk_mask], dim=2)
        map_mask[:, :, 0] = False  # prevent nan

        # actor to actor
        agent_agent = self.agent_agent(actors)  # , actor_mask)

        # map to actor
        map_feature, agent_map = [], []

        for i in range(actors.shape[1]):
            output = self.agent_map(
                agent_agent[:, i], lane_feature[:, 0], map_mask[:, 0]
            )  #  crosswalk_feature[:, i],
            map_feature.append(output[0])
            agent_map.append(output[1])
        map_feature = torch.stack(map_feature, dim=1)  # torch.Size([32, 11, 20, 256])
        agent_map = torch.stack(agent_map, dim=2)  # torch.Size([32, 6, 11, 256])

        predictions_ego = self.predict_ego(agent_map[:, :, 0:1], agent_agent[:, 0:1], ego[:, None, -1])
        predictions = self.predict(agent_map[:, :, 1:], agent_agent[:, 1:], neighbors[:, :, -1])
        predictions = torch.cat([predictions_ego, predictions], dim=2)

        # scores = self.score(map_feature, agent_agent, agent_map)

        return predictions

    def MFMA_maneuver_loss(
    self,
    predictions,
    ground_truth,
    weights
    ):
        ego_loss_weight = 0.8
        predictions = predictions * weights.unsqueeze(1)
        predictions = predictions[:, 0]
        
        prediction_loss = 0.0
        prediction_loss += F.smooth_l1_loss(predictions[:, 0], ground_truth[:, 0, :, :3]) * ego_loss_weight
        prediction_loss += F.smooth_l1_loss(predictions[:, 1:], ground_truth[:, 1:, :, :3]) * (1-ego_loss_weight)
        
        return prediction_loss.mean()


    def MFMA_maneuver_loss_overfit(
    self,
    predictions,
    scores,
    scores_neighbors,
    ground_truth,
    weights,
    true_maneuver_index,
    reduction="mean",
    ):
        predictions = predictions * weights.unsqueeze(1)
       
        # TODO  we are masking out zeros in the loss, isn't that a problem when ego future is actually 0,0 (stopping)?

        # find non zero neighbors and ego, which is always non zero
        num_agents = (ground_truth[:, 1:, :, :2].sum((-2, -1)).abs() > 0).sum(-1) + 1

        # prediction loss for ego and neighbors decoded at the mode of the GT maneuver
        prediction = torch.stack(
            [predictions[i, m] for i, m in enumerate(true_maneuver_index)]
        )
        prediction_loss = 0
        prediction_loss_fde = 0
        total_prediction_loss = []
        for i in true_maneuver_index:
            num_agents = (ground_truth[i, 1:, :, :2].sum((-2, -1)).abs() > 0).sum(-1) + 1
            # prediction_loss = F.smooth_l1_loss(prediction[i, :, -1, :3], ground_truth[i, :, -1, :3], reduction="none")
            if i != 5:
                 prediction_loss_ade = F.smooth_l1_loss(prediction[i, :, :, :3], ground_truth[i, :, :, :3], reduction="none").mean(1)
                #  prediction_loss_fde = F.smooth_l1_loss(prediction[i, :, -1, :3], ground_truth[i, :, -1, :3], reduction="none")
                 total_loss = prediction_loss_ade#+prediction_loss_fde
                 total_prediction_loss.append(total_loss)
            else:
                print('=====================================================Falsch=====================================================')
                total_prediction_loss.append(prediction_loss)
        return (
            torch.stack(total_prediction_loss).mean()
        )  # + prediction_loss_ego.mean()# + score_loss_neighbors # + score_loss
