import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
# import imageio
import glob
import sys
import os
# from model.cost_functions import calculate_cost

# from visualization.viz import plot_dipp
import torch
import logging
import glob
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
import yaml

# from visualization.viz import (
#     # plot_speed_from_centerline,
#     # plot_priority_from_centerline,
#     plot_traffic_law_text,
# )


# PATH = os.path.dirname(os.path.realpath(__file__)).split("model")[0] + "config"
# with open(PATH + "/predictor.yaml", "r", encoding="utf-8") as f:
    # config = yaml.load(f, Loader=yaml.FullLoader)
EMB_DIM = 256
N_HEADS = 8
FF_DIM = 1024
MODES = 6
AGENT_FEATURES = 5




# Local context encoders
class LaneEncoder(nn.Module):
    def __init__(self):
        super(LaneEncoder, self).__init__()

        # encdoer layer
        self.self_line = nn.Linear(
            3, 128
        )  # I think way less than 128 or no embedding at all suffices
        # self.left_line = nn.Linear(3, 128)
        # self.right_line = nn.Linear(3, 128)
        # self.speed_limit = nn.Linear(1, 64)

        # self.self_type = nn.Embedding(4, 64, padding_idx=0)
        # self.left_type = nn.Embedding(11, 64, padding_idx=0)
        # self.right_type = nn.Embedding(11, 64, padding_idx=0)
        self.traffic_light_type = nn.Embedding(
            9, 64, padding_idx=0
        )  # I think way less than 64 or no embedding at all suffices
        # self.interpolating = nn.Embedding(2, 64)
        # self.stop_sign = nn.Embedding(2, 64)
        # self.stop_point = nn.Embedding(2, 64)

        # hidden layers
        self.pointnet = nn.Sequential(
            nn.Linear(128 + 64, 384), nn.ReLU(), nn.Linear(384, EMB_DIM), nn.ReLU()
        )

    def forward(self, inputs):
        # embedding
        self_line = self.self_line(inputs[..., :3])
        # left_line = self.left_line(inputs[..., 3:6])
        # right_line = self.right_line(inputs[...,  6:9])
        # speed_limit = self.speed_limit(inputs[..., 9].unsqueeze(-1))

        # self_type = self.self_type(inputs[..., 10].int())
        # left_type = self.left_type(inputs[..., 11].int())
        # right_type = self.right_type(inputs[..., 12].int())
        traffic_light = self.traffic_light_type(inputs[..., 13].int())
        # stop_point = self.stop_point(inputs[..., 14].int())
        # interpolating = self.interpolating(inputs[..., 15].int())
        # stop_sign = self.stop_sign(inputs[..., 16].int())

        # lane_attr = self_type + left_type + right_type + traffic_light + stop_point + interpolating + stop_sign
        # lane_embedding = torch.cat([self_line, left_line, right_line, speed_limit, lane_attr], dim=-1)
        lane_embedding = torch.cat([self_line, traffic_light], dim=-1)

        # maxpool along waypoints
        lane_embedding = lane_embedding.max(dim=-2, keepdim=True)[
            0
        ]  # TODO maxpool after pointnet?

        # process
        output = self.pointnet(lane_embedding)

        return output


class CrosswalkEncoder(nn.Module):
    def __init__(self):
        super(CrosswalkEncoder, self).__init__()
        self.point_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, EMB_DIM),
            nn.ReLU(),
        )

    def forward(self, inputs):
        output = self.point_net(inputs)

        return output


# Transformer modules
class CrossTransformer(nn.Module):
    def __init__(self):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(
            EMB_DIM, N_HEADS, 0.1, batch_first=True
        )
        self.transformer = nn.Sequential(
            nn.LayerNorm(EMB_DIM),
            nn.Linear(EMB_DIM, FF_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(FF_DIM, EMB_DIM),
            nn.LayerNorm(EMB_DIM),
        )

    def forward(
        self,
        query,
        key,
        value,
        mask=None,
    ):
        attention_output, attention_weights = self.cross_attention(
            query, key, value, key_padding_mask=mask
        )
        output = self.transformer(attention_output)

        return output


class MultiModalTransformer(nn.Module):
    def __init__(self, modes=3, output_dim=256):
        super(MultiModalTransformer, self).__init__()
        self.modes = modes
        self.attention = nn.ModuleList([nn.MultiheadAttention(256, 4, 0.1, batch_first=True) for _ in range(modes)])
        self.ffn = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, output_dim), nn.LayerNorm(output_dim))

    def forward(self, query, key, value, mask=None):
        attention_output = []
        for i in range(self.modes):
            attention_output.append(self.attention[i](query, key, value, key_padding_mask=mask)[0])
        attention_output = torch.stack(attention_output, dim=1)
        output = self.ffn(attention_output)

        return output


# Transformer-based encoders
class Agent2Agent(nn.Module):
    def __init__(self):
        super(Agent2Agent, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMB_DIM,
            nhead=N_HEADS,
            dim_feedforward=FF_DIM,
            activation="relu",
            batch_first=True,
        )
        self.interaction_net = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, inputs, mask=None):
        output = self.interaction_net(inputs)  # , src_key_padding_mask=mask)

        return output


class Agent2Map(nn.Module):
    def __init__(self):
        super(Agent2Map, self).__init__()
        self.lane_attention = CrossTransformer()
        # self.crosswalk_attention = CrossTransformer()
        self.map_attention = MultiModalTransformer()

    def forward(self, actor, lanes, mask):  # crosswalks
        query = actor.unsqueeze(1)
        lanes_actor = [
            self.lane_attention(query, lanes[:, i], lanes[:, i])
            for i in range(lanes.shape[1])
        ]
        # crosswalks_actor = [self.crosswalk_attention(query, crosswalks[:, i], crosswalks[:, i]) for i in range(crosswalks.shape[1])]
        map_actor = torch.cat(lanes_actor, dim=1)  # +crosswalks_actor
        output = self.map_attention(query, map_actor, map_actor, mask).squeeze(2)

        return map_actor, output


# Decoders
class AgentDecoder(nn.Module):
    def __init__(self, future_steps, n_agents=10):
        super(AgentDecoder, self).__init__()
        self._future_steps = future_steps 
        self.num_agents = n_agents
        self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, future_steps*3))

    def transform(self, prediction, current_state):
        x = current_state[:, 0] 
        y = current_state[:, 1]
        theta = current_state[:, 2]
        delta_x = prediction[:, :, 0]
        delta_y = prediction[:, :, 1]
        delta_theta = prediction[:, :, 2]
        new_x = x.unsqueeze(1) + delta_x 
        new_y = y.unsqueeze(1) + delta_y 
        new_theta = theta.unsqueeze(1) + delta_theta
        traj = torch.stack([new_x, new_y, new_theta], dim=-1)

        return traj
       
    def forward(self, agent_map, agent_agent, current_state):
        feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, 3, 1, 1)], dim=-1)
        decoded = self.decode(feature).view(-1, 3, self.num_agents, self._future_steps, 3)
        trajs = torch.stack([self.transform(decoded[:, i, j], current_state[:, j]) for i in range(3) for j in range(self.num_agents)], dim=1)
        trajs = torch.reshape(trajs, (-1, 3, self.num_agents, self._future_steps, 3))

        return trajs

class AVDecoder(nn.Module):
    def __init__(self, future_steps=50, feature_len=9):
        super(AVDecoder, self).__init__()
        self._future_steps = future_steps
        self.control = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, future_steps*2))
        self.cost = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, feature_len), nn.Softmax(dim=-1))
        self.register_buffer('scale', torch.tensor([1, 1, 1, 1, 1, 10, 100]))
        self.register_buffer('constraint', torch.tensor([[10, 10]]))

    def forward(self, agent_map, agent_agent):
        feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, 3, 1)], dim=-1)
        actions = self.control(feature).view(-1, 3, self._future_steps, 2)
        dummy = torch.ones(1, 1).to(self.cost[0].weight.device)
        cost_function_weights = torch.cat([self.cost(dummy)[:, :7] * self.scale, self.constraint], dim=-1)

        return actions, cost_function_weights


class AgentDecoderRSS(nn.Module):
    def __init__(self, future_steps, agents):
        super(AgentDecoderRSS, self).__init__()
        self._future_steps = future_steps
        self.num_agents = agents
        self.decode = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(EMB_DIM * 2, 256),
            nn.ELU(),
            nn.Linear(256, future_steps * 6),
        )

    def transform(self, prediction, current_state):
        x = current_state[:, 0]
        y = current_state[:, 1]
        theta = current_state[:, 2]
        delta_x = prediction[:, :, 0]
        delta_y = prediction[:, :, 1]
        delta_theta = prediction[:, :, 2]
        new_x = x.unsqueeze(1) + delta_x
        new_y = y.unsqueeze(1) + delta_y
        new_theta = theta.unsqueeze(1) + delta_theta
        rss = prediction[:,:,3:]
        traj = torch.stack([new_x, new_y,new_theta], dim=-1)
        traj_rss = torch.cat([traj,rss],dim=-1)


        return traj_rss

    def forward(self, feature, current_state):
        """
        feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, 3, 1, 1)], dim=-1)
        decoded = self.decode(feature).view(-1, 3, 10, self._future_steps, 3)
        trajs = torch.stack([self.transform(decoded[:, i, j], current_state[:, j]) for i in range(3) for j in range(10)], dim=1)
        trajs = torch.reshape(trajs, (-1, 3, 10, self._future_steps, 3))
        """

        modes = feature.shape[1]
        # decoded = self.decode(feature).view(
        #     -1, modes, self.num_agents, self._future_steps, 3
        # )

        decoded = self.decode(feature).view(
            -1, modes, self.num_agents, self._future_steps, 6      #changes made for predicting RSS for each neighbor
        )
        # trajs = torch.stack(
        #     [
        #         self.transform(decoded[:, i, j], current_state[:, j])
        #         for i in range(modes)
        #         for j in range(self.num_agents)
        #     ],
        #     dim=1,
        # )
        rss = torch.reshape(
            decoded, (-1, modes, self.num_agents, self._future_steps, 6)
        )

        return rss

class Score(nn.Module):
    def __init__(self):
        super(Score, self).__init__()
        self.reduce = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(EMB_DIM * 2, EMB_DIM), nn.ELU()
        )
        self.decode = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(EMB_DIM * 2, 128), nn.ELU(), nn.Linear(128, 1)
        )

    def forward(self, map_feature, agent_agent, agent_map, modes=MODES):
        # now uses per-agent features instead of max pooling, to compute separate score for each neighbor

        # loop over agents
        scores_per_agent = []
        for i in range(map_feature.shape[1]):
            agent_map_ = agent_map[:, :, i]
            map_feature_ = torch.max(map_feature[:, i], dim=1)[
                0
            ]  # torch.max(map_feature, dim=1)[0]
            agent_agent_ = agent_agent[:, i]  # torch.max(agent_agent, dim=1)[0]
            feature = torch.cat([map_feature_, agent_agent_], dim=-1)
            feature = self.reduce(feature.detach())
            feature = torch.cat(
                [feature.unsqueeze(1).repeat(1, modes, 1), agent_map_.detach()], dim=-1
            )
            scores = self.decode(feature).squeeze(-1)
            # scores = torch.softmax(scores, dim=1) # TODO remove softmax for normal score function
            scores_per_agent.append(scores.unsqueeze(1))
        scores_per_agent = torch.stack(scores_per_agent, dim=1).squeeze(-2)
        ego_scores = scores_per_agent[:, 0]
        neighbor_scores = scores_per_agent[:, 1:]

        return ego_scores, neighbor_scores


# Agent history encoder
class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(AGENT_FEATURES, EMB_DIM, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs[:, :, :AGENT_FEATURES])
        output = traj[:, -1]

        return output


# Agent history encoder
class AgentEncoderGT(nn.Module):
    def __init__(self):
        super(AgentEncoderGT, self).__init__()
        self.motion = nn.LSTM(5, EMB_DIM, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs[:, :, :5])
        output = traj[:, -1]

        return output





def entropy_normal_from_logvar(self, logvar):
        return 0.5 * (np.log(2.0 * np.pi * np.e) + logvar)

