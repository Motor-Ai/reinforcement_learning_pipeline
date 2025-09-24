import torch
from torch import nn


class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.network = nn.LSTM(7, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.network(inputs[:, :, :7])
        return traj[:, -1]


class LaneEncoder(nn.Module):
    def __init__(self):
        super(LaneEncoder, self).__init__()
        self.network = nn.LSTM(10, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.network(inputs)
        return traj[:, -1]


class CrosswalkEncoder(nn.Module):
    def __init__(self):
        super(CrosswalkEncoder, self).__init__()
        self.network = nn.LSTM(2, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.network(inputs)
        return traj[:, -1]


class SequenceModel(nn.Module):
    def __init__(self):
        super(SequenceModel, self).__init__()
        self.network = nn.LSTM(256, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.network(inputs)
        return traj[:, -1]


class SmallEncoder(nn.Module):

    def __init__(self):
        super(SmallEncoder, self).__init__()

        # actor, lane and crosswalk networks
        self.actor_net = AgentEncoder()
        self.lane_net = LaneEncoder()
        self.crosswalk_net = CrosswalkEncoder()

        self.actor_seq = SequenceModel()
        self.lane_seq = SequenceModel()
        self.crosswalk_seq = SequenceModel()

        # output layer
        self.outdim = 512
        self.output = nn.Linear(256 * 3, self.outdim)

    def forward(self, obs):
        ego = obs["ego"].unsqueeze(dim=1)
        neighbors = obs["neighbors"]
        map_lanes = obs["map_lanes"]
        map_crosswalks = obs["map_crosswalks"]

        actors = torch.cat([ego, neighbors], dim=1)
        actors = torch.stack([self.actor_net(actors[:, i]) for i in range(actors.shape[1])], dim=1)

        map_lanes = torch.stack([self.lane_net(map_lanes[:, i]) for i in range(map_lanes.shape[1])], dim=1)
        map_crosswalks = torch.stack([self.crosswalk_net(map_crosswalks[:, i]) for i in range(map_crosswalks.shape[1])], dim=1)

        actor_embeddings = self.actor_seq(actors)
        lane_embeddings = self.lane_seq(map_lanes)
        crosswalk_embeddings = self.crosswalk_seq(map_crosswalks)
        features = torch.cat([actor_embeddings, lane_embeddings, crosswalk_embeddings], dim=1)

        return self.output(features)


