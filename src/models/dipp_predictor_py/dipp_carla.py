import torch
from torch import nn


# Agent history encoder
class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(7, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs[:, :, :7])
        output = traj[:, -1]

        return output


# Local context encoders
class LaneEncoder(nn.Module):
    def __init__(self):
        super(LaneEncoder, self).__init__()
        # encoder layer
        self.self_line = nn.Linear(3, 128)
        self.left_line = nn.Linear(3, 128)
        self.right_line = nn.Linear(3, 128)
        self.speed_limit = nn.Linear(1, 64)
        self.self_type = nn.Embedding(4, 64, padding_idx=0)
        self.left_type = nn.Embedding(11, 64, padding_idx=0)
        self.right_type = nn.Embedding(11, 64, padding_idx=0)
        self.traffic_light_type = nn.Embedding(9, 64, padding_idx=0)
        self.interpolating = nn.Embedding(2, 64)
        self.stop_sign = nn.Embedding(2, 64)
        self.stop_point = nn.Embedding(2, 64)

        # hidden layers
        self.pointnet = nn.Sequential(
            nn.Linear(448, 384),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.ReLU()
        )  # TODO 512

    def forward(self, inputs):
        # embedding
        self_line = self.self_line(inputs[..., :3])
        left_line = self.left_line(inputs[..., 3:6])
        right_line = self.right_line(inputs[..., 6:9])
        speed_limit = self.speed_limit(inputs[..., 9].unsqueeze(-1))
        # TODO self_type = self.self_type(inputs[..., 10].int())
        # TODO left_type = self.left_type(inputs[..., 11].int())
        # TODO right_type = self.right_type(inputs[..., 12].int())
        # TODO traffic_light = self.traffic_light_type(inputs[..., 30].int())
        # TODO stop_point = self.stop_point(inputs[..., 14].int())
        # TODO interpolating = self.interpolating(inputs[..., 15].int())
        # TODO stop_sign = self.stop_sign(inputs[..., 16].int())

        # TODO lane_attr = self_type + left_type + right_type + traffic_light + stop_point + interpolating + stop_sign
        lane_embedding = torch.cat([self_line, left_line, right_line, speed_limit], dim=-1)  # TODO , lane_attr

        # process
        output = self.pointnet(lane_embedding)

        return output


class CrosswalkEncoder(nn.Module):
    def __init__(self):
        super(CrosswalkEncoder, self).__init__()
        self.point_net = nn.Sequential(
            nn.Linear(2, 64),  # TODO 3
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

    def forward(self, inputs):
        output = self.point_net(inputs)

        return output


# Transformer-based encoders
class Agent2Agent(nn.Module):
    def __init__(self):
        super(Agent2Agent, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024, activation='relu', batch_first=True
        )
        self.interaction_net = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, inputs, mask=None):
        return self.interaction_net(inputs, src_key_padding_mask=mask)


# Transformer modules
class CrossTransformer(nn.Module):
    def __init__(self):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(256, 8, 0.1, batch_first=True)
        self.transformer = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(1024, 256),
            nn.LayerNorm(256)
        )

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        output = self.transformer(attention_output)

        return output


class MultiModalTransformer(nn.Module):
    def __init__(self, modes=3, output_dim=256):
        super(MultiModalTransformer, self).__init__()
        self.modes = modes
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(256, 4, 0.1, batch_first=True)
            for _ in range(modes)
        ])
        self.ffn = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, query, key, value, mask=None):
        attention_output = []
        for i in range(self.modes):
            attention_output.append(self.attention[i](query, key, value, key_padding_mask=mask)[0])
        attention_output = torch.stack(attention_output, dim=1)
        output = self.ffn(attention_output)

        return output


class Agent2Map(nn.Module):
    def __init__(self):
        super(Agent2Map, self).__init__()
        self.lane_attention = CrossTransformer()
        self.crosswalk_attention = CrossTransformer()
        self.map_attention = MultiModalTransformer()

    def forward(self, actor, lanes, crosswalks, mask):
        query = actor.unsqueeze(1)
        lanes_actor = [self.lane_attention(query, lanes[:, i], lanes[:, i]) for i in range(lanes.shape[1])]
        crosswalks_actor = [self.crosswalk_attention(query, crosswalks[:, i], crosswalks[:, i]) for i in
                            range(crosswalks.shape[1])]
        map_actor = torch.cat(lanes_actor + crosswalks_actor, dim=1)
        output = self.map_attention(query, map_actor, map_actor, mask).squeeze(2)

        return map_actor, output


# Decoders
class AgentDecoder(nn.Module):
    def __init__(self, future_steps):
        super(AgentDecoder, self).__init__()
        self._future_steps = future_steps
        self.decode = nn.Sequential(nn.Dropout(0.0), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, future_steps * 3))

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
        decoded = self.decode(feature).view(-1, 3, 20, self._future_steps, 3)
        trajs = torch.stack(
            [self.transform(decoded[:, i, j], current_state[:, j]) for i in range(3) for j in range(20)], dim=1)
        trajs = torch.reshape(trajs, (-1, 3, 20, self._future_steps, 3))

        return trajs


class AVDecoder(nn.Module):
    def __init__(self, future_steps=50, feature_len=9):
        super(AVDecoder, self).__init__()
        self._future_steps = future_steps
        self.control = nn.Sequential(nn.Dropout(0.0), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, future_steps * 3))
        self.cost = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, feature_len), nn.Softmax(dim=-1))
        self.register_buffer('scale', torch.tensor([1, 1, 1, 1, 1, 10, 100]))
        self.register_buffer('constraint', torch.tensor([[10, 10]]))

    def forward(self, agent_map, agent_agent):
        feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, 3, 1)], dim=-1)
        actions = self.control(feature).view(-1, 3, self._future_steps, 3)
        dummy = torch.ones(1, 1).to(self.cost[0].weight.device)
        cost_function_weights = torch.cat([self.cost(dummy)[:, :7] * self.scale, self.constraint], dim=-1)

        return actions, cost_function_weights


class Score(nn.Module):
    def __init__(self):
        super(Score, self).__init__()
        self.reduce = nn.Sequential(nn.Dropout(0.0), nn.Linear(512, 256), nn.ELU())
        self.decode = nn.Sequential(nn.Dropout(0.0), nn.Linear(512, 128), nn.ELU(), nn.Linear(128, 1))

    def forward(self, map_feature, agent_agent, agent_map):
        # pooling
        map_feature = map_feature.view(map_feature.shape[0], -1, map_feature.shape[-1])
        map_feature = torch.max(map_feature, dim=1)[0]
        agent_agent = torch.max(agent_agent, dim=1)[0]
        agent_map = torch.max(agent_map, dim=2)[0]

        feature = torch.cat([map_feature, agent_agent], dim=-1)
        feature = self.reduce(feature.detach())
        feature = torch.cat([feature.unsqueeze(1).repeat(1, 3, 1), agent_map.detach()], dim=-1)
        scores = self.decode(feature).squeeze(-1)

        return scores


class DIPPEncoder(nn.Module):

    def __init__(self):
        super(DIPPEncoder, self).__init__()

        # agent layer
        self.vehicle_net = AgentEncoder()
        self.pedestrian_net = AgentEncoder()
        self.cyclist_net = AgentEncoder()

        # map layer
        self.lane_net = LaneEncoder()
        self.crosswalk_net = CrosswalkEncoder()

        # attention layers
        self.agent_map = Agent2Map()
        self.agent_agent = Agent2Agent()

        # output layer
        self.outdim = 512
        self.output = nn.Linear(11264, self.outdim)

    def forward(self, obs):
        ego = obs["ego"]
        neighbors = obs["neighbors"]
        map_lanes = obs["map_lanes"]
        map_crosswalks = obs["map_crosswalks"]

        # actors
        ego_actor = self.vehicle_net(ego)
        vehicles = torch.stack([self.vehicle_net(neighbors[:, i]) for i in range(neighbors.shape[1])], dim=1)
        pedestrians = torch.stack([self.pedestrian_net(neighbors[:, i]) for i in range(neighbors.shape[1])], dim=1)
        cyclists = torch.stack([self.cyclist_net(neighbors[:, i]) for i in range(neighbors.shape[1])], dim=1)
        neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2) == 2, pedestrians, vehicles)
        neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2) == 3, cyclists, neighbor_actors)
        actors = torch.cat([ego_actor.unsqueeze(1), neighbor_actors], dim=1)
        actor_mask = torch.eq(torch.cat([ego.unsqueeze(1), neighbors[..., :]], dim=1), 0)[:, :, -1, -1]

        # maps
        lane_feature = self.lane_net(map_lanes)
        crosswalk_feature = self.crosswalk_net(map_crosswalks)
        lane_mask = torch.eq(map_lanes, 0)[:, :, 0, 0]  # TODO :,
        crosswalk_mask = torch.eq(map_crosswalks, 0)[:, :, 0, 0]  # TODO :,
        map_mask = torch.cat([lane_mask, crosswalk_mask], dim=1)
        map_mask[:, 0] = False  # prevent nan  # TODO :,

        # actor to actor
        agent_agent = self.agent_agent(actors, actor_mask)

        # vmap for speed, can be replaced with for loop for engine (same computation)
        def get_agent_map(a):
            return self.agent_map(a, lane_feature, crosswalk_feature, map_mask)
            # TODO return self.agent_map(a, lane_feature[:, 0], crosswalk_feature[:, 0], map_mask[:, 0])

        res = torch.vmap(get_agent_map, randomness="same", in_dims=0, out_dims=1)(agent_agent.transpose(1, 0))
        map_feature = res[0].contiguous()
        agent_map = res[1].transpose(2, 1).contiguous()

        agent_agent = agent_agent.reshape(1, -1)
        agent_map = agent_map.reshape(1, -1)
        features = torch.cat([agent_agent, agent_map], dim=1)
        return self.output(features)


class DIPPDecoder(nn.Module):

    def __init__(self, future_steps):
        super(DIPPDecoder, self).__init__()
        self._future_steps = future_steps

        # decode layers
        self.plan = AVDecoder(self._future_steps)
        self.predict = AgentDecoder(self._future_steps)
        self.score = Score()

    def forward(self, neighbors, map_feature, agent_agent, agent_map):

        # plan + prediction
        plans, cost_function_weights = self.plan(agent_map[:, :, 0], agent_agent[:, 0])
        predictions = self.predict(agent_map[:, :, 1:], agent_agent[:, 1:], neighbors[:, :, -1])
        scores = self.score(map_feature, agent_agent, agent_map)

        return plans, predictions, scores, cost_function_weights
