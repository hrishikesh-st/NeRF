import torch
import torch.nn as nn
import numpy as np

PI = np.pi


class NeRFModel(nn.Module):

    def __init__(self, embed_pos_L=10, embed_direction_L=4, hidden_dim_1=256, hidden_dim_2=128):
        super(NeRFModel, self).__init__()
        #############################
        # network initialization
        #############################

        self.embed_pos_L = embed_pos_L
        self.embed_direction_L = embed_direction_L

        pos_shape = 3 + self.embed_pos_L * 3 * 2
        dir_shape = 3 + self.embed_direction_L * 3 * 2

        self.input_layer = nn.Linear(pos_shape, hidden_dim_1) # 1

        self.block_1 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_1), # 2
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_1), # 3
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_1), # 4
            nn.ReLU()
        )

        self.skip_layer = nn.Linear(hidden_dim_1 + pos_shape, hidden_dim_1) # 5

        self.block_2 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_1), # 6
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_1), # 7
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_1) # 8
        )

        self.sigma_layer = nn.Linear(hidden_dim_1, hidden_dim_1 + 1) # 9

        self.dir_layer = nn.Linear(hidden_dim_1 + dir_shape, hidden_dim_2) # 10

        self.output_layer = nn.Linear(hidden_dim_2, 3) # 11

        self.relu = nn.functional.relu
        self.sigmoid = nn.functional.sigmoid

        self.double()

        
    def position_encoding(self, x, L):
        #############################
        # Implement position encoding here
        #############################
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)


    def forward(self, pos, direction):
        #############################
        # network structure
        #############################

        encoded_pos = self.position_encoding(pos, self.embed_pos_L)
        encoded_dir = self.position_encoding(direction, self.embed_direction_L)

        x = self.input_layer(encoded_pos)
        x = self.relu(x)
        x = self.block_1(x)
        x = self.skip_layer(torch.cat((x, encoded_pos), -1))
        x = self.relu(x)
        x = self.block_2(x)
        x = self.sigma_layer(x)
        sigma = x[:, -1]
        feats = x[:, :-1]
        sigma = self.relu(sigma)

        x = self.dir_layer(torch.cat((feats, encoded_dir), -1))
        x = self.relu(x)
        x = self.output_layer(x)
        rgb = self.sigmoid(x)

        # output = torch.cat((rgb, sigma), -1)

        return rgb, sigma
