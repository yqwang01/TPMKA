# TPMKA
# Copyright (C) 2026 Duke University and the Regents of the University of Michigan.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import torch
from torch import nn


class LinearEnsembleNet(nn.Module):
    def __init__(self, num_moda):
        super().__init__()
        # self.fc1 = nn.Linear(1536, 512)
        # self.bn1 = nn.BatchNorm1d(512)  # BatchNorm层
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(512, 64)
        # self.bn2 = nn.BatchNorm1d(64)  # 另一个BatchNorm层
        # self.fc3 = nn.Linear(64, 2)
        self.fc1 = nn.Linear(num_moda * 512, 256)
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Linear(256, 32)
        self.relu2 = nn.ReLU(True)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # x = self.bn1(self.relu(self.fc1(x)))
        # x = self.bn2(self.relu(self.fc2(x)))
        # x = self.fc3(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x