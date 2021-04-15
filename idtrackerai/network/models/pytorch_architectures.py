# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)

import torch.nn as nn

from idtrackerai.network.models.models_utils import compute_output_width


class DCD(nn.Module):
    def __init__(self, input_shape, out_dim):
        """
        input_shape: tuple (channels, width, height)
        out_dim: int
        """
        super(DCD, self).__init__()

        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(
            input_shape[-1],
            16,
            5,
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        w = compute_output_width(input_shape[1], 5, 2, 1)
        self.pool1 = nn.MaxPool2d(
            2,
            stride=2,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )
        w = compute_output_width(w, 2, 0, 2)
        self.conv2 = nn.Conv2d(
            16,
            64,
            5,
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        w = compute_output_width(w, 5, 2, 1)
        self.pool2 = nn.MaxPool2d(
            2,
            stride=2,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )
        w = compute_output_width(w, 2, 0, 2)
        self.conv3 = nn.Conv2d(
            64,
            100,
            5,
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        self.w = compute_output_width(w, 5, 2, 1)
        self.fc1 = nn.Linear(100 * w * w, 100)
        self.fc2 = nn.Linear(100, out_dim)

        self.conv = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            self.pool1,
            self.conv2,
            nn.ReLU(inplace=True),
            self.pool2,
            self.conv3,
            nn.ReLU(inplace=True),
        )

        self.linear = nn.Sequential(self.fc1, nn.ReLU(inplace=True))

        self.last = self.fc2

    def features(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, 100 * self.w * self.w))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class idCNN(nn.Module):
    def __init__(self, input_shape, out_dim):
        """
        input_shape: tuple (channels, width, height)
        out_dim: int
        """
        super(idCNN, self).__init__()

        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(
            input_shape[-1],
            16,
            5,
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        w = compute_output_width(input_shape[1], 5, 2, 1)
        self.pool1 = nn.MaxPool2d(
            2,
            stride=2,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )
        w = compute_output_width(w, 2, 0, 2)
        self.conv2 = nn.Conv2d(
            16,
            64,
            5,
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        w = compute_output_width(w, 5, 2, 1)
        self.pool2 = nn.MaxPool2d(
            2,
            stride=2,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )
        w = compute_output_width(w, 2, 0, 2)
        self.conv3 = nn.Conv2d(
            64,
            100,
            5,
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        self.w = compute_output_width(w, 5, 2, 1)
        self.fc1 = nn.Linear(100 * w * w, 100)
        self.fc2 = nn.Linear(100, out_dim)

        self.conv = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            self.pool1,
            self.conv2,
            nn.ReLU(inplace=True),
            self.pool2,
            self.conv3,
            nn.ReLU(inplace=True),
        )

        self.linear = nn.Sequential(self.fc1, nn.ReLU(inplace=True))

        self.last = self.fc2

        self.softmax = nn.Softmax(dim=1)

    def features(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, 100 * self.w * self.w))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

    def softmax_probs(self, x):
        x = self.forward(x)
        x = self.softmax(x)
        return x


class idCNN_adaptive(nn.Module):
    def __init__(self, input_shape, out_dim):
        """
        input_shape: tuple (width, height, channels)
        out_dim: int
        """
        super(idCNN_adaptive, self).__init__()

        self.out_dim = out_dim
        num_channels = [input_shape[-1], 16, 64, 100]
        cnn_kwargs = dict(
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        maxpool_kwargs = dict(
            stride=2,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )
        kernel_size = 5
        self.width_adaptive_pool = 3

        # Convolutional and pooling layers
        cnn_layers = []
        for i, (num_ch_in, num_ch_out) in enumerate(
            zip(num_channels[:-1], num_channels[1:])
        ):
            if i > 0:
                # no pooling after input
                cnn_layers.append(nn.MaxPool2d(2, **maxpool_kwargs))

            cnn_layers.append(
                nn.Conv2d(num_ch_in, num_ch_out, kernel_size, **cnn_kwargs)
            )
            cnn_layers.append(nn.ReLU(inplace=True))

        cnn_layers.append(nn.AdaptiveAvgPool2d(self.width_adaptive_pool))
        self.conv = nn.Sequential(*cnn_layers)

        # Fully connected layers
        self.fc1 = nn.Linear(
            num_channels[-1] * self.width_adaptive_pool ** 2, 100
        )
        self.fc2 = nn.Linear(100, out_dim)
        self.linear = nn.Sequential(self.fc1, nn.ReLU(inplace=True))
        self.last = self.fc2

        self.softmax = nn.Softmax(dim=1)

    def features(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, 100 * self.width_adaptive_pool ** 2))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

    def softmax_probs(self, x):
        x = self.forward(x)
        x = self.softmax(x)
        return x
