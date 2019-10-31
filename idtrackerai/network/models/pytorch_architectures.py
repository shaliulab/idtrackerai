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
        self.conv1 = nn.Conv2d(input_shape[-1], 16, 5,
                               stride=1, padding=2, dilation=1,
                               groups=1, bias=True, padding_mode='zeros')
        w = compute_output_width(input_shape[1], 5, 2, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0, dilation=1,
                                  return_indices=False, ceil_mode=False)
        w = compute_output_width(w, 2, 0, 2)
        self.conv2 = nn.Conv2d(16, 64, 5,
                               stride=1,
                               padding=2,
                               dilation=1,
                               groups=1,
                               bias=True,
                               padding_mode='zeros')
        w = compute_output_width(w, 5, 2, 1)
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0, dilation=1,
                                  return_indices=False, ceil_mode=False)
        w = compute_output_width(w, 2, 0, 2)
        self.conv3 = nn.Conv2d(64, 100, 5,
                               stride=1,
                               padding=2,
                               dilation=1,
                               groups=1,
                               bias=True,
                               padding_mode='zeros')
        self.w = compute_output_width(w, 5, 2, 1)
        self.fc1 = nn.Linear(100 * w * w, 100)
        self.fc2 = nn.Linear(100, out_dim)

        self.conv = nn.Sequential(
            self.conv1, nn.ReLU(inplace=True), self.pool1,
            self.conv2, nn.ReLU(inplace=True), self.pool2,
            self.conv3, nn.ReLU(inplace=True)
        )

        self.linear = nn.Sequential(
            self.fc1, nn.ReLU(inplace=True)
        )

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
        self.conv1 = nn.Conv2d(input_shape[-1], 16, 5,
                               stride=1, padding=2, dilation=1,
                               groups=1, bias=True, padding_mode='zeros')
        w = compute_output_width(input_shape[1], 5, 2, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0, dilation=1,
                                  return_indices=False, ceil_mode=False)
        w = compute_output_width(w, 2, 0, 2)
        self.conv2 = nn.Conv2d(16, 64, 5,
                               stride=1,
                               padding=2,
                               dilation=1,
                               groups=1,
                               bias=True,
                               padding_mode='zeros')
        w = compute_output_width(w, 5, 2, 1)
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0, dilation=1,
                                  return_indices=False, ceil_mode=False)
        w = compute_output_width(w, 2, 0, 2)
        self.conv3 = nn.Conv2d(64, 100, 5,
                               stride=1,
                               padding=2,
                               dilation=1,
                               groups=1,
                               bias=True,
                               padding_mode='zeros')
        self.w = compute_output_width(w, 5, 2, 1)
        self.fc1 = nn.Linear(100 * w * w, 100)
        self.fc2 = nn.Linear(100, out_dim)

        self.conv = nn.Sequential(
            self.conv1, nn.ReLU(inplace=True), self.pool1,
            self.conv2, nn.ReLU(inplace=True), self.pool2,
            self.conv3, nn.ReLU(inplace=True)
        )

        self.linear = nn.Sequential(
            self.fc1, nn.ReLU(inplace=True)
        )

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
