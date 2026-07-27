import torch.nn as nn


class LossCurveCNN(nn.Module):
    """1D-CNN regressor over fixed-length [C, T] feature tensors.

    The only trained backbone in ``phase7_release``: used for every
    ``f01..f07`` CNN experiment.  Transformer backbones were removed.
    """

    def __init__(self, input_channels: int = 2, sequence_length: int = 1500):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * (sequence_length // 4), 128),
            nn.ReLU(),
        )
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        return self.regressor(self.features(x))
