import torch
import torch.nn as nn   
from analysis_modules import *
import random

class VideoDeepFakeLSTMModel(nn.Module):
    def __init__(self, LSTM_hidden_size, num_layers):
        super().__init__()
        self.eff_net_module = NetworkProcesses.StreamA_EfficientNetNetwork()
        self.frame_difference_module = NetworkProcesses.FrameDifferenceNetwork()
        self.hfri_module = NetworkProcesses.HFRINetwork()
        
        self.LSTM_input_size = 512 + 128 + 128
        
        self.lstm = nn.LSTM(
            input_size=self.LSTM_input_size,
            hidden_size=LSTM_hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape is [Batch, Time, Channels, Height, Width]
            dropout=0.3
        )
        
        #Output Layer:
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1)
        )

    def forward(self, x_video, hidden_state = None):
        b, t, c, h, w = x_video.shape
        
        frame_motion = self.frame_difference_module(x_video)

        x_video_folded = x_video.view(b*t, c, h, w)
        visuals = self.eff_net_module(x_video_folded)
        visuals = visuals.view(b, t, -1)
        frequency = self.hfri_module(x_video_folded)
        frequency = frequency.view(b, t, -1)
        
        if self.training:
            visuals_dropout = random.random() < 0.15
            if visuals_dropout:
                visuals = visuals*0.0

        fusion_stack = torch.cat((visuals, frame_motion, frequency),dim=2)
        lstm_out, new_hidden_state = self.lstm(fusion_stack, hidden_state)
        logits = self.classifier(lstm_out)
        logits.view(b, t, 1)
        return logits, new_hidden_state

