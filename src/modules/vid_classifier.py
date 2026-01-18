import torch
import torch.nn as nn   
from analysis_modules import *

class VID_Classifier(nn.Module):
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

        # CRITICAL FIX: Use .reshape() instead of .view() or call .contiguous()
        x_video_folded = x_video.reshape(b*t, c, h, w)  # Changed from .view() to .reshape()
        visuals = self.eff_net_module(x_video_folded)
        visuals = visuals.reshape(b, t, -1)  # Changed from .view() to .reshape()

        # Frequency processing
        frequency = self.hfri_module(x_video_folded)
        frequency = frequency.reshape(b, t, -1)  # Changed from .view() to .reshape()

        # Concatenate all features
        combined = torch.cat([visuals, frequency, frame_motion], dim=2)

        # LSTM processing
        lstm_out, hidden_state = self.lstm(combined, hidden_state)

        # Final classification
        logits = self.classifier(lstm_out)

        return logits, hidden_state

