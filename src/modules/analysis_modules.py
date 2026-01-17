import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class NetworkProcesses:
    class StreamA_EfficientNetNetwork(nn.Module):
        def __init__(self):
            super().__init__()

            print("Initializing EfficientNet-B0...")
            self.backbone = models.efficientnet_b0(weights = "DEFAULT")

            for param in self.backbone.parameters():
                param.requires_grad = False # Freezes the Weights in the Layer

            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Sequential(1280, 512),
                nn.ReLU()
            )
        
        def forward(self, x):
            return self.backbone(x)


    class FrameDifferenceNetwork(nn.Module):
        def __init__(self, output_dims = 128):
            super().__init__()
            layers = []
            layer_config = [
                {'in': 3,  'out': 16,  'kernel_size': 3, 'padding': 1}, 
                {'in': 16, 'out': 32,  'kernel_size': 3, 'padding': 1},
                {'in': 32, 'out': 64,  'kernel_size': 3, 'padding': 1},
                {'in': 64, 'out': 128, 'kernel_size': 3, 'padding': 1}
            ]
            for config in layer_config:
                layers.append(nn.Conv2d(config["in"], config["out"], config["kernel_size"],padding=config["padding"]))
                layers.append(nn.BatchNorm2d(config['out']))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(2,2))
            
            self.frame_difference_network = nn.Sequential(*layers)
            
            self.feed_forward = nn.Linear(128*16*16,output_dims)

        def forward(self, x_video):
            # Dimensions = [Batch, Time(Frames), Channels, ImageHeight, ImageWidth]

            frame_difference = x_video[:, 1:] - x_video[:, :-1]

            zeros = torch.zeros_like(x_video[:, 0:1]) # Dummy First Frame
            motion_maps = torch.cat([zeros, frame_difference], dim=1) #Restoring Frame Length

            b, t, c, h, w = motion_maps.shape
            
            x_cnn = motion_maps.view(b*t, c, h, w)
            frame_difference_features = self.frame_difference_network(x_cnn)
            frame_difference_features = torch.flatten(frame_difference_features, 1)

            Frame_Difference_Out = self.feed_forward(frame_difference_features)
            Frame_Difference_Out = Frame_Difference_Out.view(b, t, -1)
            return Frame_Difference_Out

    class HFRINetwork(nn.Module):
        def __init__(self):
            super().__init__()
            
            #Tiny CNN Initialization
            layers = []
            
            layer_config = [
                {'in': 1,  'out': 16,  'kernel_size': 3, 'padding': 1}, 
                {'in': 16, 'out': 32,  'kernel_size': 5, 'padding': 2},
                {'in': 32, 'out': 64,  'kernel_size': 3, 'padding': 1},
                {'in': 64, 'out': 128, 'kernel_size': 3, 'padding': 1}
            ]
            for config in layer_config:
                layers.append(nn.Conv2d(config["in"], config["out"], config["kernel_size"],padding=config["padding"]))
                layers.append(nn.BatchNorm2d(config['out']))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(2,2))

            self.network = nn.Sequential(*layers)
            final_size = 128*16*16
            self.feed_forward = nn.Linear(final_size, 128)

        
        def forward(self, input_x):
            input_x = 0.299 * input_x[:, 0:1] + 0.587 * input_x[:, 1:2] + 0.114 * input_x[:, 2:3]
            input_fft = torch.fft.fft2(input_x)
            input_fft_shifted = torch.fft.fftshift(input_fft)

            input_final_spectrum = torch.log(torch.abs(input_fft_shifted)+ 1e-8)

            HFRI_out = self.network(input_final_spectrum)
            HFRI_out = torch.flatten(HFRI_out,1) #Stacks all the rows one behind another
            HFRI_out = self.feed_forward(HFRI_out)

            return HFRI_out
            

    class HFRFSNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []

            ##Spatial Convolutions
            self.spatial_convolution = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU()
            )

            layer_config = [
                {'in': 16,  'out': 32,  'kernel_size': 3, 'padding': 1}, 
                {'in': 32, 'out': 64,  'kernel_size': 5, 'padding': 2},
                {'in': 64, 'out': 128, 'kernel_size': 3, 'padding': 1},
                {'in': 128, 'out': 256,  'kernel_size': 3, 'padding': 1}
            ]
            for config in layer_config:
                layers.append(nn.Conv2d(config["in"], config["out"], config["kernel_size"],padding=config["padding"]))
                layers.append(nn.BatchNorm2d(config['out']))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(2,2))

            self.network = nn.Sequential(*layers)
            final_size = 256*16*16
            self.feed_forward = nn.Linear(final_size, 128)

        def forward(self, input_x):
            input_x = 0.299 * input_x[:, 0:1] + 0.587 * input_x[:, 1:2] + 0.114 * input_x[:, 2:3]
            
            spatial_input_x = self.spatial_convolution(input_x)

            input_fft = torch.fft.fft2(spatial_input_x)
            input_fft_shifted = torch.fft.fftshift(input_fft)

            input_final_spectrum = torch.log(torch.abs(input_fft_shifted)+ 1e-8)

            HFRFS_out = self.network(input_final_spectrum)
            HFRFS_out = torch.flatten(HFRFS_out,1) 
            HFRFS_out = self.feed_forward(HFRFS_out)

            return HFRFS_out
        