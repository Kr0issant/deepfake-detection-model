import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import cv2, json, random

from pathlib import Path
src_path = Path(__file__).resolve().parent.parent

class DF_Dataset(Dataset):
    def __init__(self, dataset_path: str = ".", epoch_size: int = 400, training: bool = True):
        self.dataset_path = dataset_path
        self.epoch_size = epoch_size
        self.training = training

        if training:
            with open(str(src_path / "data/img_train.json"), "r") as f:
                self.data = json.load(f)
        else:
            with open(str(src_path / "data/img_test.json"), "r") as f:
                self.data = json.load(f)

    def __len__(self):
        return self.epoch_size
    
    def __getitem__(self, index):
        return self.create_video(True)
    

if __name__ == "__main__":
    BATCH_SIZE = 8
    EPOCH_SIZE = 400
    NUM_EPOCHS = 32

    ds = DF_Dataset("C:/Users/Krishna/Downloads/face-images-10k", EPOCH_SIZE, training = True)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    from img_classifier import IMG_Classifier
    classifier = IMG_Classifier()

    for i in range(NUM_EPOCHS):
        for batch_idx, (images, labels) in enumerate(loader):
            classifier.forward(images)
            print("Batch processed.")
        print(f"Epoch - {i} completed.")