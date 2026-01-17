import numpy as np
import torch
import torch
from torch.utils.data import Dataset, DataLoader
import csv, cv2, os

from pathlib import Path
src_path = Path(__file__).resolve().parent.parent


class DF_Dataset(Dataset):
    def __init__(self, dataset_path: str = ".", training: bool = True):
        self.dataset_path = dataset_path
        self.training = training

        self.data = []

        with open(str(src_path / ("data/img_train.csv" if training else "data/img_test.csv")), "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0: continue
                
                self.data.append({
                    "path": row[5], 
                    "label": row[3]
                })

        self.epoch_size = len(self.data)
                    
    def __len__(self):
        return self.epoch_size
    
    def __getitem__(self, index):
        return self.get_image(index)
    
    def get_image(self, index):
        label = int(self.data[index]["label"])
        label = int(self.data[index]["label"])

        img = cv2.imread(os.path.join(self.dataset_path, self.data[index]["path"]).replace("\\", "/"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW
        
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return img_tensor, label_tensor
    

if __name__ == "__main__":
    BATCH_SIZE = 16
    NUM_EPOCHS = 32

    from dotenv import load_dotenv
    load_dotenv()
    IMG_DATASET_PATH = os.getenv("IMG_DATASET_PATH")

    ds = DF_Dataset(IMG_DATASET_PATH, training = True)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    import time
    from img_classifier import IMG_Classifier
    classifier = IMG_Classifier()

    for i in range(NUM_EPOCHS):
        e = time.perf_counter()
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.float() / 255.0
            classifier.forward(images)
            print("Batch processed.")
        print(f"Epoch - {i + 1} completed in {round(time.perf_counter() - e, 2)}s.")