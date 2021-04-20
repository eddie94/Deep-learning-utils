import json
import os
import cv2
import numpy as np
import torch

from torch.utils.data.dataset import Dataset
from torchvision.transforms import Normalize
from utils import channel_first


class ImgDatasetFromCSV(Dataset):
    def __init__(self, lines, path_to_img="./"):
        """
        A pytorch dataset loaded from a CSV file
        The data should follow the rules below

        File directory
        ./base_path
          └-> ./label1
            └-> imgs.jpg
          └-> ./label2
            └-> imgs.jpg
          ...
          labels.json

        CSV
        label/img_filename.jpg

        :param lines: lines from the csv
        :param path_to_img: path to where the Images exist
        """

        self.lines = lines
        self.path_to_img = path_to_img

        with open(os.path.join(path_to_img, "labels.json"), 'r') as jsonfile:
            self.labels = json.load(jsonfile)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        img = channel_first(
            cv2.resize(
                cv2.imread(os.path.join(self.path_to_img, self.lines[i])).astype(np.float32), (224, 224)
            )
        )

        normalized_img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(torch.tensor(img))

        label = self.labels[self.lines[i].split("/")[0]]

        return normalized_img, label
