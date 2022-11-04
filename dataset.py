# References:
# https://www.geeksforgeeks.org/python-k-middle-elements/

import os
import random

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from util import SampleType, DataMode

class TactileDataset(Dataset):
    def __init__(self, path, data_mode, label_scale, mode=None, 
        seq_length=None, sample_type=SampleType.RANDOM, 
        angle_difference=False, num_features=142, transform=None):

        # Get all the csvs
        self.data_path = path if mode is None else path+mode+'/'
        self.initial_path = os.getcwd()
        os.chdir(self.data_path)
        self.datapoints = os.listdir('.')
        os.chdir(self.initial_path)

        self.seq_length = seq_length
        self.label_scale = label_scale
        self.angle_difference = angle_difference
        self.num_features = num_features
        self.transform = transform
        self.data_mode = data_mode

        # Defning positions of angle/velocity in the data csvs
        # needs to be changed if unfiltered data is used
        self.crop_idx = -7
        self.angle_idx = -5
        self.vel_idx = -1 

        self.sample_type = sample_type

    def __len__(self):
        return len(self.datapoints)

    def strechAngle(self, x, data_type = DataMode.POSITION):
        # Unnormalize a datapoint x
        if type(self.label_scale) == type([]):
            label_idx = 0 if data_type == DataMode.VELOCITY else 1
            return x * self.label_scale[label_idx] 
        else:
            return x * self.label_scale 

    def __getitem__(self, i):
        df = pd.read_csv(self.data_path + self.datapoints[i])
        true_values = df.values

        # Get input features
        features = torch.Tensor(true_values[:, :self.crop_idx]).float()

        # Get the normalized gt angle and velocity
        if self.data_mode == DataMode.BOTH:
            angle = true_values[:, self.angle_idx] / self.label_scale[1]
            velocity = true_values[:, self.vel_idx] / self.label_scale[0]
        else:
            angle = true_values[:, self.angle_idx] / self.label_scale
            velocity = true_values[:, self.vel_idx] / self.label_scale

        # Apply transforms with some probability
        # Unused in paper
        if self.transform and np.random.uniform() > 0.6:
            features = self.transform(features)

        return features, angle, velocity

    def collate_fn(self, batch):

        # 0 is the data, 1 is GT
        min_length = min(map(lambda x: x[0].shape[0], batch))

        # https://www.geeksforgeeks.org/python-k-middle-elements/
        if self.seq_length is None:
            K = min_length
        elif self.seq_length > min_length:
            print(self.seq_length, ' ', min_length)
            ValueError('Seq length is too long!')
        else:
            K = self.seq_length

        # Initialize outputs
        features_array = torch.zeros((len(batch), K, self.num_features))
        if self.angle_difference:
            angle_array = torch.zeros((len(batch)))
        else:
            angle_array = torch.zeros((len(batch), K))

        true_velocity = torch.zeros((len(batch), K))

        for (index, (data, angle, velocity)) in enumerate(batch):

            # Based on sample type get the start index of the crop
            if self.sample_type == SampleType.CENTER:
                strt_idx = (len(data) // 2) - (K // 2)
            elif self.sample_type == SampleType.FRONT:
                strt_idx = 0
            elif self.sample_type == SampleType.RANDOM:
                max_start_value = len(data) - K
                strt_idx = random.randrange(0, max_start_value+1, 1)
            else:
                ValueError('Invalid sample type')

            # Crop the current sequence
            data_cropped = data[strt_idx: strt_idx + K, :]
            angle_cropped = angle[strt_idx: strt_idx + K]
            velocity_cropped = velocity[strt_idx: strt_idx + K]
            features_array[index, :, :] = data_cropped

            # If using angle difference sub the first value to ensure angle starts at 0
            if self.angle_difference:
                angle_array[index] = torch.tensor(angle_cropped - angle_cropped[0])[-1]
            else:
                angle_array[index, :] = torch.tensor(angle_cropped - angle_cropped[0])

            true_velocity[index, :] = torch.tensor(velocity_cropped - velocity_cropped[0])

        return features_array, angle_array, true_velocity
