# Import necessary libraries for data manipulation and transformation
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class ITO(Dataset):
    def __init__(self, cfg):
        super().__init__()  # Initialize the parent class (Dataset)

        # Load and process the data based on configuration settings
        self._load(cfg.datapath, cfg.concat_channels, cfg.use_one_channel)
        # Define a transformation pipeline to convert numpy arrays to PyTorch tensors
        self.transform = transforms.Compose([transforms.ToTensor()])

    def _load(self, datapath, concat_channels, use_one_channel):
        # Load the dataset from a CSV file
        raw_data = pd.read_csv(datapath)
        data = []  # Initialize an empty list to hold processed data

        # Define a list of conditions or labels present in the dataset
        condition_list = ["M1", "M2", "M3", "Normal", "P1", "P2", "P3"]
        # Extract condition labels from the dataset
        conditions = np.asarray(raw_data["Condition"])

        # Assuming fixed sensor names, extract sensor data
        S11 = np.asarray(raw_data["S11"])
        S21 = np.asarray(raw_data["S21"])

        # Process each entry in the dataset
        for i in range(len(conditions)):
            if use_one_channel:
                # If configured to use one channel, process S21 sensor data only
                measurements = np.asarray([float(v) for v in S21[i].split(" ")])
                if not concat_channels:
                    # If channels should not be concatenated, ensure measurements are 2D
                    measurements = np.expand_dims(measurements, axis=-1)
            else:
                # If using both channels (S11 and S21)
                if concat_channels:
                    # Concatenate measurements from both sensors into a single flat array
                    measurements = np.asarray(
                        [float(v) for v in S11[i].split(" ")]
                        + [float(v) for v in S21[i].split(" ")]
                    )
                else:
                    # Stack measurements from both sensors into a 2D array
                    measurements = np.stack(
                        [
                            np.asarray([float(v) for v in S11[i].split(" ")]),
                            np.asarray([float(v) for v in S21[i].split(" ")]),
                        ],
                        axis=-1,
                    )
            # Append the condition label and processed measurements to the data list
            data.append([condition_list.index(conditions[i]), measurements])

        self.data = data  # Store processed data in the class instance

    def __len__(self):
        # Return the total number of items in the dataset
        return len(self.data)

    def __getitem__(self, index):
        # Retrieve an item by its index
        condition, measurements = self.data[index]
        # Return condition and measurements, converting measurements to a numpy array if not already
        return condition, np.asarray(measurements)


def get_dataset(cfg):
    # A function to instantiate the ITO dataset(s) based on configuration
    if type(cfg.datapath) == str:
        # If datapath is a single string, load a single dataset
        return ITO(cfg)
    else:  # If datapath is a list, load multiple datasets
        datasets = []  # Initialize an empty list for datasets
        datapaths = (
            cfg.datapath.copy()
        )  # Copy the list of datapaths to avoid modifying the original

        for datapath in datapaths:
            cfg.datapath = datapath  # Update the datapath in the configuration
            datasets.append(ITO(cfg))  # Create and append a new dataset instance

        return datasets  # Return the list of dataset instances
