# import torch
# from multiprocessing import Pool
# from torch.utils.data import DataLoader

# class BatchProcessor:
#     def __init__(self, num_workers):
#         self.num_workers = num_workers

#     def process_batch(self, batch, func):
#         with Pool(self.num_workers) as pool:
#             results = pool.map(func, batch)
#         return torch.stack(results)

# class MyDataset:
#     def __init__(self, data):
#         self.data = data

#     def __getitem__(self, idx):
#         return self.data[idx]

#     def __len__(self):
#         return len(self.data)

#     def process_tensor(self, tensor):
#         return tensor * 2

# # Create a dataset with 10 random tensors
# data = [torch.randn(3, 5) for _ in range(10)]
# dataset = MyDataset(data)

# # Create a dataloader with batch size of 4
# dataloader = DataLoader(dataset, batch_size=4)

# # Create a BatchProcessor object with 2 workers
# batch_processor = BatchProcessor(num_workers=2)

# # Iterate over the dataloader and apply process_tensor to each tensor in each batch
# for batch in dataloader:
#     breakpoint()
#     processed_batch = batch_processor.process_batch(batch, dataset.process_tensor)
#     print(processed_batch)


import os
import numpy as np

# Dictionary of 2D NumPy arrays
# data_dict = {"key1": np.array([[1, 2, 3], [4, 5, 6]]),
#              "key2": np.array([[7, 8, 9], [10, 11, 12]])}

# New key and corresponding 2D NumPy array
new_key = "key4"
new_value = np.array([[13, 14, 15], [16, 17, 18]])

data_dict = {}
data_dict[new_key] = new_value

# File path to save or update the dictionary
file_path = "data.npz"

# Check if file exists
if os.path.isfile(file_path):
    # Load existing data from file
    data = np.load(file_path, allow_pickle=True)
    # Update the loaded data with the new dictionary
    data_dict.update(dict(data))
else:
    # Create a new dictionary with the provided data
    data_dict = data_dict
breakpoint()
# Add new key-value pair if not exist
if new_key not in data_dict:
    data_dict[new_key] = new_value

# Save the updated or new data to the file
np.savez(file_path, **data_dict)