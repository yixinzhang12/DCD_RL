import torch
import pickle

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = []
        self._preprocess_data(data_path)

    def _preprocess_data(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, index):
    #     item = self.data[index]
    #     # TODO YOUR CODE HERE
    #     raise NotImplementedError()
    
    def __getitem__(self, index):
        item = self.data[index]
        state_np, action_np = item
        state = torch.tensor(state_np, dtype=torch.float32)
        action = torch.tensor(action_np, dtype=torch.long)
        return {'state': state, 'action': action} 
