from torch.utils.data import Dataset

class TensorDataset(Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == t.size(0) for t in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return len(self.tensors[0])
