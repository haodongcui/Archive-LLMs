from torch.utils.data import Dataset

from utils.tokenizer import MyBPETokenizer

class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.tokenizer = MyBPETokenizer()
        self.tokenizer.load("tokenizer.json")

    def __len__(self):
        return

    def __getitem__(self, index):
        return 