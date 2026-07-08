from pathlib import Path

from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    """Placeholder dataset that reads samples from a directory.

    Replace this with a dataset implementation matching your data format.
    """

    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir) / split
        self.files = sorted(self.data_dir.glob("*")) if self.data_dir.exists() else []

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        raise NotImplementedError("Implement sample loading for your dataset")
