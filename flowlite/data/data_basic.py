from ..autograd import Tensor
from typing import Optional, List
import numpy as np


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """
    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms
        
    def __getitem__(self, index) -> object:
        raise NotImplementedError()
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            for trans in self.transforms:
                x = trans(x)
        return x
    
class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)), 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        self.cur_idx = 0
        return self

    def __next__(self):
        if self.cur_idx >= len(self.ordering):
            self.cur_idx = 0
            raise StopIteration
        batch_data = self.dataset[self.ordering[self.cur_idx]]
        batch_data = [Tensor(s) for s in batch_data]
        self.cur_idx += 1
        return batch_data

    