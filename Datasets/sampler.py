import numpy as np
from torch.utils.data import Sampler


class CellSampler(Sampler):
    def __init__(self, data, seed=None):
        self.data = data
        self.seed = seed
        n_pair = 0
        for row_index in self.data.index:
            n_pair += len(self.data.loc[row_index, 'cell_order']) - 1
        self.n_pair = n_pair

    def __len__(self):
        return self.n_pair

    def __iter__(self):
        pairs = []
        for row_index in self.data.index:
            cells = self.data.loc[row_index, 'cell_order'].copy()
            if self.seed:
                rng = np.random.default_rng(self.seed)
                rng.shuffle(cells)
            else:
                np.random.shuffle(cells)
            for cell_index in range(len(cells) - 1):
                pairs.append([row_index, cells[cell_index], cells[cell_index + 1]])

        for pair in pairs:
            yield pair