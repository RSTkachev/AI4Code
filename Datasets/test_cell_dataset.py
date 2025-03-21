import numpy as np

from Datasets.cell_dataset import CellDataset


class TestCellDataset(CellDataset):
    def __init__(self, path, data, tokenizer, max_length):
        super().__init__(path, data, tokenizer, max_length)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        file_id = self.data.iloc[idx].name
        correct_order = self.data.iloc[idx].item()
        random_order = correct_order.copy()
        np.random.shuffle(random_order)

        cells = []
        for index in random_order:
            input_ids, att_mask, cell_type = self.files[file_id][index].get()
            cells.append([index, input_ids, att_mask, cell_type])

        return cells, correct_order
