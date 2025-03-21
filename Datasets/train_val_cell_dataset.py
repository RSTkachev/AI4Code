from Datasets.cell_dataset import CellDataset


class TrainValCellDataset(CellDataset):
    def __init__(self, path, data, tokenizer, max_length):
        super().__init__(path, data, tokenizer, max_length)
        
        n_pair = 0
        for row_index in self.data.index:
            n_pair += len(self.data.loc[row_index, 'cell_order']) - 1
        self.n_pair = n_pair
               
        
    def __len__(self):
        return self.n_pair
        
    def __getitem__(self, idx):
        filename = idx[0]
        first_cell_id = idx[1]
        second_cell_id = idx[2]

        first_position = self.data.loc[filename, 'cell_order'].index(first_cell_id)
        second_position = self.data.loc[filename, 'cell_order'].index(second_cell_id)
        order = 0 if first_position < second_position else 1
        
        return (
            (
                self.files[filename][first_cell_id].get(),
                self.files[filename][second_cell_id].get()
            ),
            order
        )
