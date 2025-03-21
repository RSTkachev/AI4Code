class Cell:
    def __init__(self, input_ids, att_mask, cell_type):
        self.input_ids = input_ids
        self.att_mask = att_mask
        self.cell_type = cell_type

    def get(self):
        return (self.input_ids, self.att_mask, self.cell_type)
