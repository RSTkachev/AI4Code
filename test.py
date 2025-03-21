import torch
from tqdm import tqdm
from utils import kendall_tau
from functools import cmp_to_key


class Tester:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def test(self, test_dataloader):
        self.model.to(self.device)
        self.model.eval()
        true_order = []
        predicted_order = []
        
        with torch.no_grad():
            for cells, correct_order in tqdm(test_dataloader):
                sorted_cells = sorted(cells, key=cmp_to_key(self._custom_compare))
                sorted_order = [cell[0] for cell in sorted_cells]
                true_order.append(correct_order)
                predicted_order.append(sorted_order)
        
        return kendall_tau(true_order, predicted_order)
   
    def _custom_compare(self, cell1, cell2):
        result = self.model(
            cell1[1].squeeze(0).to(self.device), cell1[2].squeeze(0).to(self.device), cell1[3].squeeze(0).to(self.device),
            cell2[1].squeeze(0).to(self.device), cell2[2].squeeze(0).to(self.device), cell2[3].squeeze(0).to(self.device)
        )
        
        if result.item() <= 0.5:
            return -1
        else:
            return 1
        