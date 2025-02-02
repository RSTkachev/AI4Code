from ignite.metrics import Metric
from bisect import bisect
from collections import defaultdict
from typing import Dict

def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions


def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0  
    for gt, pred in zip(ground_truth, predictions):
        ranks = [
            gt.index(x) for x in pred
        ] 
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max

class KendallTauNaive(Metric):
    def __init__(self, val_data: Dict[str, Sample]):
        super().__init__()
        self.val_data = val_data
        self.reset()

    def reset(self):
        self._predictions = defaultdict(dict)
        self._all_predictions = []
        self._all_targets = []
        self._submission_data = {}

    def update(self, output):
        loss, scores, sample_ids, cell_ids = output
        for score, sample_id, cell_id in zip(scores, sample_ids, cell_ids):
            self._predictions[sample_id][cell_id] = score.item()

    def compute(self):
        for sample in self.val_data.values():
            all_preds = []
            for cell_key in sample.cell_keys:
                cell_type = sample.cell_types[cell_key]
                cell_rank = sample.cell_ranks_normed[cell_key]
                if cell_type == "code":
                    # keep the original cell_rank
                    item = (cell_key, cell_rank)
                else:
                    item = (cell_key, self._predictions[sample.id][cell_key])
                all_preds.append(item)
            cell_id_predicted = [
                item[0] for item in sorted(all_preds, key=lambda x: x[1])
            ]
            self._submission_data[sample.id] = cell_id_predicted
            self._all_predictions.append(cell_id_predicted)
            self._all_targets.append(sample.orders)

        score = kendall_tau(self._all_targets, self._all_predictions)
        print("Kendall Tau: ", score)
        return score