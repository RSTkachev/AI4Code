import random
import torch
import os

def shuffle_batch(tensor):
    len_of_tensor = tensor.shape[0]
    shuffled_indices = random.sample(list(range(len_of_tensor)), len_of_tensor)
    unshuffled_indices = [shuffled_indices.index(k) for k, i in enumerate(shuffled_indices)]
    return shuffled_indices, unshuffled_indices


def get_ranks(cell_types, cell_orders, cell_keys):
    code_cells_num = len([cell_type for cell_type in cell_types if cell_type == "code"])
    code_cell_keys = cell_keys[:code_cells_num]

    code_bins = {(i, i+1): [] for i in range(code_cells_num + 1)}
    code_cell_orders = [cell_orders.index(cell_key) for cell_key in code_cell_keys]

    cell_ranks = {}
    for k, (cell_type, cell_order, cell_key) in enumerate(zip(cell_types, cell_orders, cell_keys)):
        cell_order = cell_orders.index(cell_key)
        if cell_type == "code":
            cell_ranks[cell_key] = k + 1
            continue
        for i, code_cell_order in enumerate(code_cell_orders):
            if cell_order < code_cell_order:
                code_bins[(i, i+1)].append((cell_order, cell_key))
                break
        else:
            code_bins[(i+1, i+2)].append((cell_order, cell_key))

    for bins, values in code_bins.items():
        markdowns_sorted = sorted(values, key=lambda x: x[0])
        step = 1 / (len(markdowns_sorted) + 1)
        for j, (markdown_cell_order, markdown_cell_key) in enumerate(markdowns_sorted):
            cell_ranks[markdown_cell_key] = bins[0] + step * (j + 1)

    return cell_ranks
