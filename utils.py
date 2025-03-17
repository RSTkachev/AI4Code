import os
import torch
from time import localtime, strftime
from bisect import bisect


def prepare_folders():
    current_time = strftime('%d.%m.%Y-%H:%M', localtime())
    savedir = f'./checkpoints/{current_time}/'

    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints/')
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    else:
        for root, dirs, files in os.walk(savedir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    return savedir

def get_device():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
        else "cpu"
    )
    return device

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
        ranks = [gt.index(x) for x in pred]
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max
