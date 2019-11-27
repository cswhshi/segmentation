# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
import numpy as np
from mypath import Path

def calculate_weigths_labels(dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
    np.save(classes_weights_path, ret)

    return ret

"""
humanparsing_classes_weights：

array([ 1.89052439, 46.85964492, 23.10886715, 49.04330667,  8.61486792,
       21.30913324, 16.86125489, 16.7120056 , 47.87030499, 44.00055106,
       44.11843348, 31.20368993, 32.57941705, 32.93326263, 32.55160972,
       32.53228729, 28.5528652 , 44.14586841])
"""
