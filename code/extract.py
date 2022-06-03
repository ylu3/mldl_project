#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

topK = 20
data_root = '../datasets/ROD-synROD'

def extract_dataset(topK, input, ouput):
    input = os.path.join(data_root, input)
    df_rod = pd.read_csv(input, sep=" ", header=None)
    df_rod.columns = ["path", "category"]
    df_rod.groupby('category').head(topK).to_csv(ouput, header=None, index=None, sep=' ')

extract_dataset(topK, 'ROD/wrgbd_40k-split_sync.txt', 'smalldatasets/smallROD.txt')
extract_dataset(topK, 'synROD/synARID_50k-split_sync_train1.txt', 'smalldatasets/smallsynROD_train.txt')
extract_dataset(topK, 'synROD/synARID_50k-split_sync_test1.txt', 'smalldatasets/smalltsynROD_test.txt')