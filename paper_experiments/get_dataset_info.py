import sys
import numpy as np


from utils import get_dataset, load_libsvm_data


data_directory = sys.argv[1]

datasets = ['ijcnn1', 'phishing', 'mushrooms', 'australian', 'german.numer', 'splice']

header_str = "\\hline\nDataset & $n$ & $d$ \\\\\n\\hline\n"


with open("./dataset_details.log", 'w') as f:
    print(header_str)
    f.write(header_str)    
    for dataset_name in datasets:
        dataset = get_dataset(name=dataset_name, data_path=data_directory)
        print(f"{dataset_name} & {dataset.X.shape[0]} & {dataset.d} \\\\")
        f.write(f"{dataset_name} & {dataset.X.shape[0]} & {dataset.d} \\\\ \n")
        f.flush()