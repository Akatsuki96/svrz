from torch.utils.data import random_split



def build_synthetic_dataset(full_dataset, tr_fraction = 0.8, vl_fraction = 0.3, generator = None):
    n = len(full_dataset)
    n_tr = int(n * tr_fraction)
    n_vl = int(n_tr * vl_fraction)
    training_dataset, test_dataset = random_split(full_dataset, lengths=[n_tr, n - n_tr], generator=generator)
    training_dataset, validation_dataset = random_split(training_dataset, lengths=[n_tr - n_vl, n_vl], generator=generator)
    return training_dataset, validation_dataset, test_dataset