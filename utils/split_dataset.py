from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def stratified_split(dataset):
    labels = [label for _, label in dataset.samples]

    train_idx, temp_idx = train_test_split(
        range(len(labels)),
        test_size=0.3,
        stratify=labels,
        random_state=42
    )

    temp_labels = [labels[i] for i in temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)