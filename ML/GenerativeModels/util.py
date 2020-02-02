import torch

def convert_onehot(label, num_classes):
    """Return one-hot encoding of given list of label

    Args: 
        label: A list of true label
        num_classes: The number of total classes for y
    Return:
        one_hot: Encoded vector of shape [length of data, num of classes]
    """
    one_hot = torch.zeros(label.shape[0], num_classes).scatter_(1, label, 1)
    return one_hot
