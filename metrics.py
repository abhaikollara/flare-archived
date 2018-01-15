import torch
from torch.nn import functional as F

#
# Some helper functions
#


def get_metric(metric_name, data):
    names = {
        'accuracy': categorical_accuracy,
        'top_k': top_k_categorical_accuracy,
        'f1': f1_score,
    }

    y_pred, y_true = data
    return names[metric_name](y_pred, y_true)


def _one_hot(sparse_var, num_classes):
    onehot = torch.zeros(sparse_var.size()[0], num_classes)
    onehot.scatter_(1, sparse_var.data.view(-1, 1), 1)
    return onehot


def _bincount(x, n_elems):
    return torch.sum(_one_hot(x, n_elems), dim=0)


def categorical_accuracy(preds, targets, sparse=True):
    """
    # Arguments
        preds : Output of softmax layer
        targets: Target values, one hot encoded
                 if sparse is False
        sparse: Denotes whether targets are one-hot
                tensors or indices
    # Returns
        Mean accuracy as float
    """
    class_preds = torch.max(preds, dim=-1)[1]
    if sparse is False:
        targets = torch.max(targets, dim=-1)[1]

    return torch.mean(torch.eq(class_preds, targets).float()).data[0]


def top_k_categorical_accuracy(preds, targets, k=3, sparse=True):
    """
    # Arguments
        preds : Output of softmax layer
        targets: Target values, one hot encoded
                 if sparse is False
        k: Specifies the number of top k classes
           to be considered
        sparse: Denotes whether targets are one-hot
                tensors or indices
    # Returns
        Mean accuracy as float
    """
    top_k = torch.topk(preds, dim=-1, k=k)[1]
    if sparse is False:
        targets = torch.max(targets, dim=-1)[1]

    targets = targets.view(-1, 1)
    acc = torch.sum(torch.eq(targets, top_k).float()) / targets.size()[0]
    return acc.data[0]


def f1_score(preds, targets, sparse=True):
    """ Calculates F1 score
    # Arguments
        preds : Output of softmax layer
        targets: Target values, one hot encoded
                 if sparse is False
        sparse: Denotes whether targets are one-hot
                tensors or indices

    # Returns
        A tuple whose first element is the f1_score
        and second element is a tuple containting
        the number of true positives, predicted
        postitives, true positives
    """
    num_classes = preds.size()[-1]
    class_preds = torch.max(preds, dim=-1)[1]

    tp = torch.eq(class_preds, targets)
    tp_bin = targets[tp]

    tp_sum = _bincount(tp_bin, num_classes)
    pred_sum = _bincount(class_preds, num_classes)
    true_sum = _bincount(targets, num_classes)

    precision = torch.mean(tp_sum / pred_sum, dim=-1)
    recall = torch.mean(tp_sum / true_sum, dim=-1)
    f1 = 2 * precision * recall / (precision + recall)

    return (f1[0], (tp_sum, pred_sum, true_sum))
