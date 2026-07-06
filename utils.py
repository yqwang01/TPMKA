import torch
import numpy as np
from pathlib import PurePosixPath
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference


def split_sample_path(path):
    return PurePosixPath(str(path).replace("\\", "/")).parts


def metric(output, target, sens_attr=None, group_wise=False, equity_alpha=1.):
    
    return_dict = {}
    target = np.array(target)
    output = np.array(output, dtype=np.float32)
    num_classes = output.shape[1] if output.ndim >= 2 else len(np.unique(target))
    labels = list(range(num_classes))
    output_pred = np.argmax(output, axis=1)
    if np.isnan(output_pred).any():
        print("Warning: NaN detected in predictions! Replacing with 0.5...")
        output_pred = np.nan_to_num(output_pred, nan=0.5)

    if np.isnan(output).any():
        print("Warning: NaN detected in predictions! Replacing with 0.5...")
        output = np.nan_to_num(output, nan=0.5)
    
    return_dict['overall_confusion_matrix'] = confusion_matrix(target, output_pred)
    return_dict['overall_classification_report'] = classification_report(
        target, output_pred, labels=labels, output_dict=True, zero_division="warn"
    )
    return_dict['overall_bacc'] = balanced_accuracy_score(target, output_pred)
    return_dict['overall_auc'] = compute_auc(output, target, num_classes=num_classes)
    
    if sens_attr is not None and len(sens_attr) > 0:
        sens_attr = np.array(sens_attr)
        return_dict['overall_ES_acc'] = equity_scaled_accuracy(output, target, sens_attr, alpha=equity_alpha)
        return_dict['overall_ES_AUC'] = equity_scaled_AUC(output, target, sens_attr, alpha=equity_alpha, num_classes=len(np.unique(target)))
        return_dict['overall_ES_Class_Attr_Scaled_F1'] = equity_class_attr_scaled_f1(output, target, sens_attr, alpha=equity_alpha)
        return_dict['overall_EO'] = multiclass_equalized_odds(output, target, sens_attr)
        return_dict['overall_DP'] = multiclass_demographic_parity(output, target, sens_attr)

        if group_wise:
            for one_attr in np.unique(sens_attr):
                idx = sens_attr == one_attr
                attr_output = output[idx]
                attr_target = target[idx]
                attr_output_pred = np.argmax(attr_output, axis=1)
                return_dict['attr_{}_confusion_matrix'.format(one_attr)] = confusion_matrix(attr_target, attr_output_pred)
                return_dict['attr_{}_classification_report'.format(one_attr)] = classification_report(
                    attr_target, attr_output_pred, labels=labels, output_dict=True, zero_division="warn"
                )
                return_dict['attr_{}_AUC'.format(one_attr)] = compute_auc(attr_output, attr_target, num_classes=num_classes)

    return return_dict

def compute_auc(pred_prob, y, num_classes=2):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")

    y_onehot = num_to_onehot(y, num_classes)
    try:
        auc_val = roc_auc_score(y_onehot, pred_prob, average='macro', multi_class='ovr')
    except ValueError:
        return float("nan")

    return auc_val


def num_to_onehot(nums, num_to_class):
    nums = nums.astype(int)
    n_values = num_to_class
    onehot_vec = np.eye(n_values)[nums]
    return onehot_vec


def prob_to_label(pred_prob):
    # Find the indices of the highest probabilities for each sample
    max_prob_indices = np.argmax(pred_prob, axis=1)

    # Create one-hot vectors for each sample
    one_hot_vectors = np.zeros_like(pred_prob)
    one_hot_vectors[np.arange(len(max_prob_indices)), max_prob_indices] = 1

    return one_hot_vectors


def numeric_to_one_hot(y, num_classes=None):
    y = np.asarray(y, dtype=np.int32)

    if num_classes is None:
        num_classes = np.max(y) + 1
    
    one_hot_array = np.zeros((len(y), num_classes))
    one_hot_array[np.arange(len(y)), y] = 1
    
    return one_hot_array


def multiclass_demographic_parity(pred_prob, y, attrs):

    pred_one_hot = prob_to_label(pred_prob)
    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = demographic_parity_difference(pred_one_hot[:,i],
                                gt_one_hot[:,i],
                                sensitive_features=attrs)
        scores.append(tmp_score)

    avg_score = np.mean(scores)

    return avg_score


def multiclass_equalized_odds(pred_prob, y, attrs):

    pred_one_hot = prob_to_label(pred_prob)
    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = equalized_odds_difference(pred_one_hot[:,i],
                            gt_one_hot[:,i],
                            sensitive_features=attrs)
        scores.append(tmp_score)

    avg_score = np.mean(scores)
        
    return avg_score


def equity_scaled_accuracy(output, target, attrs, alpha=1.):
    es_acc = 0
    if len(output.shape) >= 2:
        overall_acc = np.sum(np.argmax(output, axis=1) == target)/target.shape[0]
    else:
        overall_acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if len(pred_group.shape) >= 2:
            acc = np.sum(np.argmax(pred_group, axis=1) == gt_group)/gt_group.shape[0]
        else:
            acc = np.sum((pred_group >= 0.5).astype(float) == gt_group)/gt_group.shape[0]

        identity_wise_perf.append(acc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_acc)
    es_acc = (overall_acc / (alpha*tmp + 1))
    
    return es_acc


def equity_class_attr_scaled_f1(output, target, attrs, alpha=1.):
    es_f1 = 0
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []

    output_pred = np.argmax(output, axis=1)
    overall_f1 = f1_score(target, output_pred, average='macro')

    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]
        pred_group_pred = np.argmax(pred_group, axis=1)

        f1_per_class = f1_score(gt_group, pred_group_pred, average=None, labels=[0, 1], zero_division="warn") # type: ignore
        for i in range(len(f1_per_class)):  # pyright: ignore[reportArgumentType]
            identity_wise_perf.append(f1_per_class[i])  # pyright: ignore[reportIndexIssue]
            identity_wise_num.append(gt_group.shape[0])


    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_f1)
    es_f1 = (overall_f1 / (alpha*tmp + 1))

    return es_f1


def equity_scaled_AUC(output, target, attrs, alpha=1., num_classes=2):
    es_auc = 0
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    

    y_onehot = num_to_onehot(target, num_classes)
    overall_auc = roc_auc_score(y_onehot, output, average='macro', multi_class='ovr')


    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        y_onehot = num_to_onehot(gt_group, num_classes)
        group_auc = roc_auc_score(y_onehot, pred_group, average='macro', multi_class='ovr')
        
        identity_wise_perf.append(group_auc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_auc)
    es_auc = (overall_auc / (alpha*tmp + 1))

    return es_auc
