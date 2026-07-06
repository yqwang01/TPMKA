from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from numpy.core import defchararray
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms as T
import random
from functools import reduce
import torchvision.transforms.functional as F
from copy import deepcopy


def dataset_split(train_csv_path, num_var, moda, var_idx=-1, sen_attr_idx=-1, site=None, for_json=False, avail_moda=5):
    
    train_csv_file = pd.read_csv(train_csv_path).values
    train_valid_cases = train_csv_file[train_csv_file[:, 1 + avail_moda] > -1]

    np.random.shuffle(train_valid_cases)
    print("The number of total valid patients in the dataset:", len(train_valid_cases))
    moda_dict = {"B": 0, "ScS": 1, "W": 2, "O": 3, "S": 4}
    if site is not None:
        if site == 'India':
            train_indices = ((train_valid_cases[:, moda_dict[moda]] == 1) & (np.char.find(train_valid_cases[:, avail_moda + 2].astype(str), "Aravind") >= 0)).tolist()  # pyright: ignore[reportOperatorIssue]  # pyright: ignore[reportOperatorIssue]
        elif site == 'US':
            train_indices = ((train_valid_cases[:, moda_dict[moda]] == 1) & (np.char.find(train_valid_cases[:, avail_moda + 2].astype(str), "Aravind") < 0)).tolist()
        else:
            raise ValueError("Invalid site specified. Choose either 'India' or 'US'.")
    else:
        if avail_moda != 0:
            train_indices = (train_valid_cases[:, moda_dict[moda]] == 1).tolist()
        else:
            train_indices = [True] * len(train_valid_cases)
    train_cases = train_valid_cases[:, avail_moda:]

    train_list, train_label, train_sen_attr, train_var = [], [], [], []

    for case in train_cases:
        train_list.append(case[0])
        train_label.append(case[1])
        if sen_attr_idx >= 0:
            train_sen_attr.append(case[sen_attr_idx + 3])
        if num_var > 0:
            train_var.append(case[var_idx + 3: var_idx + 3 + num_var].astype(np.float32))
    return train_list, train_label, train_sen_attr, train_var, train_indices # type: ignore


def dataset_split_ensemble(train_csv_path, num_var, var_idx=-1, sen_attr_idx=-1, site=None, for_json=False, avail_moda=5):
    
    train_csv_file = pd.read_csv(train_csv_path).values  # pyright: ignore[reportAttributeAccessIssue]
    train_valid_cases = train_csv_file[train_csv_file[:, 1 + avail_moda] > -1]

    np.random.shuffle(train_valid_cases)
    print("The number of total valid patients in training set:", len(train_valid_cases))
    if site is not None:
        if site == 'India':
            train_indices = (np.char.find(train_valid_cases[:, avail_moda + 2].astype(str), "Aravind") >= 0).tolist()  # pyright: ignore[reportOperatorIssue]
        elif site == 'US':
            train_indices = (np.char.find(train_valid_cases[:, avail_moda + 2].astype(str), "Aravind") < 0).tolist()
        else:
            raise ValueError("Invalid site specified. Choose either 'India' or 'US'.")
    else:
        train_indices = None
    train_cases = train_valid_cases[:, avail_moda:]
    
    train_list, train_label, train_var, train_sen_attr = [], [], [], []
    for case in train_cases:
        train_list.append(case[0])
        train_label.append(case[1])
        if sen_attr_idx >= 0:
            train_sen_attr.append(case[sen_attr_idx + 3])
        if num_var > 0:
            train_var.append(case[var_idx + 3: var_idx + 3 + num_var].astype(np.float32))
    return train_list, train_label, train_var, train_sen_attr, train_indices # type: ignore


def dataset_split_feature(data_dir, train_csv_path, num_var, moda, var_idx=-1, sen_attr_idx=-1, for_json=False, avail_moda=5):
    
    train_csv_file = pd.read_csv(train_csv_path).values
    train_valid_cases = train_csv_file[train_csv_file[:, 1 + avail_moda] > -1]

    np.random.shuffle(train_valid_cases)
    print("The number of total valid patients in training set:", len(train_valid_cases))
    moda_dict = {"B": 0, "ScS": 1, "W": 2, "O": 3, "S": 4}
    if moda is not None:
        train_moda_indices = (train_valid_cases[:, moda_dict[moda]] == 1)
    else:
        train_moda_indices = [True] * len(train_valid_cases)
    train_cases = train_valid_cases[train_moda_indices, avail_moda:]

    pat_list, test_list, test_label, test_sen_attr, test_var = [], [], [], [], []
    for case in train_cases:
        patient = case[0]
        pat_list.append(patient)
        if moda is not None:
            for file_name in os.listdir(os.path.join(data_dir, patient, moda)):
                test_list.append(os.path.join(patient, moda, file_name))
                test_label.append(case[1])
                if sen_attr_idx >= 0:
                    test_sen_attr.append(case[sen_attr_idx + 3])
                if num_var > 0:
                    test_var.append(case[var_idx + 3: var_idx + 3 + num_var].astype(np.float32))
        else:
            for file_name in os.listdir(os.path.join(data_dir, patient)):
                test_list.append(os.path.join(patient, file_name))
                test_label.append(case[1])
                if sen_attr_idx >= 0:
                    test_sen_attr.append(case[sen_attr_idx + 3])
                if num_var > 0:
                    test_var.append(case[var_idx + 3: var_idx + 3 + num_var].astype(np.float32))

    return pat_list, test_list, test_label, test_sen_attr, test_var


def k_fold_cross_validation(full_list, full_label, full_sen_attr, full_var, data_dir, moda, moda_indices, multi_moda, fold_id, folds=5):

    cut_point_1 = int(fold_id / folds * len(full_list))
    cut_point_2 = int((fold_id + 1) / folds * len(full_list))
    train_indices = np.array(moda_indices[: cut_point_1] + moda_indices[cut_point_2:])
    val_indices = np.array(moda_indices[cut_point_1: cut_point_2])
    train_list = np.array(full_list[: cut_point_1] + full_list[cut_point_2:])
    val_list = np.array(full_list[cut_point_1: cut_point_2])
    train_label = np.array(full_label[: cut_point_1] + full_label[cut_point_2:])
    val_label = np.array(full_label[cut_point_1: cut_point_2])
    train_var = np.array(full_var[: cut_point_1] + full_var[cut_point_2:])
    val_var = np.array(full_var[cut_point_1: cut_point_2])
    train_sen_attr = np.array(full_sen_attr[: cut_point_1] + full_sen_attr[cut_point_2:])
    val_sen_attr = np.array(full_sen_attr[cut_point_1: cut_point_2])

    if not multi_moda:
        train_list = (train_list[train_indices]).tolist()
        val_list = (val_list[val_indices]).tolist()
        train_label = (train_label[train_indices]).tolist()
        val_label = (val_label[val_indices]).tolist()
        if full_var is not None and len(full_var) > 0:
            train_var = train_var[train_indices]
            val_var = val_var[val_indices]
        if full_sen_attr is not None and len(full_sen_attr) > 0:
            train_sen_attr = train_sen_attr[train_indices]
            val_sen_attr = val_sen_attr[val_indices]
    else:
        train_list = train_list.tolist()
        val_list = val_list.tolist()
        train_label = train_label.tolist()
        val_label = val_label.tolist()

    print("The number of valid cases in training set:", len(train_list))
    print("The number of valid cases in validation set:", len(val_list))

    test_list, test_label, test_sen_attr, test_var = [], [], [], []

    if not multi_moda:
        if full_sen_attr is not None and len(full_sen_attr) > 0 and full_var is not None and len(full_var) > 0:
            for case in zip(val_list, val_label, val_sen_attr, val_var):
                patient = case[0]
                for file_name in os.listdir(os.path.join(data_dir, patient, moda if moda is not None else "")):
                    test_list.append(os.path.join(patient, moda if moda is not None else "", file_name))
                    test_label.append(case[1])
                    test_sen_attr.append(case[2])
                    test_var.append(case[3])
        elif full_sen_attr is not None and len(full_sen_attr) > 0:
            for case in zip(val_list, val_label, val_sen_attr):
                patient = case[0]
                for file_name in os.listdir(os.path.join(data_dir, patient, moda if moda is not None else "")):
                    test_list.append(os.path.join(patient, moda if moda is not None else "", file_name))
                    test_label.append(case[1])
                    test_sen_attr.append(case[2])
        elif full_var is not None and len(full_var) > 0:
            for case in zip(val_list, val_label, val_var):
                patient = case[0]
                for file_name in os.listdir(os.path.join(data_dir, patient, moda if moda is not None else "")):
                    test_list.append(os.path.join(patient, moda if moda is not None else "", file_name))
                    test_label.append(case[1])
                    test_var.append(case[2])
        else:
            for case in zip(val_list, val_label):
                patient = case[0]
                for file_name in os.listdir(os.path.join(data_dir, patient, moda if moda is not None else "")):
                    test_list.append(os.path.join(patient, moda if moda is not None else "", file_name))
                    test_label.append(case[1])

    return train_list, val_list, test_list, train_label, val_label, test_label, train_var, val_var, test_var, train_sen_attr, val_sen_attr, test_sen_attr


def k_fold_cross_validation_ensemble(full_list, full_label, full_sen_attr, full_var, fold_id, indices=None, folds=5):

    cut_point_1 = int(fold_id / folds * len(full_list))
    cut_point_2 = int((fold_id + 1) / folds * len(full_list))
    train_list = full_list[: cut_point_1] + full_list[cut_point_2:]
    val_list = full_list[cut_point_1: cut_point_2]
    train_label = full_label[: cut_point_1] + full_label[cut_point_2:]
    val_label = full_label[cut_point_1: cut_point_2]
    train_sen_attr = full_sen_attr[: cut_point_1] + full_sen_attr[cut_point_2:]
    val_sen_attr = full_sen_attr[cut_point_1: cut_point_2]
    if len(full_var) > 0:
        train_var = full_var[: cut_point_1] + full_var[cut_point_2:]
        val_var = full_var[cut_point_1: cut_point_2]
    else:
        train_var, val_var = [], []

    
    if indices is not None:
        train_indices = indices[: cut_point_1] + indices[cut_point_2:]
        val_indices = indices[cut_point_1: cut_point_2]
        train_list = np.array(train_list)[train_indices].tolist()
        val_list = np.array(val_list)[val_indices].tolist()
        train_label = np.array(train_label)[train_indices].tolist()
        val_label = np.array(val_label)[val_indices].tolist()
        train_sen_attr = np.array(train_sen_attr)[train_indices].tolist()
        val_sen_attr = np.array(val_sen_attr)[val_indices].tolist()
        if len(full_var) > 0:
            train_var = np.array(train_var)[train_indices]
            val_var = np.array(val_var)[val_indices]
        else:
            train_var, val_var = [], []

    return train_list, val_list, train_label, val_label, train_sen_attr, val_sen_attr, train_var, val_var


def dataset_sampling_weights(sample_mode, sens_attr_list, label_list, weight_power=1.0):

    num_labels = len(set(label_list))
    num_groups = len(set(sens_attr_list))

    if sample_mode == "group" or sample_mode == "balanced":
        group_array = deepcopy(sens_attr_list)
        if sample_mode == "balanced":
            group_array = (np.asarray(group_array) * num_labels + np.asarray(label_list)).tolist()
        
    elif sample_mode == "class":
        group_array = label_list
    
    else:
        raise ValueError("Invalid sample mode. Choose from 'group', 'balanced', or 'class'.")
    
    group_array = torch.LongTensor(group_array)
    if sample_mode == 'group':
        group_counts = (torch.arange(num_groups).unsqueeze(1) == group_array).sum(1).float()
    elif sample_mode == 'balanced':
        group_counts = (torch.arange(num_labels * num_groups).unsqueeze(1) == group_array).sum(1).float()
    elif sample_mode == 'class':
        group_counts = (torch.arange(num_labels).unsqueeze(1) == group_array).sum(1).float()
    else:
        raise ValueError("Invalid sample mode. Choose from 'group', 'balanced', or 'class'.")

    group_weights = [1.0 / (x.item() ** weight_power) for x in group_counts]
    sample_weights = [group_weights[int(i)] for i in group_array]
    return sample_weights


def train_transform(image, img_size, mask=None, second_mask=None):
    color_jitter = T.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6)
    image = color_jitter(image)

    if random.random() > 0.5:
        image = F.hflip(image)
        if mask is not None:
            mask = F.hflip(mask)
        if second_mask is not None:
            second_mask = F.hflip(second_mask)
    if random.random() > 0.5:
        image = F.vflip(image)
        if mask is not None:
            mask = F.vflip(mask)
        if second_mask is not None:
            second_mask = F.vflip(second_mask)
    
    angle = random.uniform(-30, 30)
    image = F.rotate(image, angle)
    if mask is not None:
        mask = F.rotate(mask, angle)
    if second_mask is not None:
        second_mask = F.rotate(second_mask, angle)

    width, height = image.size # type: ignore
    scale = (0.8, 1.0)
    ratio = (3.0 / 4.0, 4.0 / 3.0)
    area = width * height
    target_area = random.uniform(scale[0], scale[1]) * area
    aspect_ratio = random.uniform(ratio[0], ratio[1])
    crop_width = int(round((target_area * aspect_ratio) ** 0.5))
    crop_height = int(round((target_area / aspect_ratio) ** 0.5))
    if crop_width > width:
        crop_width = width
    if crop_height > height:
        crop_height = height
    top = random.randint(0, height - crop_height)
    left = random.randint(0, width - crop_width)
    image = F.resized_crop(image, top, left, crop_height, crop_width, img_size)
    if mask is not None:
        mask = F.resized_crop(mask, top, left, crop_height, crop_width, img_size)
    if second_mask is not None:
        second_mask = F.resized_crop(second_mask, top, left, crop_height, crop_width, img_size)
    
    return image, mask, second_mask


def raw_collate_fn(batch):
    from torch.utils.data._utils.collate import default_collate

    labels = default_collate([b["label"] for b in batch])
    name = default_collate([b["name"] for b in batch])
    var = default_collate([b["var"] for b in batch])

    if "image" in batch[0]:
        images = default_collate([b["image"] for b in batch])
        return {
            "image": images,
            "var": var,
            "label": labels,
            "name": name,
        }
    else:
        images_1 = default_collate([b["image_1"] for b in batch])
        images_2 = default_collate([b["image_2"] for b in batch])
        return {
            "image_1": images_1,
            "image_2": images_2,
            "var": var,
            "label": labels,
            "name": name,
        }



class AQUADataset(Dataset):

    def __init__(self, data_dir, img_path_list, label_list, var_list, sens_attr_list, moda, moda_list, 
                 multi_moda=False, img_size=(256, 256), pair=False, train=False):
        super().__init__()
        
        self.data_dir = data_dir
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.var_list = var_list
        self.sens_attr_list = sens_attr_list
        self.moda = moda
        self.moda_list = moda_list
        self.multi_moda = multi_moda
        self.img_size = img_size
        self.pair = pair
        self.train = train
        
        self.transforms_img = T.Compose([
                                T.Resize(self.img_size), 
                                T.ToTensor(),      
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    def __getitem__(self, idx):

        pat_id = self.img_path_list[idx]
        labels = self.label_list[idx]
        name = pat_id
        sens_attr = self.sens_attr_list[idx] if len(self.sens_attr_list) > 0 else None
        if self.var_list is not None:
            var = self.var_list[idx]
            var = torch.from_numpy(var)
        else:
            var = None
        
        if not self.pair:
            img_path = os.path.join(self.data_dir, pat_id)
            img_file = np.load(img_path, allow_pickle=True)
            img = Image.fromarray(img_file['img'])
            img_file.close()
            
            if self.train:
                img, mask, second_mask = train_transform(img, self.img_size)
            image = self.transforms_img(img)
            
            return_dict = {
                "image": image,
                "label": labels,
                "name": name,
                "sens_attr": sens_attr if len(self.sens_attr_list) > 0 else -1,
            }
            if var is not None:
                return_dict["var"] = var
            return return_dict
        
        else:
            img_list = []
            if self.multi_moda:
                for moda in self.moda_list:
                    if os.path.exists(os.path.join(self.data_dir, name, moda)):
                        moda_imgs = os.listdir(os.path.join(self.data_dir, name, moda))
                        for moda_img in moda_imgs:
                            img_list.append(moda + '/' + moda_img)
            else:
                moda_imgs = os.listdir(os.path.join(self.data_dir, name, self.moda))
                for moda_img in moda_imgs:
                    img_list.append(self.moda + '/' + moda_img)

            random.shuffle(img_list)
            img_path_1 = os.path.join(self.data_dir, name, img_list[0])
            img_1_file = np.load(img_path_1, allow_pickle=True)
            img_1 = Image.fromarray(img_1_file['img'])
            img_1_file.close()
            if self.train:
                img_1, _, _ = train_transform(img_1, self.img_size)
            image_1 = self.transforms_img(img_1)
            

            img_path_2 = os.path.join(self.data_dir, name, img_list[-1])
            img_2_file = np.load(img_path_2, allow_pickle=True)
            img_2 = Image.fromarray(img_2_file['img'])
            img_2_file.close()
            if self.train:
                img_2, _, _ = train_transform(img_2, self.img_size)
            image_2 = self.transforms_img(img_2)

            return_dict = {
                "image_1": image_1,
                "image_2": image_2,
                "label": labels,
                "name": name,
                "sens_attr": sens_attr if len(self.sens_attr_list) > 0 else -1,
            }
            if var is not None:
                return_dict["var"] = var
            return return_dict

    def __len__(self):
        return len(self.img_path_list)


class EnsembleDataset(Dataset):

    def __init__(self, data_dir, img_path_list, label_list, moda_list, sens_attr_list):
        super().__init__()
        
        self.data_dir = data_dir
        self.label_list = label_list
        self.moda_list = moda_list
        self.img_path_list = img_path_list
        self.sens_attr_list = sens_attr_list

    def __getitem__(self, idx):

        labels = self.label_list[idx]
        name = self.img_path_list[idx]
        sens_attr = self.sens_attr_list[idx] if len(self.sens_attr_list) > 0 else -1

        pat_moda = os.listdir(os.path.join(self.data_dir, name))
        feat_array = np.zeros((len(self.moda_list), 512), dtype=np.float32)
        for i, moda in enumerate(self.moda_list):
            if moda in pat_moda:
                moda_feat = []
                moda_feats = os.listdir(os.path.join(self.data_dir, name, moda))
                for moda_f in moda_feats:
                    feat_path = os.path.join(self.data_dir, name, moda, moda_f)
                    feat_file = np.load(feat_path, allow_pickle=True)
                    feat = np.asarray(feat_file['img'], dtype=np.float32).squeeze()
                    feat_file.close()
                    moda_feat.append(feat)
                moda_feat_mean = np.mean(np.stack(moda_feat, axis=0), axis=0)
                feat_array[i] = moda_feat_mean

        x_feat = torch.from_numpy(feat_array).float()

        return (x_feat, labels, sens_attr, name)

    def __len__(self):
        return len(self.img_path_list)
    
