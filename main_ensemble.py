import numpy as np
import argparse
from config import Config, ENSEMBLE_TRAIN, ENSEMBLE_TEST
import os, json
import random


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["ensemble_train", "ensemble_test"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
    args = parser.parse_args()
    if args.mode == "ensemble_train": 
        mode = ENSEMBLE_TRAIN
    else:
        mode = ENSEMBLE_TEST

    config = Config(mode)
    os.makedirs(config.output_dir, exist_ok=True)
    json_path = os.path.join(config.output_dir, 'hyperparameter.json')
    if mode != ENSEMBLE_TEST:
        with open(json_path,'w') as f:
            f.write(json.dumps(vars(config), ensure_ascii=False, indent=4, separators=(',', ':')))
        
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    random.seed(config.seed)
    np.random.seed(config.seed)

    from data_loader import dataset_split_ensemble, k_fold_cross_validation_ensemble, EnsembleDataset
    from torch.utils.data import DataLoader
    from ensemble_funcs import TrainingModel
    from losses import FocalLoss
    from torch.nn import CrossEntropyLoss
    from models.linear_ensemble import LinearEnsembleNet
    from data_loader import dataset_sampling_weights
    from torch.utils.data import WeightedRandomSampler
    import torch


    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    full_list, full_label, full_var, full_sen_attr, indices = dataset_split_ensemble( # type: ignore
                            config.train_csv_path, config.num_var, config.var_idx, config.sen_attr_idx, config.site, False, config.avail_moda)
    
    for fold_id in range(config.folds):
        print("\n-----------Fold: {}------------".format(fold_id))
        train_list, val_list, train_label, val_label, train_sen_attr, val_sen_attr, train_var, val_var = k_fold_cross_validation_ensemble(
                                        full_list, full_label, full_sen_attr, full_var, fold_id, indices, config.folds)
        
        if config.ense_model == "Linear":
            net = LinearEnsembleNet(num_moda=len(config.moda_list))
        else:
            raise ValueError("Unknown ensemble model: %s"%config.ense_model)

        fold_dir = os.path.join(config.feature_dir, 'fold_{}'.format(fold_id))
        dataset_train = EnsembleDataset(fold_dir, train_list, train_label, config.moda_list, train_sen_attr)
        dataset_val = EnsembleDataset(fold_dir, val_list, val_label, config.moda_list, val_sen_attr)
        dataset_test = EnsembleDataset(fold_dir, val_list, val_label, config.moda_list, val_sen_attr)

        if config.sampling_mode is not None:
            train_weights = dataset_sampling_weights(config.sampling_mode, train_sen_attr, train_label, weight_power=config.weight_power)
            print(f"Using {config.sampling_mode} sampling strategy for training data.")
            print("Sample weights range: [{:.4f}, {:.4f}]".format(min(train_weights), max(train_weights)))
            sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        loader_train = DataLoader(dataset_train, 
                                batch_size=config.batch_size,
                                shuffle=shuffle,
                                sampler=sampler,
                                pin_memory=config.pin_mem,
                                num_workers=config.num_cpu_workers
                                )
        loader_val = DataLoader(dataset_val,
                                batch_size=config.batch_size,
                                shuffle=False,
                                pin_memory=config.pin_mem,
                                num_workers=config.num_cpu_workers
                                )
        loader_test = DataLoader(dataset_test,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=config.pin_mem,
                                num_workers=config.num_cpu_workers
                                )
        
            
        if config.ensemble_loss == "CE":
            ense_loss = CrossEntropyLoss()
        elif config.ensemble_loss == "Focal":
            ense_loss = FocalLoss(gamma=config.gamma, alpha=config.alpha)
        else:
            raise ValueError("Unknown ensemble loss: %s"%config.ensemble_loss)

        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = None
        model = TrainingModel(fold_id, net, ense_loss, loader_train, loader_val, loader_test, config, optimizer, scheduler)

        if config.mode == ENSEMBLE_TRAIN:
            model.train()
            model.test()
        else:
            model.test()




