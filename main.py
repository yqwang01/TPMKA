import numpy as np
import argparse
from config import Config, PRETRAINING, FINE_TUNING, TESTING, HEATMAP
import os, json
import random
import gc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning", 'testing', 'heatmap'], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
    args = parser.parse_args()
    if args.mode == "pretraining": 
        mode = PRETRAINING 
    elif args.mode == "finetuning":
        mode = FINE_TUNING
    elif args.mode == "heatmap":
        mode = HEATMAP
    else:
        mode = TESTING

    config = Config(mode)
    os.makedirs(config.output_dir, exist_ok=True)
    json_path = os.path.join(config.output_dir, 'hyperparameter.json')
    if mode != TESTING and mode != HEATMAP:
        with open(json_path,'w') as f:
            f.write(json.dumps(vars(config), ensure_ascii=False, indent=4, separators=(',', ':')))
        
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    random.seed(config.seed)
    np.random.seed(config.seed)

    from data_loader import dataset_split, k_fold_cross_validation, AQUADataset, dataset_sampling_weights
    from torch.utils.data import DataLoader
    from main_funcs import TrainingModel
    from losses import GeneralizedSupervisedNTXenLoss, FocalLoss, NTXenLoss
    from torch.nn import CrossEntropyLoss
    from torch.utils.data import WeightedRandomSampler
    from models.densenet import densenet121
    from models.resnet import resnet50
    from models.pvt_v2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2
    from models.swin_transformer_v2 import swin_transformer_v2_base, swin_transformer_v2_small, swin_transformer_v2_tiny
    import torch

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    full_list, full_label, full_sen_attr, full_var, full_moda_indices = dataset_split( # type: ignore
                            config.train_csv_path, config.num_var, config.moda, config.var_idx, config.sen_attr_idx, config.site, avail_moda=config.avail_moda)    
    
    
    for fold_id in range(config.folds):
        print("\n-----------Fold: {}------------".format(fold_id))
        train_list, val_list, test_list, train_label, val_label, test_label, train_var, val_var, test_var, train_sen_attr, val_sen_attr, test_sen_attr = k_fold_cross_validation(
                                        full_list, full_label, full_sen_attr, full_var, config.data_dir, config.moda, full_moda_indices, config.multi_moda, fold_id, config.folds)
        if not config.use_var:
            train_var, val_var, test_var = None, None, None

        if config.sampling_mode is not None:
            train_weights = dataset_sampling_weights(config.sampling_mode, train_sen_attr, train_label, weight_power=config.weight_power)
            print(f"Using {config.sampling_mode} sampling strategy for training data.")
            print("Sample weights range: [{:.4f}, {:.4f}]".format(min(train_weights), max(train_weights)))
            sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        dataset_train = AQUADataset(config.data_dir, train_list, train_label, train_var, train_sen_attr,
                                    config.moda, config.moda_list, config.multi_moda, config.input_size, True, True)
        dataset_val = AQUADataset(config.data_dir, val_list, val_label, val_var, val_sen_attr,
                                  config.moda, config.moda_list, config.multi_moda, config.input_size, True, False)
        dataset_test = AQUADataset(config.data_dir, test_list, test_label, test_var, test_sen_attr,
                                   config.moda, config.moda_list, config.multi_moda, config.input_size, False, False)

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
        print("Train dataset size: {}, Val dataset size: {}, Test dataset size: {}".format(
            len(dataset_train), len(dataset_val), len(dataset_test)))
        
        if config.mode == PRETRAINING:
            if config.model == "DenseNet":
                net = densenet121(config.imagenet_pretrained, mode="encoder", use_var=config.use_var, 
                                in_channels=config.in_channel, num_var=config.num_var)
            elif config.model == "ResNet50":
                net = resnet50(config.imagenet_pretrained, mode="encoder", use_var=config.use_var, 
                            in_channels=config.in_channel, num_var=config.num_var)
            elif config.model == "PVT_v2_b0":
                net = pvt_v2_b0(config.imagenet_pretrained, mode="encoder", use_var=config.use_var, 
                                in_chans=config.in_channel, num_var=config.num_var)
            elif config.model == "PVT_v2_b1":
                net = pvt_v2_b1(config.imagenet_pretrained, mode="encoder", use_var=config.use_var, 
                                in_chans=config.in_channel, num_var=config.num_var)
            elif config.model == "PVT_v2_b2":
                net = pvt_v2_b2(config.imagenet_pretrained, mode="encoder", use_var=config.use_var, 
                                in_chans=config.in_channel, num_var=config.num_var)
            elif config.model == "SwinT_V2_base":
                net = swin_transformer_v2_base(config.imagenet_pretrained, mode="encoder", use_var=config.use_var,
                                            in_chans=config.in_channel, num_var=config.num_var)
            elif config.model == "SwinT_V2_small":
                net = swin_transformer_v2_small(config.imagenet_pretrained, mode="encoder", use_var=config.use_var,
                                            in_chans=config.in_channel, num_var=config.num_var)
            elif config.model == "SwinT_V2_tiny":
                net = swin_transformer_v2_tiny(config.imagenet_pretrained, mode="encoder", use_var=config.use_var,
                                            in_chans=config.in_channel, num_var=config.num_var)
            else:
                raise ValueError("Unknown model: %s"%config.model)
        else:
            if config.model == "DenseNet":
                net = densenet121(config.imagenet_pretrained, mode="classifier", use_var=config.use_var, 
                                num_classes=config.num_classes, in_channels=config.in_channel, num_var=config.num_var)
            elif config.model == "ResNet50":
                net = resnet50(config.imagenet_pretrained, mode="classifier", use_var=config.use_var, 
                            in_channels=config.in_channel, num_var=config.num_var, num_classes=config.num_classes)
            elif config.model == "PVT_v2_b0":
                net = pvt_v2_b0(config.imagenet_pretrained, mode="classifier", use_var=config.use_var, 
                                in_chans=config.in_channel, num_var=config.num_var, num_classes=config.num_classes)
            elif config.model == "PVT_v2_b1":
                net = pvt_v2_b1(config.imagenet_pretrained, mode="classifier", use_var=config.use_var, 
                                in_chans=config.in_channel, num_var=config.num_var, num_classes=config.num_classes)
            elif config.model == "PVT_v2_b2":
                net = pvt_v2_b2(config.imagenet_pretrained, mode="classifier", use_var=config.use_var, 
                                in_chans=config.in_channel, num_var=config.num_var, num_classes=config.num_classes)
            elif config.model == "SwinT_V2_base":
                net = swin_transformer_v2_base(config.imagenet_pretrained, mode="classifier", use_var=config.use_var,
                                            in_chans=config.in_channel, num_var=config.num_var, num_classes=config.num_classes)
            elif config.model == "SwinT_V2_small":
                net = swin_transformer_v2_small(config.imagenet_pretrained, mode="classifier", use_var=config.use_var,
                                            in_chans=config.in_channel, num_var=config.num_var, num_classes=config.num_classes)
            elif config.model == "SwinT_V2_tiny":
                net = swin_transformer_v2_tiny(config.imagenet_pretrained, mode="classifier", use_var=config.use_var,
                                            in_chans=config.in_channel, num_var=config.num_var, num_classes=config.num_classes)
            else:
                raise ValueError("Unknown model: %s"%config.model)
            
        if config.cont_loss == "Supervised":
            cont_loss = GeneralizedSupervisedNTXenLoss(temperature=config.temperature, 
                                                    kernel='rbf', sigma=config.sigma, return_logits=True) # type: ignore
        elif config.cont_loss == "Unsupervised":
            cont_loss = NTXenLoss(temperature=config.temperature, return_logits=True)
        if config.clas_loss == "CE":
            clas_loss = CrossEntropyLoss()
        elif config.clas_loss == "Focal":
            clas_loss = FocalLoss(gamma=config.gamma, alpha=config.alpha) # type: ignore

        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = None # ExponentialLR(optimizer, gamma=config.power)
        model = TrainingModel(fold_id, net, cont_loss, clas_loss, loader_train, loader_val, loader_test, config, optimizer, scheduler) # type: ignore

        if config.mode == PRETRAINING:
            model.pretraining()
        elif config.mode == FINE_TUNING:
            model.fine_tuning()
            model.test()
        elif config.mode == HEATMAP:
            model.heatmap()
        else:
            model.test()

        torch.cuda.empty_cache()
        gc.collect()


