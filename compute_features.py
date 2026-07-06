from config import Config, TESTING
import os, random
import numpy as np
from tqdm import tqdm
from utils import split_sample_path


def load_model(path, model):
    try:
        checkpoint = torch.load(path, map_location='cpu')
    except BaseException as e:
        raise ValueError('Impossible to load the checkpoint: %s' % str(e))
    try:
        if hasattr(checkpoint, "state_dict"):
            unexpected = model.load_state_dict(checkpoint.state_dict(), strict=False)
            print('Model loading info: {}'.format(unexpected))
        elif isinstance(checkpoint, dict):
            if "model" not in checkpoint:
                raise ValueError("Checkpoint dictionary does not contain a 'model' key.")
            unexpected = model.load_state_dict(checkpoint["model"], strict=False)
            print('Model loading info: {}'.format(unexpected))
        else:
            unexpected = model.load_state_dict(checkpoint)
            print('Model loading info: {}'.format(unexpected))
    except BaseException as e:
        raise ValueError('Error while loading the model\'s weights: %s' % str(e))

    return model


if __name__ == "__main__":


    config = Config(TESTING)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    random.seed(config.seed)
    np.random.seed(config.seed)

    import torch
    from data_loader import dataset_split_feature, AQUADataset
    from torch.utils.data import DataLoader
    from models.densenet import densenet121
    from models.resnet import resnet50
    from models.pvt_v2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2
    
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if config.cuda else "cpu")

    pat_list, test_list, test_label, test_sen_attr, test_var = dataset_split_feature(config.data_dir, config.train_csv_path, config.num_var, config.moda, # type: ignore
            var_idx=config.var_idx, sen_attr_idx=config.sen_attr_idx, for_json=False, avail_moda=config.avail_moda)
    if not config.use_var:
        test_var = None

    dataset_test = AQUADataset(config.data_dir, test_list, test_label, test_var, test_sen_attr, config.moda, config.moda_list, config.multi_moda, config.input_size, False, False)

    loader_test = DataLoader(dataset_test, 
                             batch_size=1,
                             shuffle=False,
                             pin_memory=config.pin_mem,
                             num_workers=config.num_cpu_workers
                             )
    
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
    else:
        raise ValueError("Unknown model: %s"%config.model)
    
    net = net.to(device)


    for fold_id in range(config.folds):
        net = load_model(os.path.join(config.output_dir, "MultiModaCL_Finetuned_Best_fold_{fold}.pth".format(fold=fold_id)), net)
        nb_batch = len(loader_test)
        pbar = tqdm(total=nb_batch, desc="Test", mininterval=5)

        with torch.no_grad():
            net.eval() # type: ignore
            for batch in loader_test:
                pbar.update()
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)
                name = batch['name'][0]
                sens_attr = batch['sens_attr'] if batch['sens_attr'][0] != -1 else None
                if config.use_var:
                    var = batch['var'].to(device)
                else:
                    var = None

                path_parts = split_sample_path(name)
                if len(path_parts) < 3:
                    raise ValueError("Expected sample path '<patient>/<modality>/<filename>', got: %s" % name)
                patient, modality, filename = path_parts[0], path_parts[1], path_parts[2]
                os.makedirs(os.path.join(config.feature_dir, 'fold_{}'.format(fold_id), patient, modality), exist_ok=True)

                feat, z, y = net(inputs, var) # type: ignore
                feat = feat.cpu().numpy()
                assert feat.shape == (1, 512)
            
                npz_file = os.path.join(config.feature_dir, 'fold_{}'.format(fold_id), patient, modality, filename)
                np.savez_compressed(npz_file, img=feat)
