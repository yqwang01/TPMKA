import os

PRETRAINING = 0
FINE_TUNING = 1
TESTING = 2
ENSEMBLE_TRAIN = 3
ENSEMBLE_TEST = 4
HEATMAP = 5


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def env_path(name, default):
    return os.path.abspath(os.getenv(name, default))


class Config:

    def __init__(self, mode):
        assert mode in {PRETRAINING, FINE_TUNING, TESTING, HEATMAP, ENSEMBLE_TRAIN, ENSEMBLE_TEST}, "Unknown mode: %i"%mode

        # System settings
        self.gpu = os.getenv("CUDA_VISIBLE_DEVICES", "0")
        self.cuda = True
        self.pin_mem = True
        self.num_cpu_workers = int(os.getenv("AQUA_NUM_WORKERS", "20"))
        self.seed = 0
        # Dataset settings
        self.mode = mode
        self.data_dir = env_path("AQUA_DATA_DIR", os.path.join(PROJECT_DIR, "data", "AQUA_Dataset"))
        self.train_csv_path = env_path("AQUA_TRAIN_CSV", os.path.join(PROJECT_DIR, "data", "data_list.csv"))
        self.folds = 5
        self.input_size = (256, 256)
        self.in_channel = 3
        self.num_var = 7 # 7, 13, 24
        self.var_idx = 0
        self.sen_attr_idx = 13
        self.avail_moda = 5
        self.use_var = True
        self.imagenet_pretrained = True
        self.moda = 'B' # 'B', 'ScS', 'W', 'O', 'S'
        self.moda_list = ['B', 'ScS', 'W']
        self.multi_moda = True if self.mode == PRETRAINING else False
        self.site = None # 'India', 'US', None
        self.sampling_mode = None # "balanced", "group", "label", None
        self.weight_power = 1.0 # 1.0, 2.0, None
        self.equity_alpha = 1.0 # 1.0, 2.0, None
        self.metric = "acc" # "class_attr_scaled_f1", "f1", "bacc", "acc"
        # Loss settings
        self.power_lr = 0.9
        self.cont_loss = "Unsupervised"
        self.clas_loss = "CE"
        self.ensemble_loss = "CE"
        self.temperature = 0.1
        self.sigma = 1.0
        
        if self.mode == PRETRAINING:
            self.batch_size = 32
            self.nb_epochs = 150
            # Optimizer
            self.lr = 3e-5
            self.warm_epochs = 15
            self.weight_decay = 1e-3
            self.model = "PVT_v2_b1" # "SwinT_V2_tiny", "PVT_v2_b1"
            self.output_dir = env_path(
                "AQUA_OUTPUT_DIR",
                os.path.join(PROJECT_DIR, "snapshots", "pretraining_PVTv2b1"),
            )

        elif self.mode == FINE_TUNING or self.mode == TESTING or self.mode == HEATMAP:
            self.batch_size = 32
            self.nb_epochs = 100
            self.num_classes = 2
            # Optimizer
            self.lr = 1e-5
            self.warm_epochs = 15
            self.weight_decay = 1e-3
            self.lam_cls = 0.9 # 0.9, 1
            self.gamma = 2
            self.alpha = 1
            self.model = "PVT_v2_b1" # "SwinT_V2_tiny", "PVT_v2_b1"
            self.output_dir = env_path(
                "AQUA_OUTPUT_DIR",
                os.path.join(PROJECT_DIR, "snapshots", "finetuning_PVTv2b1"),
            )
            self.feature_dir = env_path(
                "AQUA_FEATURE_DIR",
                os.path.join(PROJECT_DIR, "features", "finetuning_PVTv2b1"),
            )
            self.out_file_path = os.path.join(self.output_dir, "output_test_" + self.site if self.site is not None else "output_test")
            self.pretrained_path = os.getenv("AQUA_PRETRAINED_PATH")
                        
        else:
            self.ense_model = "Linear" # "Linear", "Transformer", "Attention"
            self.batch_size = 32
            self.warm_epochs = 20
            self.nb_epochs = 50
            self.num_classes = 2
            # Optimizer
            self.lr = 1e-5
            self.weight_decay = 1e-3
            self.feature_dir = env_path(
                "AQUA_FEATURE_DIR",
                os.path.join(PROJECT_DIR, "features", "finetuning_PVTv2b1"),
            )
            self.output_dir = env_path(
                "AQUA_OUTPUT_DIR",
                os.path.join(PROJECT_DIR, "snapshots", "ensemble_PVTv2b1"),
            )
            self.out_file_path = os.path.join(self.output_dir, "output_test_" + self.site if self.site is not None else "output_test")
