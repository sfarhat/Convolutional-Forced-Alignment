DATASET_DIR = "/mnt/d/Datasets"
CHECKPOINT_DIR_NAME = "checkpoints"

hparams = {
    "ADAM_lr": 10e-5,
    "batch_size": 3,
    "SGD_lr": 10e-5,
    "SGD_l2_penalty": 1e-5,
    "weights_init_a": -0.05,
    "weights_init_b": 0.05,
    "epochs": 50,
    "activation": "prelu",
    "n_mels": 40,
    "start_epoch": 50,
    "train": False,
    "dataset": 'TIMIT'
}
