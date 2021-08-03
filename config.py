DATASET_DIR = '/mnt/d/Datasets'
CHECKPOINT_DIR_NAME = 'checkpoints'

hparams = {
    'ADAM_lr': 10e-5,
    'batch_size': 3,
    'SGD_lr': 10e-5,
    'SGD_l2_penalty': 1e-5,
    'weights_init_a': -0.05,
    'weights_init_b': 0.05,
    'epochs': 20,
    'activation': 'prelu',
    'n_mels': 40,
    'start_epoch': 75,
    'mode': 'align',
    'dataset': 'TIMIT',
    'sample_path': 'audio.wav',
    'sample_transcript': 'lyrics.txt',
    'timit_sample_path': '/mnt/d/Datasets/timit/data/TRAIN/DR4/MESG0/SX72' ,
    'model': 'zhang'
}
