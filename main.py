import torch
import torch.nn as nn
import torchaudio
import os
from config import hparams, DATASET_DIR
from preprocess import LibrispeechCollator, TIMITCollator
from utils import weights_init_unif, load_from_checkpoint, save_checkpoint, TextTransformer
from model import ASR_1 
from training import train
from inference import test
from timit_utils import TIMITDataset, PhonemeTransformer
from loss import CollapsedCTCLoss

def main():

    torch.manual_seed(9)
    
    # Without this, torchaudio 0.7 uses deprecated "sox" backend which only supports 16-bit integers
    torchaudio.set_audio_backend("sox_io")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    if hparams["dataset"] == 'Librispeech':
        train_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_DIR, url="train-clean-100", download=True)
        dev_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_DIR, url="dev-clean", download=True)
        test_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_DIR, url="test-clean", download=True)
        transformer = TextTransformer()
        collator = LibrispeechCollator(hparams["n_mels"], transformer)
        # 1 channel input from feature spectrogram, 29 dim output from char_map + blank for CTC, 120 (40 mels + deltas + delta-deltas) features
        net = ASR_1(in_dim=1, num_classes=len(transformer.char_map)+1, num_features=hparams["n_mels"]*3, activation=hparams["activation"], dropout=0.3)
        criterion = nn.CTCLoss().to(device)
    elif hparams["dataset"] == 'TIMIT':
        train_dataset = TIMITDataset(os.path.join(DATASET_DIR, 'timit', 'data', 'TRAIN'))
        test_dataset = TIMITDataset(os.path.join(DATASET_DIR, 'timit', 'data', 'TEST'))
        transformer = PhonemeTransformer()
        collator = TIMITCollator(hparams['n_mels'], transformer)
        net = ASR_1(in_dim=1, num_classes=len(transformer.phon), num_features=hparams["n_mels"]*3, activation=hparams["activation"], dropout=0.3)
        criterion = CollapsedCTCLoss()
    else:
        raise Exception("Not a valid dataset. Please choose between 'Librispeech' or 'TIMIT'.")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True, collate_fn=collator, pin_memory=use_cuda)
    # dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=hparams["batch_size"], shuffle=True, collate_fn=collator, pin_memory=use_cuda)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False, collate_fn=collator, pin_memory=use_cuda)

    net.to(device)
    weights_init_unif(net, hparams["weights_init_a"], hparams["weights_init_b"])

    # Save last layer activations for GRAD-CAM, only needed for test-time
    activations = []
    def get_activation(model, input, output):
        """Forward hook that returns output at chosen layer of network"""
        activations.append(output.detach())

    gradients = []
    def get_gradients(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # ADAM loss w/ lr=10e-4, batch size 20, initial weights initialized uniformly from [-0.05, 0.05], dropout w/ p=0.3 used in all layers except in and out
    # for fine tuning: SGD w/ lr 10e-5, l2 penalty w/ coeff=1e-5

    # TODO: consider using ADAMW for LR decay
    optimizer = torch.optim.Adam(net.parameters(), lr=hparams["ADAM_lr"])
    finetune_optimizer = torch.optim.SGD(net.parameters(), lr=hparams["SGD_lr"], weight_decay=hparams["SGD_l2_penalty"])

    if hparams["start_epoch"] > 0: 
        net, optimizer = load_from_checkpoint(net, optimizer, hparams["activation"], hparams["ADAM_lr"], hparams["start_epoch"], device, hparams["dataset"])
        start_epoch = hparams["start_epoch"]
    else:
        start_epoch = 0

    if hparams["train"]:
        for epoch in range(start_epoch, start_epoch + hparams["epochs"]):
            train(net, train_loader, criterion, optimizer, epoch, device)
            save_checkpoint(net, optimizer, epoch, hparams["activation"], hparams["ADAM_lr"], hparams["dataset"])
    else: 
        net.cnn_layers[-1].register_forward_hook(get_activation)
        net.cnn_layers[-1].register_backward_hook(get_gradients)
        test(net, test_loader, criterion, device, transformer)

if __name__ == "__main__":
    main()
