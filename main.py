import torch
import torch.nn as nn
import torchaudio
# from torch.utils.tensorboard import SummaryWriter
from config import hparams, DATASET_DIR
from preprocess import LibrispeechCollator
from utils import weights_init_unif, load_from_checkpoint, save_checkpoint
from model import ASR_1
from training import train
from inference import test

def main():

    torch.manual_seed(9)
    
    # Without this, torchaudio 0.7 uses deprecated "sox" backend which only supports 16-bit integers
    torchaudio.set_audio_backend("sox_io")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    train_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_DIR, url="train-clean-100", download=True)
    # dev_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_DIR, url="dev-clean", download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_DIR, url="test-clean", download=True)

    collator = LibrispeechCollator(hparams["n_mels"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True, collate_fn=collator, pin_memory=use_cuda)
    # dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=hparams["batch_size"], shuffle=True, collate_fn=preprocess, pin_memory=use_cuda)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False, collate_fn=collator, pin_memory=use_cuda)

    # 1 channel input from feature spectrogram, 29 dim output from char_map + blank for CTC, 120 features
    net = ASR_1(in_dim=1, num_classes=29, num_features=120, activation=hparams["activation"], dropout=0.3)
    net.to(device)
    weights_init_unif(net, hparams["weights_init_a"], hparams["weights_init_b"])

    # ADAM loss w/ lr=10e-4, batch size 20, initial weights initialized uniformly from [-0.05, 0.05], dropout w/ p=0.3 used in all layers except in and out
    # for fine tuning: SGD w/ lr 10e-5, l2 penalty w/ coeff=1e-5

    criterion = nn.CTCLoss().to(device)
    # TODO: consider using ADAMW for LR decay
    optimizer = torch.optim.Adam(net.parameters(), lr=hparams["ADAM_lr"])
    finetune_optimizer = torch.optim.SGD(net.parameters(), lr=hparams["SGD_lr"], weight_decay=hparams["SGD_l2_penalty"])

    # writer = SummaryWriter()
    # writer.add_graph(net)

    if hparams["start_epoch"] > 0: 
        net, optimizer = load_from_checkpoint(net, optimizer, hparams["activation"], hparams["ADAM_lr"], hparams["start_epoch"], device)
        start_epoch = hparams["start_epoch"]
    else:
        start_epoch = 0

    if hparams["train"]:
        for epoch in range(start_epoch, start_epoch + hparams["epochs"]):
            train(net, train_loader, criterion, optimizer, epoch, device)
            save_checkpoint(net, optimizer, epoch, hparams["activation"], hparams["ADAM_lr"])
    else: 
        test(net, test_loader, criterion, device)

    # TODO: Where/when to do dev set?

    

if __name__ == "__main__":
    main()
