import torch
import torch.nn as nn
import torchaudio
import os
from config import hparams, DATASET_DIR
from preprocess import LibrispeechCollator, TIMITAlignmentCollator, TIMITCollator, preprocess_single_waveform
from utils import weights_init_unif, load_from_checkpoint, save_checkpoint
from model import ASR_1 
from training import train
from inference import alignment_error, show_activation_map, force_align, test_accuracy, test_alignment
from dataset_utils import TIMITDataset, PhonemeTransformer, TextTransformer, get_word_timestamp_information
from loss import SequentialNLLLoss
from pronouncing import get_lyrics

def main():

    torch.manual_seed(9)
    
    # Without this, torchaudio 0.7 uses deprecated 'sox' backend which only supports 16-bit integers
    torchaudio.set_audio_backend('sox_io')

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    if hparams['dataset'] == 'Librispeech':

        train_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_DIR, url='train-clean-100', download=True)
        dev_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_DIR, url='dev-clean', download=True)
        test_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_DIR, url='test-clean', download=True)
        transformer = TextTransformer()
        collator = LibrispeechCollator(hparams['n_mels'], transformer)
        # 1 channel input from feature spectrogram, 29 dim output from char_map + blank for CTC, 120 (40 mels + deltas + delta-deltas) features
        # net = ASR_1(in_dim=1, num_classes=len(transformer.char_map)+1, num_features=hparams['n_mels']*3, activation=hparams['activation'], dropout=0.3)
        output_dim = len(transformer.char_map)+1
        criterion = nn.CTCLoss().to(device)

    elif hparams['dataset'] == 'TIMIT':
        
        train_dataset = TIMITDataset(os.path.join(DATASET_DIR, 'timit', 'data', 'TRAIN'))
        test_dataset = TIMITDataset(os.path.join(DATASET_DIR, 'timit', 'data', 'TEST'))
        transformer = PhonemeTransformer()
        collator = TIMITCollator(hparams['n_mels'], transformer)
        # net = ASR_1(in_dim=1, num_classes=len(transformer.phon), num_features=hparams['n_mels']*3, activation=hparams['activation'], dropout=0.3)
        output_dim = len(transformer.phon)
        criterion = SequentialNLLLoss()

    else:

        raise Exception('Not a valid dataset. Please choose between \'Librispeech\' or \'TIMIT\'.')

    if hparams['model'] == 'zhang':
        net = ASR_1(in_dim=1, num_classes=output_dim, num_features=hparams['n_mels']*3, activation=hparams['activation'], dropout=0.3)
    else:
        raise Exception('Not a valid model architecture. Please choose between \'zhang\'.')


    # dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=hparams['batch_size'], shuffle=True, collate_fn=collator, pin_memory=use_cuda)

    net.to(device)
    weights_init_unif(net, hparams['weights_init_a'], hparams['weights_init_b'])

    # ADAM loss w/ lr=10e-4, batch size 20, initial weights initialized uniformly from [-0.05, 0.05], dropout w/ p=0.3 used in all layers except in and out
    # for fine tuning: SGD w/ lr 10e-5, l2 penalty w/ coeff=1e-5

    optimizer = torch.optim.Adam(net.parameters(), lr=hparams['ADAM_lr'])
    finetune_optimizer = torch.optim.SGD(net.parameters(), lr=hparams['SGD_lr'], weight_decay=hparams['SGD_l2_penalty'])

    if hparams['start_epoch'] > 0: 

        net, optimizer = load_from_checkpoint(net, optimizer, hparams['activation'], hparams['ADAM_lr'], hparams['start_epoch'], device, hparams['dataset'])
        start_epoch = hparams['start_epoch']

    else:

        start_epoch = 0

    if hparams['mode'] == 'train':

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, collate_fn=collator, pin_memory=use_cuda)
        # test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collator, pin_memory=use_cuda)
        for epoch in range(start_epoch, start_epoch + hparams['epochs']):
            train(net, train_loader, criterion, optimizer, epoch, device)
            save_checkpoint(net, optimizer, epoch, hparams['activation'], hparams['ADAM_lr'], hparams['dataset'])
            # test_accuracy(net, test_loader, criterion, device, transformer)

    elif hparams['mode'] == 'test': 

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collator, pin_memory=use_cuda)
        test_accuracy(net, test_loader, criterion, device, transformer)

    elif hparams['mode'] == 'cam':

        waveform, _ = torchaudio.load(hparams['sample_path'])
        input, _ = preprocess_single_waveform(waveform, hparams['n_mels'])
        show_activation_map(net, device, input, [1, 2, 10])

    elif hparams['mode'] == 'align':

        if hparams['timit_sample_path'] != '':
            timit_sample = True
        else:
            timit_sample = False

        if not timit_sample:
            audio_path = hparams['sample_path']
            transcript_path = hparams['sample_transcript']
        else:
            audio_path = hparams['timit_sample_path'] + '.WAV'
            transcript_path = hparams['timit_sample_path'] + '.TXT'

        waveform, _ = torchaudio.load(audio_path)
        transcript = get_lyrics(transcript_path, timit=timit_sample)
        # plt.plot(waveform[0])
        # plt.savefig('waveform.png')
        # plt.show()
        input, spectrogram_generator = preprocess_single_waveform(waveform, hparams['n_mels'])
        # plt.imshow(input.permute(1, 2, 0))
        # plt.savefig('spectrogram.png')
        # plt.show()
        guessed_alignment = force_align(net, transformer, device, input, spectrogram_generator, transcript)
        print(guessed_alignment)
        if timit_sample:
            true_alignment = get_word_timestamp_information(hparams['timit_sample_path'])
            ae = alignment_error(guessed_alignment, true_alignment)
            print('Alignment Error: ', ae)

    elif hparams['mode'] == 'test-align':

        alignment_collator = TIMITAlignmentCollator(hparams['n_mels'], transformer)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=alignment_collator, pin_memory=use_cuda)
        train_aer = test_alignment(net, train_loader, device, transformer)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=alignment_collator, pin_memory=use_cuda)
        test_aer = test_alignment(net, test_loader, device, transformer)
        aer = (train_aer + test_aer) / 2
        print('Overall AER: {}%'.format(aer))

    else:

        raise Exception('Not a valid mode. Please choose between \'train\', \'test\', \'cam\', \'align\', or \'test-align\'.')

if __name__ == '__main__':
    main()
