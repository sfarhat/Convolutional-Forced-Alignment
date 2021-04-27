import torch
import torch.nn as nn
import torchaudio
from utils import text_to_target, char_map

def features_from_waveform(waveform, n_mfcc=40):
    """Generate features from an audio waveform.

    Raw audio is transformed into 40-dimensional log mel-filter-bank (plus energy term) coefficients with deltas and delta-deltas, which reasults in 123 dimensional features.
    Each dimension is normalized to have zero mean and unit variance over the training set.
    Basically this is just MFCC but without taking DCT at the end, but for the sake of cleanliness, I'll stick with MFCC for now. 
    Also, I don't know what they mean by "energy term" (aren't the coefficients already energy terms?) so I'm omitting that.

    Args:
        waveform (Tensor): [shape: (channel x time)] Time series data representing spoken input
        n_mfcc (int): Number of desired MFC coefficients

    Returns:
        input_features_normalized (Tensor): [shape: (features x time)] "Spectrogram" of MFCC, delta, and delta-delta features
    """

    # Waveform has channel first dimension, gives shape (1, ...) which causes shape problems when stacking features
    data = waveform.squeeze(dim=0)

    # Grab desired features
    # Takes in audio of shape (..., time) returns (..., n_mfcc, new_time) where n_mfcc defaults to 40
    mfcc_features = torchaudio.transforms.MFCC(n_mfcc=n_mfcc, log_mels=True)(data) 
    deltas = torchaudio.functional.compute_deltas(mfcc_features)
    delta_deltas = torchaudio.functional.compute_deltas(deltas)

    # Stack feature Tensors together into (n_mfcc*3, new_time)
    input_features = torch.cat((mfcc_features, deltas, delta_deltas), 0)

    # Normalize (0 mean, 1 std) features along time dimension
    input_features_normalized = nn.LayerNorm(input_features.shape[1], elementwise_affine=False)(input_features)

    return input_features_normalized

def preprocess_librispeech(dataset, n_mfcc):

    """
    Preprocesses LIBRISPEECH dataset for CTC decoding.

    Each datapoint in LIBRISPEECH dataset is a Tuple of (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)

    1. Convert waveforms to input features
    2. Convert transcripts to output class indices
    3. Get input sequence (feature) lengths before padding
    4. Get target lengths before padding
    5. Pad inputs and targets for consistent sizes

    This is fed into the DataLoader as the collate_fn, so keep in mind batch dimension.

    Args:
        dataset (list[tuple]): [shape: (batch)] Batched Librispeech samples 

    Returns:
        inputs (Tensor): [shape: (batch x channel x features x time)] Input 'spectrograms' to network, padded along time dimension
        input_lengths (list[int]): [shape: (batch)] Python list of respective lengths (time) of inputs in batch 
        targets (Tensor): [shape: (batch x transcript)] Transcript of sound converted to integers instead of characters, padded like inputs
        target_lengths (list[int]): [shape: (batch)] Python list of respective lenghts of transcripts in batch
    """

    # These are necessary for CTCLoss
    inputs = [] 
    targets = [] 
    # For these, CTCLoss expects them to be pre-padding
    input_lengths = [] 
    target_lengths = [] 

    for waveform, _, transcript, _, _, _ in dataset:
        # Transpose to move time dimension into proper padding position for later
        features = features_from_waveform(waveform, n_mfcc=n_mfcc).transpose(0, 1)
        # Adding 'spectrograms' of shape (time x features)
        inputs.append(features)
        input_lengths.append(features.shape[0])

        target = text_to_target(transcript.lower(), char_map)
        targets.append(target)
        target_lengths.append(len(target))

    # Each 'spectrogram' has different lengths in time dimension, so we need to pad them to be uniform within batch
    # pad_sequence requires padding-needing dimension to be 0th dimension and all trailing dims to be the same 
    # This transformation doesn't affect the model architecture since we are only flattening/connecting the feature dimension, not the time one (nn.Linear allows for this flexibility)
    # Need to add back (unsqueeze) channel dimension and undo 'padding ordering' to get shape (batch, channel, features, time)
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True).unsqueeze(1).transpose(2, 3) 

    # This will pad with 0, which represents the blank, but this shouldn't be a problem since we're providing the target_length of the unpadded target
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return (inputs, input_lengths, targets, target_lengths)

