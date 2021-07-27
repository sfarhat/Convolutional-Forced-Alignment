import torch
import torch.nn as nn
import torchaudio
from dataset_utils import create_timit_target, PhonemeTransformer, TextTransformer, get_phoneme_timestamp_information

class LibrispeechCollator(object):

    def __init__(self, n_mels, transformer):
        self.n_mels = n_mels
        self.transformer = transformer
        if not isinstance(self.transformer, TextTransformer):
            raise TypeError('Librispeech must use a TextTransformer, but it was given a ' + type(self.transformer))

    def __call__(self, batch):
        return self.preprocess_librispeech(batch)

    def preprocess_librispeech(self, dataset):
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

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels)

        for waveform, _, transcript, _, _, _ in dataset:
            # Transpose to move time dimension into proper padding position for later
            features = features_from_waveform(waveform, mel_spectrogram).transpose(0, 1)
            # Debug note: breakpoint here for expression: torch.isnan(features).any()

            # Adding 'spectrograms' of shape (time x features)
            inputs.append(features)
            input_lengths.append(features.shape[0]) # some examples online divide the shape by 2, why?

            # TODO: change this to 'to_int()' for consistency between tranformers?
            target = self.transformer.char_to_int(transcript.lower())
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

class TIMITCollator(object):

    def __init__(self, n_mels, transformer):
        self.n_mels = n_mels
        self.transformer = transformer
        if not isinstance(self.transformer, PhonemeTransformer):
            raise TypeError('TIMIT must use a PhonemeTransformer, but it was given a ' + type(self.transformer))

    def __call__(self, batch):
        return self.preprocess_timit(batch)

    def preprocess_timit(self, batch):

        inputs = [] 
        input_lengths = []
        targets = [] 
        target_lengths = []

        # Put this here instead of in features_from_waveform() becuase its parameters needed for transcript creation
        spectrogram_generator = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels)

        for sample in batch:
            waveform, phonemes = sample['audio'], sample['phonemes']
            features = features_from_waveform(waveform, spectrogram_generator).transpose(0, 1)
            inputs.append(features)

            input_lengths.append(features.shape[0])

            target_with_duration, target_without_duration = create_timit_target(phonemes, features.shape[0], spectrogram_generator)
            converted_target = self.transformer.phone_to_int(target_with_duration)
            targets.append(converted_target)
            
            target_lengths.append(len(converted_target))

        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True).unsqueeze(1).transpose(2, 3) 

        targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.transformer.blank_idx)

        # Only returning input_lengths to keep training code general and clean, improve when you can...
        return (inputs, input_lengths, targets, target_lengths)

class TIMITAlignmentCollator(object):

    def __init__(self, n_mels, transformer):
        self.n_mels = n_mels
        self.transformer = transformer
        if not isinstance(self.transformer, PhonemeTransformer):
            raise TypeError('TIMIT must use a PhonemeTransformer, but it was given a ' + type(self.transformer))

    def __call__(self, batch):
        return self.preprocess_timit(batch)

    def preprocess_timit(self, batch):

        inputs = [] 
        samples = []

        # Put this here instead of in features_from_waveform() becuase its parameters needed for transcript creation
        spectrogram_generator = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels)

        for sample in batch:
            waveform, phonemes = sample['audio'], sample['phonemes']
            features = features_from_waveform(waveform, spectrogram_generator).transpose(0, 1)
            inputs.append(features)
            samples.append(sample)

        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True).unsqueeze(1).transpose(2, 3) 

        return inputs, samples, spectrogram_generator

def preprocess_single_waveform(waveform, n_mels):

    spectrogram_generator = torchaudio.transforms.MelSpectrogram(n_mels=n_mels)

    if len(waveform.shape) > 1:
        # dual channel edge case
        waveform = waveform[0]

    features = features_from_waveform(waveform, spectrogram_generator)

    # phonetic_information = get_phoneme_timestamp_information(timit_sample_path)

    # target_with_duration, target_without_duration = create_timit_target(phonetic_information, features.shape[1], spectrogram_generator)

    # Returns features of shape channel x features x time
    return features.unsqueeze(0), spectrogram_generator

def features_from_waveform(waveform, spectrogram_generator):
    """Generate features from an audio waveform.

    Raw audio is transformed into 40-dimensional log mel-filter-bank coefficients with deltas and delta-deltas, which reasults in 120 dimensional features.
    Each dimension is normalized to have zero mean and unit variance over the training set.
    Unlike the inspired Zhang model, we do not use an energy term.

    Args:
        waveform (Tensor): [shape: (channel x time)] Time series data representing spoken input
        n_mels (int): Number of desired mels 

    Returns:
        input_features_normalized (Tensor): [shape: (features x time)] log-mel spectrogram, delta, and delta-delta features
    """

    # Waveform has channel first dimension, gives shape (1, ...) which causes shape problems when stacking features
    data = waveform.squeeze(dim=0)

    # Grab desired features
    # Takes in audio of shape (..., time) returns (..., n_mels, new_time) where n_mels defaults to 128
    log_offset = 1e-6
    # adding offset because log(0) is nan, led to inputs becoming nan -> nan ouputs -> nan loss
    log_mel_spectrogram = torch.log(spectrogram_generator(data) + log_offset)
    # Takes in audio of shape (..., time) returns (..., n_mfcc, new_time) where n_mfcc defaults to 40
    # mfcc_features = torchaudio.transforms.MFCC(n_mfcc=n_mfcc, log_mels=True)(data) 
    deltas = torchaudio.functional.compute_deltas(log_mel_spectrogram)
    delta_deltas = torchaudio.functional.compute_deltas(deltas)

    # Stack feature Tensors together into (n_mfcc*3, new_time)
    input_features = torch.cat((log_mel_spectrogram, deltas, delta_deltas), 0)

    # Normalize (0 mean, 1 std) features along time dimension
    input_features_normalized = nn.LayerNorm(input_features.shape[1], elementwise_affine=False)(input_features)

    return input_features_normalized