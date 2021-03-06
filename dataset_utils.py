import torch
from torch.utils.data import Dataset
import torchaudio
import os

SPACE_TOKEN = '<sp>'

class PhonemeTransformer:

    def __init__(self):

        # Map from 61 phonemes to 39 as proposed in (Lee & Hon, 1989)
        self.collapse_phon_map = { 
            'aa': 'aa',	           
            'ae': 'ae', 
            'ah': 'ah',	
            'ao': 'aa',
            'aw': 'aw',	
            'ax': 'ah',	
            'ax-h': 'ah',
            'axr': 'er',
            'ay': 'ay',	
            'b': 'b', 	
            'bcl': 'sil',
            'ch': 'ch',	
            'd': 'd', 	
            'dcl': 'sil',
            'dh': 'dh',	
            'dx': 'dx',	
            'eh': 'eh',	
            'el': 'l',	
            'em': 'm',	
            'en': 'n',	
            'eng': 'ng',
            'epi': 'sil',
            'er': 'er',
            'ey': 'ey',	
            'f': 'f',	 
            'g': 'g', 
            'gcl': 'sil',
            'h#': 'sil',	
            'hh': 'hh',	
            'hv': 'hh',	
            'ih': 'ih',	
            'ix': 'ih',	
            'iy': 'iy',	
            'jh': 'jh',	
            'k': 'k',	 
            'kcl': 'sil',
            'l': 'l', 	
            'm': 'm', 
            'n': 'n',	
            'ng': 'ng',	
            'nx': 'n',	
            'ow': 'ow',	
            'oy': 'oy',	
            'p': 'p', 	
            'pau': 'sil',
            'pcl': 'sil',
            'q': None,
            'r': 'r', 	
            's': 's', 	
            'sh': 'sh',	
            't': 't', 	
            'tcl': 'sil',
            'th': 'th',	
            'uh': 'uh',	
            'uw': 'uw',	
            'ux': 'uw',	
            'v': 'v', 	
            'w': 'w', 	
            'y': 'y', 	
            'z': 'z', 	
            'zh': 'sh',
            SPACE_TOKEN: SPACE_TOKEN
        } 

        # if self.collapse:
        #     # 39 collapsed phonemes
            # self.phon = ['aa', 'ae', 'ah', 'aw', 'er', 'ay', 'b', 'sil', 'ch', 'd', 'dh', 'dx',
            #             'eh', 'l', 'm', 'n', 'ng', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh',
            #             'k', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v',
            #             'w', 'y', 'z']
        #     self.phon_map = {self.phon[i]: i for i in range(len(self.phon))}
        #     self.idx_map = {i : self.phon[i] for i in range(len(self.phon))}
        # else:

        # Full 61 phonemes
        self.phon = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
                    'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey',
                    'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy',
                    'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p',
                    'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw',
                    'ux', 'v', 'w', 'y', 'z', 'zh']
        self.phon_map = {self.phon[i]: i for i in range(len(self.phon))}
        self.idx_map = {i : self.phon[i] for i in range(len(self.phon))}

        # ARPABET is missing 'sil' and 'dx'
        arpabet = ['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh',
                    'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k',
                    'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't',
                    'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']

    def phone_to_int(self, phonemes):
        """Converts list of phonemes to integer Tensor"""

        target = []

        for p in phonemes:
            target.append(self.phon_map[p])

        return torch.Tensor(target)

    def target_to_text(self, target):
        """Converts target list of integers to phoneme transcripts while collapsing 61 phones to 39"""

        transcript = []

        for idx in target:
            transcript.append(self.idx_map[int(idx)])

        transcript = self.map_to_39(transcript)

        return transcript

    def map_to_39(self, phonemes):
        """Collapses list of phonemes into 39 test ones for TIMIT"""

        collapsed = []
        for p in phonemes:
            if p == 'q':
                continue
            collapsed.append(self.collapse_phon_map[p])

        return collapsed

    @property
    def blank_idx(self):
        """Used for pad_sequence in preprocessing"""

        return self.phon_map['h#']

class TIMITDataset(Dataset):

    def __init__(self, path):
        self.prefix_paths = self.create_timit_paths(path)

    def __len__(self):
        return len(self.prefix_paths)

    def __getitem__(self, idx):
        """
        Generates samples on-the-fly. Each sample contains the audio waveform, phonemes, words, and transcript. 
        The phonemes and words are contain the start/end times as well as the respective phoneme/word in that duration. 
        So, the following dictionary is returned:
        
        {'audio': ..., 
            'phonemes': {'start': ..., 'end': ..., 'phoneme': ...}
            'words': {'start': ..., 'end': ..., 'word': ...}
            'transcript': ...}
        }
        """

        sample_path = self.prefix_paths[idx]

        sample = {'audio': [], 'phonemes': [], 'words': [], 'transcript': ''}
        
        wavpath = sample_path + '.WAV'
        waveform, sr = torchaudio.load(wavpath)
        sample['audio'] = waveform
        sample['words'] = get_word_timestamp_information(sample_path)
        sample['transcript'] = get_transcript(sample_path)
        sample['phonemes'] = get_phoneme_timestamp_information(sample_path)

        return sample

    def create_timit_paths(self, path):
        """
        Given a path to a TIMIT dataset (train/test), this will generate a list of paths to each group of files referring to one sample.
        Paths of the form timit/data/(TRAIN/TEST)/dr/speaker_id/sample_code, e.g. timit/data/TRAIN/DR1/FCJF0/SI648. This can be used so that
        the Dataset can generate samples on-the-fly by grabbing the apporpriate data from the files beginning with the prefix: prefix.WAV, 
        prefix.PHN, prefix.WRD, and prefix.TXT.

        Args:
            path (String): Path to the desired TIMIT dataset (train/test) 
        """
        
        paths = set()
        for root, dirs, files in os.walk(path):
            for fname in files:
                sample_id = fname.split('.')[0]
                if 'SA' in sample_id:
                    continue
                fpath = os.path.join(root, sample_id)
                paths.add(fpath)
                
        return list(paths)

def get_word_timestamp_information(sample_path):

    wrdpath = sample_path + '.WRD'
    all_word_details = []

    with open(wrdpath) as f:
        for line in f.read().splitlines():
            word_details = {}
            # start_index | end_index | word
            word_details['start'], word_details['end'], word_details['word'] = line.split(' ')
            word_details['start'] = int(word_details['start']) / 16500
            word_details['end'] = int(word_details['end']) / 16500
            all_word_details.append(word_details)

    return all_word_details

def get_transcript(sample_path):

    txtpath = sample_path + '.TXT'
    with open(txtpath) as f:
        for line in f.read().splitlines():
            # start_index | end_index | transcript
            # transcript = ' '.join(line.split(' ')[2:])
            transcript = line.split()[2:]
            return transcript

def get_phoneme_timestamp_information(sample_path):

    phnpath = sample_path + '.PHN'
    all_phon_details = []

    with open(phnpath) as f:
        for line in f.read().splitlines():
            phonetic_details = {}
            # start_index | end_index | phoneme
            phonetic_details['start'], phonetic_details['end'], phonetic_details['phoneme'] = line.split(' ')
            phonetic_details['start'] = int(phonetic_details['start'])
            phonetic_details['end'] = int(phonetic_details['end'])
            all_phon_details.append(phonetic_details)

    return all_phon_details

def create_timit_target(phonemes, spectrogram_len, spectrogram_generator):
    """Creates target transcript given phonemes and respective durations.
       Example: 'she' w/ 5 timesteps, 'sh' from [0-2], 'ix' from [3-4] -> ['sh', 'sh', 'sh', 'ix', 'ix']

    Args:
        phonemes ([type]): [description]
        transcript_len (int): Length of desired transcript, should match time dimension of input 
        mel_spectrogram ([type]): [description]

    Returns:
        list: Target transcript of length transcript_len
    """
    target_with_duration = []
    target = []

    for phoneme_details in phonemes:
        phon_start = waveform_time_to_spec_time(phoneme_details['start'], spectrogram_len, spectrogram_generator)
        phon_end = waveform_time_to_spec_time(phoneme_details['end'], spectrogram_len, spectrogram_generator)
        phoneme = phoneme_details['phoneme']

        target_with_duration.extend([phoneme] * (phon_end - phon_start))
        target.append(phoneme)

    return target_with_duration, target

def waveform_time_to_spec_time(t, spectrogram_len, spectrogram_generator):
    """Converts time in waveform space to time in spectrogram space given parameters of STFT"""

    hop_length, window_length = spectrogram_generator.hop_length, spectrogram_generator.win_length
    for hop in range(spectrogram_len):
        if t <= hop * hop_length + window_length / 2 and t >= hop * hop_length - window_length / 2:
            # As a convention, we will use the first hop that covers t, even though multiple may cover it as well
            return hop

def spec_time_to_waveform_time(tau, spectrogram_generator):
    # Use parameters to find range of t that each tau corresponds to, use middle value (heuristic)

    hop_length = spectrogram_generator.hop_length
    return tau * hop_length

class TextTransformer:
    """Handles all transformations bewteen text strings and integer equivalents"""

    def __init__(self):

        # index 0 is reserved for blank character in CTC
        self.char_map_str = """
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        ' 28
        """
        self.char_map, self.idx_map = self.create_char_map() 

    def create_char_map(self):
        """Creates char <-> int mappings"""

        char_map, idx_map = {}, {}
        for line in self.char_map_str.strip().split('\n'):
            c, num = line.split()
            char_map[c] = int(num)
            idx_map[int(num)] = c
        return char_map, idx_map

    def char_to_int(self, text):
        """Converts string to integer Tensor"""

        target = []
        for c in text:
            if c == ' ':
                target.append(self.char_map['<SPACE>'])
            else:
                target.append(self.char_map[c])
        return torch.Tensor(target)

    def target_to_text(self, target):
        """Converts integer array to string"""
        
        text = ''
        for idx in target:
            idx = int(idx)
            if idx == 1:
                text += ' '
            else:
                text += self.idx_map[idx]
        return text