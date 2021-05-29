import torch
from torch.utils.data import Dataset
import torchaudio
from config import DATASET_DIR
import os


class PhonemeTransformer:

    def __init__(self):
        # Map from 61 phonemes to 39 as proposed in (Lee & Hon, 1989)
        # Added <SPACE> token for custom loss
        self.phon_map = { 
            '<SPACE>': '<SPACE>',
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
            'zh': 'sh'	
        } 

        self.train_phon = ['<SPACE>', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
                            'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p',
                            'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw',
                            'ux', 'v', 'w', 'y', 'z', 'zh']
        self.train_phon_map = {self.train_phon[i]: i for i in range(len(self.train_phon))}

        self.test_phon = ['<SPACE>', 'aa', 'ae', 'ah', 'aw', 'er', 'ay', 'b', 'sil', 'ch', 'd', 'dh', 'dx',
                            'eh', 'l', 'm', 'n', 'ng', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh',
                            'k', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v',
                            'w', 'y', 'z']
        self.test_phon_map = {self.test_phon[i]: i for i in range(len(self.test_phon))}

    def phone_to_int(self, phonemes, collapse=True):
        """Converts phonemes to integer Tensor"""

        target = []

        if collapse:
            phonemes = self.collapse_phones(phonemes)
            for p in phonemes:
                target.append(self.test_phon_map[p])
        else:
            for p in phonemes:
                target.append(self.train_phon_map[p])

        return torch.Tensor(target)

    def collapse_phones(self, phonemes):

        collapsed = []
        for p in phonemes:
            if p == 'q':
                continue
            collapsed.append(self.phon_map[p])

        return collapsed

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

        path = self.prefix_paths[idx]

        sample = {'audio': [], 'phonemes': [], 'words': [], 'transcript': ''}
        
        wavpath = path + '.WAV'
        waveform, sr = torchaudio.load(wavpath)
        sample['audio'] = waveform

        wrdpath = path + '.WRD'
        with open(wrdpath) as f:
            for line in f.read().splitlines():
                word_details = {}
                # start_index | end_index | word
                word_details['start'], word_details['end'], word_details['word'] = line.split(' ')
                sample['words'].append(word_details)

        txtpath = path + '.TXT'
        with open(txtpath) as f:
            for line in f.read().splitlines():
                # start_index | end_index | transcript
                transcript = ' '.join(line.split(' ')[2:])
                # TODO: lowercase?
                sample['transcript'] = transcript

        phnpath = path + '.PHN'
        with open(phnpath) as f:
            for line in f.read().splitlines():
                phonetic_details = {}
                # start_index | end_index | phoneme
                phonetic_details['start'], phonetic_details['end'], phonetic_details['phoneme'] = line.split(' ')
                # TODO: omit h# phoneme?
                sample['phonemes'].append(phonetic_details)

        return sample

    def create_timit_paths(self, path):
        """
        Given a path to a TIMIT dataset (train/test), this will generate a list of paths to each group of files referring to one sample.
        Paths of the form timit/data/(TRAIN/TEST)/dr/speaker_id/sample_code, i.e. timit/data/TRAIN/DR1/FCJF0/SI648. This can be used so that
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

def create_timit_target(words, phonemes):
    """
    Take phonemes + words and create transcript with <SPACE> separating word-phonemes.
    Example: 'she had' -> ['sh', 'ix', '<SPACE>', 'hv', 'eh', 'dcl']

    Problem: overlapping timestamps exist
    Solution: go with start times of phonemes
    """

    target = []
    for i in range(len(words)):
        curr_word_start = int(words[i]['start'])
        if i < len(words) - 1:
            next_word_start = int(words[i+1]['start'])
        else:
            next_word_start = int(words[i]['end'])

        for phoneme_details in phonemes:
            phon_start, phoneme = int(phoneme_details['start']), phoneme_details['phoneme']
            if phon_start >= curr_word_start and phon_start < next_word_start:
                target.append(phoneme)
        
        if i < len(words) - 1:
            # Prevents extra space at end
            target.append('<SPACE>')

    return target