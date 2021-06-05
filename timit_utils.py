import torch
from torch.utils.data import Dataset
import torchaudio
from config import DATASET_DIR
import os


class PhonemeTransformer:

    def __init__(self, collapse):
        # For now, only supports collapsing both train/test together
        self.collapse = collapse
        # Map from 61 phonemes to 39 as proposed in (Lee & Hon, 1989)
        # Added <SPACE> token for custom loss
        self.collapse_phon_map = { 
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

        if self.collapse:
            # 39 collapsed phonemes
            self.phon = ['<SPACE>', 'aa', 'ae', 'ah', 'aw', 'er', 'ay', 'b', 'sil', 'ch', 'd', 'dh', 'dx',
                                'eh', 'l', 'm', 'n', 'ng', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh',
                                'k', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v',
                                'w', 'y', 'z']
            self.phon_map = {self.phon[i]: i for i in range(len(self.phon))}
        else:
            # Full 61 phonemes
            self.phon = ['<SPACE>', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
                                'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey',
                                'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy',
                                'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p',
                                'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw',
                                'ux', 'v', 'w', 'y', 'z', 'zh']
            self.phon_map = {self.phon[i]: i for i in range(len(self.phon))}

    def phone_to_int(self, phonemes):
        """Converts phonemes to integer Tensor"""

        target = []

        if self.collapse:
            phonemes = self.collapse_phones(phonemes)

        for p in phonemes:
            target.append(self.phon_map[p])

        return torch.Tensor(target)

    def collapse_phones(self, phonemes):

        collapsed = []
        for p in phonemes:
            if p == 'q':
                continue
            collapsed.append(self.collapse_phon_map[p])

        return collapsed

    @property
    def blank_idx(self):
        """Used for pad_sequence in preprocessing"""
        if self.collapse:
            return self.phon_map['sil']
        else:
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
                # IMPORTANT: by forcibly removing any h# at the beginning or end, the padding logic in create_timit_target() works
                # If this is changed, things would break down
                if phonetic_details['phoneme'] != 'h#':
                    sample['phonemes'].append(phonetic_details)

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

def create_timit_target(words, phonemes, waveform):
    """
    Take phonemes + words and create transcript with <SPACE> separating word-phonemes.
    Example: 'she had' -> ['sh', 'ix', '<SPACE>', 'hv', 'eh', 'dcl']
    The <SPACE> functions similar to the blank in CTC, but we use it much less often

    Original idea: force the 1-to-1 correspondence for alignments by collapsing time dimension, didn't work well

    Updated improvement: keep length consistent with length of audio input, have phoneme repeat
    until next phoneme begins, TIMIT timestamps coming in clutch here
    Example: 'she' w/ 5 timesteps, 'sh' from [0-2], 'ix' from [3-4] -> ['sh', 'sh', 'sh', 'ix', 'ix']

    Note: overlapping timestamps exist for words (not phonemes) so go with start times of words/phonemes
    """

    waveform_len = waveform.shape[1]
    target = []
    for i in range(len(words)):
        curr_word_start, curr_word_end = int(words[i]['start']), int(words[i]['end'])

        if curr_word_start > 0 and i == 0:
            # pad with h# before first word (some samples do this already, some don't, but we force them all to omit in preprocessing)
            target.extend(['h#'] * (curr_word_start - 1))
            target.append('<SPACE>')

        if i < len(words) - 1:
            next_word_start = int(words[i+1]['start'])
        else:
            next_word_start = curr_word_end 

        for phoneme_details in phonemes:
            # repeatedly add phoneme for duration
            # TODO: add indicator for new phoneme for future interpretability? Or can do this in post since a timestep corresponding to 
            # a new phoneme makes no sense intuitively (and is reminscent of the blank in CTC), will also require new label
            phon_start, phon_end, phoneme = int(phoneme_details['start']), int(phoneme_details['end']), phoneme_details['phoneme']
            if phon_start >= curr_word_start: 
                if phon_start < next_word_start:
                    # target.append(phoneme)
                    target.extend([phoneme] * (phon_end - phon_start))
                else:
                    # Pop last repeat phoneme in word to make room for <SPACE>
                    target.pop()
                    # break will ensure this only happens once right after we reach the last phoneme within the word
                    break

        target.append('<SPACE>')

    # pad with h# after last word, off-by-1 for added <SPACE> after last word
    target.extend(['h#'] * (waveform_len - curr_word_end - 1))

    return target