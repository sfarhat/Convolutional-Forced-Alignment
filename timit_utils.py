from torch.utils.data import Dataset
import torchaudio
from config import DATASET_DIR
import os

# 61 labels training (some do 48), 39 labels for testing
timit_training_labels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw',
                        'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux',
                        'er', 'ax', 'ix', 'axr', 'ax-h', 'jh', 'ch', 'b',
                        'd', 'g', 'p', 't', 'k', 'dx', 's', 'sh', 'z',
                        'zh', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng',
                        'em', 'nx', 'en', 'eng', 'l', 'r', 'w', 'y', 'hh',
                        'hv', 'el', 'bcl', 'gcl', 'pcl', 'tcl', 'kcl',
                        'q', 'pau', 'epi', 'h#']

# aa, ao
# ah, ax, ax-h
# er, axr
# hh, hv
# ih, ix
# l, el
# m, em
# n, en, nx
# ng, eng
# sh, zh
# uw, ux
# pcl, tcl, kcl, bcl, dcl, gcl, h#, pau, epi
# q	
# become mapped to
# aa
# ah
# er
# hh
# ih
# l
# m
# n
# ng
# sh
# uw
# sil
# -

# aa	aa	aa
# ae	ae	ae
# ah	ah	ah
# ao	ao	aa
# aw	aw	aw
# ax	ax	ah
# ax-h	ax	ah
# axr	er	er
# ay	ay	ay
# b	b	b
# bcl	vcl	sil
# ch	ch	ch
# d	d	d
# dcl	vcl	sil
# dh	dh	dh
# dx	dx	dx
# eh	eh	eh
# el	el	l
# em	m	m
# en	en	n
# eng	ng	ng
# epi	epi	sil
# er	er	er
# ey	ey	ey
# f	f	f
# g	g	g
# gcl	vcl	sil
# h#	sil	sil
# hh	hh	hh
# hv	hh	hh
# ih	ih	ih
# ix	ix	ih
# iy	iy	iy
# jh	jh	jh
# k	k	k
# kcl	cl	sil
# l	l	l
# m	m	m
# n	n	n
# ng	ng	ng
# nx	n	n
# ow	ow	ow
# oy	oy	oy
# p	p	p
# pau	sil	sil
# pcl	cl	sil
# q
# r	r	r
# s	s	s
# sh	sh	sh
# t	t	t
# tcl	cl	sil
# th	th	th
# uh	uh	uh
# uw	uw	uw
# ux	uw	uw
# v	v	v
# w	w	w
# y	y	y
# z	z	z
# zh	zh	sh

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
        
        paths = []
        for root, dirs, files in os.walk(path):
            for fname in files:
                sample_id = fname.split('.')[0]
                if 'SA' in sample_id:
                    continue
                fpath = os.path.join(root, sample_id)
                paths.append(fpath)
                
        return paths

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