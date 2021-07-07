from dataset_utils import PhonemeTransformer
import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict
import re
import torch
from inference import collapse_repeats

# Based on ARPAbet symbol set
# http://www.speech.cs.cmu.edu/cgi-bin/cmudict

cmu_d = cmudict.dict()

def get_lyrics(fname):

    final_lyrics = []
    
    with open(fname, "r") as f:
        lyrics = f.read().lower()
        lyrics = re.sub(r"[\"(),.?!\-]", "", lyrics).split('\n')
        for line in lyrics:
            words = line.split()
            final_lyrics.extend(words)
            
    # Temporary slicing for TIMIT transcripts having timsteps at beginning
    return final_lyrics[2:]
    
# Spring Street is straight ahead.

fpath = '/mnt/d/Datasets/timit/data/TRAIN/DR4/MESG0/SX72.TXT'

lyrics = get_lyrics(fpath)

transformer = PhonemeTransformer()

p = []
for word in lyrics:
    # Remove stress numbers
    p_with_stresses = cmu_d[word][0]
    for phon in p_with_stresses:
        p.append(re.sub(r"[0-9]", "", phon).lower())

p = transformer.map_to_39(p)

print(p)

guess = torch.Tensor([27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 48, 48, 48, 48, 48, 48, 48, 48,
        45, 45, 45, 45, 45, 45, 43, 47, 47, 47, 47, 47, 30, 30, 30, 30, 39, 39,
        39, 39, 39, 38, 48, 48, 48, 48, 48, 48, 48, 51, 51, 50, 50, 50, 47, 47,
        47, 47, 47, 32, 32, 32, 32, 32, 32, 32, 32, 32, 31, 31, 49, 49, 48, 49,
        48, 48, 49, 48, 49, 48, 51, 51, 50, 50, 50, 50, 22, 22, 22, 22, 47, 22,
        23, 23, 23, 15, 15, 15, 12, 29, 29, 29, 29, 29, 29, 29, 29, 16, 16, 16,
        16, 30, 30, 30, 30, 30, 30, 30, 13, 13, 13, 13, 13, 27, 13, 27, 27, 27,
        27, 27, 27, 27, 27, 27, 27, 27, 27])

# Must do int -> 39 txt phones -> collapsing repeats
# Going from int to 39 can introduce extra repeats as well
guess = collapse_repeats(transformer.target_to_text(guess))

print(guess)

# Groud truth: ['s', 'p', 'r', 'ih', 'ng', 's', 't', 'r', 'iy', 't', 'ih', 'z', 's', 't', 'r', 'ey', 't', 'ah', 'hh', 'eh', 'd']
# Guess: ['sil', 's', 'sil', 'p', 'r', 'ih', 'ng', 'n', 's', 'sil', 't', 'r', 'iy', 'ih', 'sh', 's', 'sh', 's', 'sh', 's', 'sh', 's', 'sil', 't', 'er', 'r', 'er', 'ey', 'dx', 'd', 'hh', 'eh', 'ih', 'sil']