import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict
import re
from dataset_utils import SPACE_TOKEN

# Based on ARPAbet symbol set
# http://www.speech.cs.cmu.edu/cgi-bin/cmudict

cmu_d = cmudict.dict()

def get_lyrics(fname, timit=False):

    final_lyrics = []
    
    with open(fname, "r") as f:
        lyrics = f.read().lower()
        lyrics = re.sub(r"[\"(),.;?!\-]", "", lyrics).split('\n')
        for line in lyrics:
            words = line.split()
            final_lyrics.extend(words)

    if timit:
        print(final_lyrics)
        return final_lyrics[2:]
    else:        
        return final_lyrics

def pronunciation_model(transcript, transformer):

    p = []
    for word in transcript:
        word = (re.sub(r"[\"(),.?;!\-]", "", word)).lower()
        # Remove stress numbers
        if word not in cmu_d:
            # word still appears in alignment dictionary and so will look for spaces to map to, what should we do here?
            # TODO: this is still an edge case that can break things pretty badly
            p.append(SPACE_TOKEN)
            continue
        p_with_stresses = cmu_d[word][0]
        for phon in p_with_stresses:
            p.append(re.sub(r"[0-9]", "", phon).lower())
        p.append(SPACE_TOKEN)

    return transformer.map_to_39(p)

def generate_spaces_in_guess(guessed_transcript, true_transcript, path):

    # To find where the <sp> should occur in guessed string, look on edit path through DP matrix and find the guessed-string-index that
    # corresponds to where the <sp> in the truth-string is on the edit path
    # Also returns list of indices where <sp>'s are

    guessed_string_with_spaces = []

    # First node in path is always (0, 0) which corresponds to blanks
    path.pop(0)

    prev = None
    for guessed_transcript_idx, true_transcript_idx in path:
        # DP matrix has strings starting at index 1 of matrix, hence -1
        curr = guessed_transcript[guessed_transcript_idx-1]
        # Horizontal moves along DP path correspond would cause repeated phones
        if curr != prev:
            guessed_string_with_spaces.append(curr)
        if true_transcript[true_transcript_idx-1] == SPACE_TOKEN:
            guessed_string_with_spaces.append(SPACE_TOKEN)
        prev = curr

    return guessed_string_with_spaces


