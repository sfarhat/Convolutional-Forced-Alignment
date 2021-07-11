import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict
import re

# Based on ARPAbet symbol set
# http://www.speech.cs.cmu.edu/cgi-bin/cmudict

cmu_d = cmudict.dict()

def get_lyrics(fname, timit=False):

    final_lyrics = []
    
    with open(fname, "r") as f:
        lyrics = f.read().lower()
        lyrics = re.sub(r"[\"(),.?!\-]", "", lyrics).split('\n')
        for line in lyrics:
            words = line.split()
            final_lyrics.extend(words)
            
    if timit:
        # Temporary slicing for TIMIT transcripts having timsteps at beginning
        return final_lyrics[2:]
    else:
        return final_lyrics

def pronunciation_model(transcript, transformer):

    p = []
    for word in transcript:
        # Remove stress numbers
        p_with_stresses = cmu_d[word][0]
        for phon in p_with_stresses:
            p.append(re.sub(r"[0-9]", "", phon).lower())
        p.append("<sp>")

    return transformer.map_to_39(p)

def generate_spaces_in_guess(guessed_transcript, true_transcript, path):

    # To find where the <sp> should occur in guessed string, look on edit path through DP matrix and find the guessed-string-index that
    # corresponds to where the <sp> in the truth-string is on the edit path

    guessed_string_with_spaces = []
    space_indices = []

    # First node in path is always (0, 0) which corresponds to blanks
    path.pop(0)

    for guessed_transcript_idx, true_transcript_idx in path:
        # DP matrix has strings starting at index 1 of matrix, hence -1
        guessed_string_with_spaces.append(guessed_transcript[guessed_transcript_idx-1])
        if true_transcript[true_transcript_idx-1] == '<sp>':
            guessed_string_with_spaces.append('<sp>')
            space_indices.append(len(guessed_string_with_spaces) - 1)

    return guessed_string_with_spaces, space_indices


