from nltk.translate.bleu_score import corpus_bleu
import numpy as np


def token_ngrams(sentence, n):
    """ Helper function that makes a dictionary with the
        ngram as key and the value is its count
    """
    ngram_dict = {}
    for i in range(len(sentence)-n+1):
        ng_key = " ".join([str(i) for i in sentence[i:i+n]]).lower()
        if ng_key  in ngram_dict:
            ngram_dict[ng_key] += 1
        else:
            ngram_dict[ng_key] = 1
    return ngram_dict

def token_bleu(references, captions, weights):
    """ These Equations came straight from the paper:
        'BLEU a Method for Automatic Evaluation of Machine Translation'
    """
    assert len(references)==len(captions)
    if type(weights)==int: weights=[0]*(weights-1) + [1]
    assert type(weights)==list
    # Lists for each ngram numerator and denominator
    bleu_num, bleu_dem = [0]*len(weights), [0]*len(weights)
    # Totals for the brevity penalty
    cap_lens, ref_lens = 0, 0
    # For each weight get the modified precision
    for refs, cap in zip(references, captions):
        for i, w in enumerate(weights):
            if w==0: continue
            n = i+1  # List are indexed with i, but the ngrams is plus 1
            # First make ngram dictionaries for the caption and each reference
            cap_ngram = token_ngrams(cap, n)
            ref_ngrams = [token_ngrams(ref, n) for ref in refs]
            for cap_key, cap_count in cap_ngram.items():
                # For each ngram in the caption, find the highest count in the references
                max_ref_count = max([ref_ngram[cap_key] if cap_key in ref_ngram else 0 for ref_ngram in ref_ngrams ])
                # Add the totals for modified precision of this ngram size
                bleu_num[i] += min(cap_count, max_ref_count)  # Clip the cap_count to max count in references
                bleu_dem[i] += cap_count
                # Add the totals for the brevity penalty
        cap_lens += len(cap)
        # Find the closest length by make a tuple with absolute difference and length, find the min and take the length
        ref_lens += min((abs(len(ref)-len(cap)), len(ref)) for ref in refs)[1]

    brevity_penalty = np.exp( min(0, 1-ref_lens/cap_lens) )  # Should not be more than 1 so us min(0, x)
    # print(f"{cap_lens = } {ref_lens = }")
    # print(f"{brevity_penalty = }")
    # print(f"{bleu_num = } {bleu_dem = }")
    return brevity_penalty*np.exp( np.sum([w*np.log(bleu_num[i]/bleu_dem[i]) if bleu_dem[i]!=0 else 0 for i,w in enumerate(weights)]) )


references = [
    [[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6], [5, 6, 7, 8]],
    [[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6], [5, 6, 7, 8]],
]
captions = [
    [2, 4, 5, 6, 7],
    [1, 3, 6, 7, 8, 9],
]


weights = [
    [1, 0, 0, 0],
    [0.5, 0.5, 0, 0],
    [0.33, 0.33, 0.33, 0],
    [0.25, 0.25, 0.25, 0.25],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]


for w in weights:
    bleu = corpus_bleu(references, captions, weights=w)
    my_bleu = token_bleu(references, captions, weights=w)
    print(f"{bleu=:.4f} {my_bleu=:.4f} {w=} ")



