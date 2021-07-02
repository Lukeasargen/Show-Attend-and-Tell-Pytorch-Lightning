from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.chrf_score import corpus_chrf
from nltk.translate.gleu_score import corpus_gleu
from nltk.translate.nist_score import corpus_nist
from nltk.translate.ribes_score import corpus_ribes

references = [
    [["a", "b", "c", "d", "e", "f"], ["c", "d", "e", "f"], ["e", "f", "g", "h"]],
    [["a", "b", "c", "d", "e", "f"], ["c", "d", "e", "f"], ["e", "f", "g", "h"]],
    [["a", "b", "c", "d", "e", "f"], ["c", "d", "e", "f"], ["e", "f", "g", "h"]],
]
captions = [
    ["b", "d", "e", "f", "g"],
    ["a", "b", "c", "f", "g", "h", "i"],
    ["c", "d", "e", "f"],
]

# references = [
#     [[1, 2, 3, 4, 5, 6], [3, 4, 5, 6], [5, 6, 7, 8]],
#     [[1, 2, 3, 4, 5, 6], [3, 4, 5, 6], [5, 6, 7, 8]],
# ]
# captions = [
#     [2, 4, 5, 6, 7],
#     [1, 3, 6, 7, 8, 9],
# ]

bleu1 = corpus_bleu(references, captions, weights=(1.0, 0, 0, 0))
bleu2 = corpus_bleu(references, captions, weights=(0.5, 0.5, 0, 0))
bleu3 = corpus_bleu(references, captions, weights=(0.33, 0.33, 0.33, 0))
bleu4 = corpus_bleu(references, captions, weights=(0.25, 0.25, 0.25, 0.25))
print(f"{bleu1 = }")
print(f"{bleu2 = }")
print(f"{bleu3 = }")
print(f"{bleu4 = }")


gleu = corpus_gleu(references, captions)
print(f"{gleu = }")

# This has a divide by zero error
nist = corpus_nist(references, captions, n=4)
print(f"{nist = }")

# This has a divide by zero error
ribes = corpus_ribes(references, captions)
print(f"{ribes = }")





cr, cc = [], []
for refs, c in zip(references, captions):
    mc = " ".join(c)
    mr = []
    for r  in refs:
        cr.append(r)
        cc.append(c)
        mr.append(" ".join(r))
    # m = meteor_score(mr, mc)
    # print(f"{m = }")


# Requires strings
chrf = corpus_chrf(cr, cc)
print(f"{chrf = }")



from nltk.metrics.scores import accuracy as corpus_accuracy
acc = corpus_accuracy(cr, cc)
print(f"{acc = }")

print(f"{cr = }")
print(f"{cc = }")




