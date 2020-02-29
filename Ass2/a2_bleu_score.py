# Copyright 2020 University of Toronto, all rights reserved

'''Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

from math import exp  # exp(x) gives e^x


def grouper(seq, n):
    '''Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    '''
    list = []
    for i in range(0, len(seq) - n + 1):
        list.append(" ".join(seq[i: i + n]))
    return list



def n_gram_precision(reference, candidate, n):
    '''Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''

    # special case
    if len(candidate) == 0:
        return 0

    # variable and modified variables
    reference_n_grams = set(grouper(reference, n))
    candidate_n_grams = grouper(candidate, n)
    N = len(candidate_n_grams)
    C = 0

    # cound occurances
    for n_gram in candidate_n_grams:
        if n_gram in reference_n_grams:
            C = C + 1

    return C/N





def brevity_penalty(reference, candidate):
    '''Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''
    # special case
    if len(candidate) == 0:
        return 0

    # variables
    BP = 0
    br = len(reference)/len(candidate)

    if br <= 1:
        BP = 1
    else:
        BP = exp(1-br)
    return BP



def BLEU_score(reference, hypothesis, n):
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''
    bleu = 1

    for i in range(0+1, n+1):
        bleu = bleu * n_gram_precision(reference, hypothesis, i)
    bleu = bleu * brevity_penalty(reference, hypothesis)

    return bleu



# to test the bleu score
def test():
    references = [
        "It is a guide to action that ensures that the military will forever heed Party commands",
        "It is the guiding principle which guarantees the military forces always being under command of the Party",
        "It is the practical guide for the army always to heed the directions of the party"
    ]
    candidates = [
        "It is a guide to action which ensures that the military always obeys the commands of the party",
        "It is the insure the troops forever hearing the activity guidebook that party direct"
    ]
    references_list = []
    for item in references:
        references_list.append(item.split())
        for i in range(0, len(references_list[-1])):
            references_list[-1][i] = references_list[-1][i].lower()
    candidates_list = []
    for item in candidates:
        candidates_list.append(item.split())
        for i in range(0, len(candidates_list[-1])):
            candidates_list[-1][i] = candidates_list[-1][i].lower()
    n = 2
    print(BLEU_score(references_list[0], candidates_list[0], n))

# print(grouper(references_list[0], n))