import numpy as np

PHRASES = ['determine', 'unable', 'cannot', 'unknown', 'unsure', 'not possible']

def fallback_to_cot(res, cot, phrases):
    scores = []
    num_fallback = 0
    for k in res.keys():
        if any([phrase in res[k]['answer'].casefold() for phrase in phrases]) or not res[k]['answer']:
            scores.append(cot[k]['score'])
            num_fallback += 1
        else:
            scores.append(res[k]['score'])
    return np.mean(scores), num_fallback