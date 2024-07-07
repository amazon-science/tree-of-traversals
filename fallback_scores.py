import numpy as np
import json

load = lambda x: json.load(open(x))


def fallback(answers, score_a, score_cot, condition):
    count = 0
    scores = []
    for i in range(len(answers)):
        if condition(answers[i]):
            scores.append(score_cot[i])
            count += 1
        else:
            scores.append(score_a[i])
    return np.mean(scores), count