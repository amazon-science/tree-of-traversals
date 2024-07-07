import os
import json
import numpy as np

RESULTS_DIR = 'results'


def main():
    for subdir in os.listdir(RESULTS_DIR):
        subdir = os.path.join(RESULTS_DIR, subdir)
        for res_file in os.listdir(subdir):
            res = read_file(subdir, res_file)
            mean_score = average_score(res)
            print(f"{subdir}/{res_file}\n\tsamples:{len(res)}\tscore:{mean_score}")


def read_file(subdir, filename):
    res = json.load(open(os.path.join(subdir, filename)))
    return res


def average_score(res):
    scores = np.array([r['score'] for r in res.values()])
    return scores.mean()

if __name__=="__main__":
    main()
