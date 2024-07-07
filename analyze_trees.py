import math
import pickle
import os
import json
import statistics
import argparse
import pandas as pd
import scipy.stats as stats

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from treeoftraversals import TreeOfTraversals
from wikimultihop import score_answers

TREE_DIR = 'results/2wiki/ToTraversals_b3_d7_aug8'
RESULTS_FILE = 'results/2wiki/ToTraversals_b3_d7_aug8_corrected.json'

parser = argparse.ArgumentParser()
parser.add_argument('--results_file', type=str, default='figures', help='File results were saved in')
parser.add_argument('--figures_dir', type=str, default='figures', help='File to save figures')

args = parser.parse_args()

if args.results_file is not None:
    RESULTS_FILE = args.results_file
    TREE_DIR = os.path.splitext(RESULTS_FILE)[0]

def question2tree(idx):
    file = os.path.join(TREE_DIR, f"{idx}_tree.pkl")
    tree = pickle.load(open(file, 'rb'))
    return tree

def states_expanded(t):
    states = list(t.state_tree.traverse())
    expanded_states = [s for s in states if s.children]
    return len(expanded_states)

def tree_size(t):
    return len(list(t.state_tree.traverse()))

def get_answer_value(t):
    answer_state = t.answer_state()
    nodes = list(t.state_tree.traverse())
    answer_node = [node for node in nodes if node.state is answer_state][0]
    return answer_node.value

def get_average_answer_value(t):
    answer_nodes = []
    for node in t.state_tree.traverse():
        if len(node.state.trajectory) and "ANSWER" in node.state.trajectory[-1]:
            answer_nodes.append(node)
    ans_vals = [a.value for a in answer_nodes]
    return sum(ans_vals) / len(ans_vals)

def get_and_score_all_answers(t: TreeOfTraversals, label):
    answer_nodes = []
    for node in t.state_tree.traverse():
        if len(node.state.trajectory) and "ANSWER" in node.state.trajectory[-1]:
            answer_nodes.append(node)
    ans_vals = [a.value for a in answer_nodes]
    ans = [t.answer_from_state(node.state) for node in answer_nodes]
    ans_scores = [score_answers(a, label) for a in ans]
    return ans_vals, ans_scores

def get_ratio_ans_val_over_threshold(t):
    answer_nodes = []
    for node in t.state_tree.traverse():
        if len(node.state.trajectory) and "ANSWER" in node.state.trajectory[-1]:
            answer_nodes.append(node)
    over_threshold = [a.value > 0.8 for a in answer_nodes]
    return sum(over_threshold) / len(over_threshold)

def alternative_answers_rejected(t, label):
    answer_state = t.answer_state()
    nodes = list(t.state_tree.traverse())
    answer_nodes = [node for node in nodes if node.state is answer_state]
    if len(answer_nodes) == 0:
        raise ValueError("No answer states")
    answer_node = answer_nodes[0]
    answer_node_parent = answer_node.parent

    prior_rejected_answer_nodes = []
    for node in t.state_tree.traverse():
        if len(node.state.trajectory) and "ANSWER" in node.state.trajectory[-1] and node.parent != answer_node_parent:
            prior_rejected_answer_nodes.append(node)
    rejected_answers = [t.answer_from_state(node.state) for node in prior_rejected_answer_nodes]
    rejected_scores = [score_answers(a, label) for a in rejected_answers]
    rejected_vals = [node.value for node in prior_rejected_answer_nodes]
    return rejected_vals, rejected_scores

def first_alt_answer_rejected(t, label):
    answer_state = t.answer_state()
    nodes = list(t.state_tree.traverse())
    answer_nodes = [node for node in nodes if node.state is answer_state]
    if len(answer_nodes) == 0:
        raise ValueError("No answer states")
    answer_node = answer_nodes[0]
    answer_node_parent = answer_node.parent

    prior_rejected_answer_nodes = []
    for node in t.state_tree.traverse():
        if len(node.state.trajectory) and "ANSWER" in node.state.trajectory[-1] and node.parent != answer_node_parent:
            prior_rejected_answer_nodes.append(node)
    if not prior_rejected_answer_nodes:
        return None, None
    first_rejected = prior_rejected_answer_nodes[0]
    for node in prior_rejected_answer_nodes:
        try:
            if node.parent.order < first_rejected.parent.order:
                first_rejected = node
        except AttributeError:
            return None, None
    all_first_rejected = [node for node in prior_rejected_answer_nodes if node.parent.order == first_rejected.parent.order]
    best_first_rejected = max(all_first_rejected, key=lambda x: x.value)
    best_first_rejected_val = best_first_rejected.value
    best_first_rejected_score = score_answers(t.answer_from_state(best_first_rejected.state), label)

    return best_first_rejected_val, best_first_rejected_score



def get_answer_path_values(t):
    answer_state = t.answer_state()
    nodes = list(t.state_tree.traverse())
    answer_nodes = [node for node in nodes if node.state is answer_state]
    if len(answer_nodes) == 0:
        raise ValueError("No answer states")
    answer_node = answer_nodes[0]
    path_to_root = []
    current = answer_node
    path_to_root.append(current)
    while current is not None:
        path_to_root.append(current)
        current = current.parent
    path_to_answer = path_to_root[::-1][1:]
    vals = [n.value for n in path_to_answer]
    alternative_max = []
    alternative_expanded = []
    for i in range(len(path_to_answer) - 1):
        n = path_to_answer[i]
        if not n.children:
            break
        alt_childs = [c for c in n.children if c is not path_to_answer[i+1]]
        child_vals = [c.value for c in alt_childs]
        alternative_expanded.append(any([c.children for c in alt_childs]))
        if child_vals:
            alternative_max.append(max(child_vals))
        else:
            alternative_max.append(None)
    return vals, alternative_max, alternative_expanded


def create_histogram(pos_results, neg_results, key=None, title=None, auto_bin=False, xlabel='Value', ylabel='Frequency'):
    plt.style.use('seaborn-v0_8-whitegrid')
    pathmins_pos = np.array([r[key] for r in pos_results if key in r])
    pathmins_neg = np.array([r[key] for r in neg_results if key in r])
    weightspos = np.ones_like(pathmins_pos) * 1/len(pathmins_pos)
    weightsneg = np.ones_like(pathmins_neg) * 1 / len(pathmins_neg)
    bins = np.arange(-0.05, 1.05, 0.1)
    if auto_bin:
        if auto_bin is True:
            bins = None
        else:
            bins = auto_bin
    plt.hist([pathmins_neg, pathmins_pos],
             bins=bins,
             alpha=1.0,
             weights=[weightsneg, weightspos],
             label=['Incorrect', 'Correct'])
    # plt.hist(pathmins_neg, bins=np.arange(0, 1.1, 0.1), alpha=0.5, label='Incorrect')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(args.figures_dir, title + ".pdf"), format='pdf')
    # plt.show()
    plt.close()


    # plt.hist(pathmins_neg,
    #          bins=bins,
    #          alpha=0.5,
    #          weights=weightsneg,
    #          label='Incorrect')
    # plt.hist(pathmins_pos,
    #          bins=bins,
    #          alpha=0.5,
    #          weights=weightspos,
    #          label='Correct')
    # # plt.hist(pathmins_neg, bins=np.arange(0, 1.1, 0.1), alpha=0.5, label='Incorrect')
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.title(title)
    # plt.legend()
    # plt.show()


def main():
    results = data = json.load(open(RESULTS_FILE, 'r'))
    for i in results.keys():
        # results[i]['tree'] = question2tree(i)
        results[i]['tree'] = pickle.load(open(results[i]['treefile'], 'rb'))

    # tree_files = [os.path.join(TREE_DIR, f) for f in os.listdir(TREE_DIR)]
    # trees = [pickle.load(open(t, 'rb')) for t in tree_files]
    pos_results = [r for r in results.values() if r['score'] > 0]
    neg_results = [r for r in results.values() if r['score'] == 0]

    all_ans_vals = []
    all_ans_scores = []

    ans_path_vals_correct = []
    ans_path_vals_incorrect = []

    for i in results.keys():
        t = results[i]['tree']
        try:
            vpath, maxalt, altexpanded = get_answer_path_values(t)
            if results[i]['score'] > 0:
                ans_path_vals_correct += vpath
            else:
                ans_path_vals_incorrect += vpath
            rejected_vals, rejected_scores = alternative_answers_rejected(t, results[i]['answer'])
            if rejected_scores:
                results[i]['mean_rejected_ans_score'] = np.mean(rejected_scores)
            first_rejected_val, first_rejected_score = first_alt_answer_rejected(t, results[i]['answer'])
            if first_rejected_score is not None:
                results[i]['first_rejected_score'] = first_rejected_score
            results[i]['path_min'] = min([v for v in vpath if v >= 0.0])
            results[i]['alternative_expanded'] = any(altexpanded)
            results[i]['count_alternative_expanded'] = sum(altexpanded)
            results[i]['contains_answer_state'] = True
            results[i]['path_avg'] = statistics.mean([v for v in vpath if v >= 0.0])
            results[i]['ratio_ans_over_threshold'] = get_ratio_ans_val_over_threshold(t)
            results[i]['average_ans_value'] = get_average_answer_value(t)
            results[i]['final_ans_value'] = get_answer_value(t)
            ans_vals, ans_scores = get_and_score_all_answers(t, results[i]['answer'])
            all_ans_vals += ans_vals
            all_ans_scores += ans_scores
        except ValueError:
            results[i]['contains_answer_state'] = False
        results[i]['tree_size'] = tree_size(t)


    # TODO: Make this good
    plt.style.use('seaborn-v0_8-whitegrid')
    ans_path_vals_correct = list(filter(lambda x: x>=0, ans_path_vals_correct))
    ans_path_vals_incorrect = list(filter(lambda x: x >= 0, ans_path_vals_incorrect))
    weightspos = np.ones_like(ans_path_vals_correct) * 1 / len(ans_path_vals_correct)
    weightsneg = np.ones_like(ans_path_vals_incorrect) * 1 / len(ans_path_vals_incorrect)
    plt.hist([ans_path_vals_incorrect, ans_path_vals_correct],
             bins=np.arange(-0.05, 1.05, 0.1),
             alpha=1.0,
             weights=[weightsneg, weightspos],
             label=['Incorrect', 'Correct'])
    plt.title('Distribution of Values on Answer Path')
    plt.xlabel('Node Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(args.figures_dir, 'Distribution_of_Values_on_Answer_Path.pdf'), format='pdf')
    # plt.show()
    plt.close()

    # plot counterfactual, without backtracking on answer
    num_backtrack_counterfactuals = len([r['score'] for r in results.values() if 'first_rejected_score' in r])
    score_with_backtrack_on_answer = np.mean([r['score'] for r in results.values() if 'first_rejected_score' in r])
    score_without_backtrack_on_answer = np.mean([r['first_rejected_score'] for r in results.values() if 'first_rejected_score' in r])
    x = ['No Backtrack on Answer', 'Yes Backtrack on Answer']
    y = np.array([score_without_backtrack_on_answer, score_with_backtrack_on_answer])
    print(x[0], y[0])
    print(x[1], y[1])
    print('num backtrack counterfactuals', num_backtrack_counterfactuals)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.bar([], [])
    bars = plt.bar(x, y)
    plt.ylabel('Score (EM-in)')
    plt.title(f'Effect of Backtracking on Answer (relevant subset only, {num_backtrack_counterfactuals})')
    # plt.text(x[0], y[0] + 0.01, "{:.3f}".format(score_without_backtrack_on_answer), horizontalalignment='left')
    # plt.text(x[1], y[1] + 0.01, "{:.3f}".format(score_with_backtrack_on_answer), horizontalalignment='left')
    for p in bars:
        plt.annotate("{:.3f}".format(p.get_height()),
                xy=(p.get_x() + p.get_width() / 2, p.get_height()),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')
    plt.ylim(0, score_with_backtrack_on_answer + 0.1)
    plt.savefig(os.path.join(args.figures_dir, 'Effect_of_Backtracking.pdf'), format='pdf')
    # plt.show()
    plt.close()



    create_histogram(pos_results, neg_results, 'first_rejected_score', 'Score of First Rejected Answer', auto_bin=True, xlabel="Score of First Rejected Answer")
    create_histogram(pos_results, neg_results, 'path_min', 'Min Value on Answer Path', xlabel='Min Value')
    create_histogram(pos_results, neg_results, 'path_avg', 'Average Value on Answer Path', xlabel='Average Value')
    create_histogram(pos_results, neg_results, 'mean_rejected_ans_score', 'Mean Score of Rejected Answers', auto_bin=True, xlabel="Mean Score of Rejected Answers for Each Tree")
    create_histogram(pos_results, neg_results, 'alternative_expanded', 'Alternative Value Expanded', auto_bin=True)
    create_histogram(pos_results, neg_results, 'count_alternative_expanded', 'Alternative Subtrees Explored', auto_bin=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], xlabel="Count of Alternative Subtrees Explored")
    create_histogram(pos_results, neg_results, 'ratio_ans_over_threshold', 'Ratio of Answers over Acceptance Threshold')
    create_histogram(pos_results, neg_results, 'average_ans_value', 'Average Answer Value')
    create_histogram(pos_results, neg_results, 'final_ans_value', 'Final Answer Value')
    create_histogram(pos_results, neg_results, 'tree_size', 'Tree Size', auto_bin=True)
    v_correct = [v for v, s in zip(all_ans_vals, all_ans_scores) if s > 0]
    v_incorrect = [v for v, s in zip(all_ans_vals, all_ans_scores) if s == 0]
    v_hist_correct, bin_edges = np.histogram(v_correct, 10, (0, 1.0))
    v_hist_incorrect, bin_edges = np.histogram(v_incorrect, 10, (0, 1.0))
    correct_ratio = v_hist_correct / (v_hist_incorrect + v_hist_correct)
    incorrect_ratio = v_hist_incorrect / (v_hist_incorrect + v_hist_correct)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.bar(bin_edges[:-1], incorrect_ratio, width=np.diff(bin_edges), align='edge', bottom=correct_ratio,
            label='Incorrect')
    plt.bar(bin_edges[:-1], correct_ratio, width=np.diff(bin_edges), align='edge', label='Correct')
    plt.xlabel('Bins')
    plt.ylabel('Ratio')
    plt.title('Correct Answers based on Final Value')
    plt.legend()
    plt.savefig(os.path.join(args.figures_dir, 'Correct_Answers_based_on_Final_value.pdf'), format='pdf')
    # plt.show()
    plt.close()

    print("Correlation between answer values and answer scores")
    print(np.corrcoef(all_ans_vals, all_ans_scores))

    df = pd.DataFrame({'Value': all_ans_vals, 'Answer Score': all_ans_scores})
    jittered_df = df + + np.random.normal(0, 0.05, size=(len(df), 2))
    sns.scatterplot(x='Value', y='Answer Score', data=jittered_df, alpha=0.05)
    sns.regplot(x='Value', y='Answer Score', data=df,
                ci=95,
                scatter=False)
    r, p = stats.pearsonr(df['Value'], df['Answer Score'])
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.text(0.05, .55, 'corr = {:.2f}\np = {:.2g}'.format(r, p))
    plt.savefig(os.path.join(args.figures_dir, 'Answer_Score_From_Value.pdf'), format='pdf')
    plt.close()


    val_correct, counts_correct = np.unique(v_correct, return_counts=True)
    val_incorrect, counts_incorrect = np.unique(v_incorrect, return_counts=True)
    val_intersect = np.intersect1d(val_correct, val_incorrect)
    counts_correct_filtered = counts_correct[np.isin(val_correct, val_intersect)]
    counts_incorrect_filtered = counts_incorrect[np.isin(val_incorrect, val_intersect)]
    ratio = counts_correct_filtered / (counts_correct_filtered + counts_incorrect_filtered)
    print(f"Ratio correct for value 0.0: {ratio[0]}")
    print(f"Ratio correct for value 1.0: {ratio[-1]}")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.plot(val_intersect, ratio, '-o')
    plt.xlabel('Value Function for Answer State')
    plt.ylabel('Percent Correct (EM-in)')
    plt.title('Value Function Predicts Correctness')
    plt.savefig(os.path.join(args.figures_dir, 'Value_Function_Predicts_Correctness.pdf'), format='pdf')
    # plt.show()
    plt.close()



if __name__ == "__main__":
    main()