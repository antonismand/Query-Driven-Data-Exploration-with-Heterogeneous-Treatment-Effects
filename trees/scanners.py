from copy import deepcopy
import itertools
from math import comb

from tqdm import tqdm


def Greedy(D, subgroups: list, max_t: float, op_matrix: list):
    if len(subgroups) <= D.k:
        return subgroups

    subgroups = sorted(subgroups, key=lambda x: x["t_est"], reverse=True)
    max_t = subgroups[0]["t_est"]

    total_subgroups = len(subgroups)
    percentile_index = int(0.9 * total_subgroups)
    bottom_10th_percentile = subgroups[percentile_index]["t_est"]
    theta = bottom_10th_percentile
    i = D.k + 1

    top_subs = deepcopy(subgroups[: D.k])
    scores = []

    for top in top_subs:
        top = D.compute_score_for_subgroup(
            top_subgroups=top_subs, s=top, max_t=max_t, op_matrix=op_matrix
        )
        scores.append(top["score"])

    min_score = min(scores)
    min_score_idx = scores.index(min_score)

    while i < total_subgroups and subgroups[i]["t_est"] > theta:
        candidate_sub = D.compute_score_for_subgroup(
            top_subgroups=top_subs, s=subgroups[i], max_t=max_t, op_matrix=op_matrix
        )

        if candidate_sub["score"] > min_score:
            top_subs.pop(min_score_idx)
            top_subs.append(candidate_sub)

            scores.pop(min_score_idx)
            scores.append(candidate_sub["score"])

            min_score = min(scores)
            min_score_idx = scores.index(min_score)

        i += 1

    return top_subs


def Exhaustive(D, subgroups: list, max_t: float, op_matrix: list):
    best_score = 0
    best_subs = None

    n_subgroups = len(subgroups)

    for candidates in tqdm(
        itertools.combinations(subgroups, D.k), total=comb(n_subgroups, D.k)
    ):
        subs, score = D.compute_scores_for_subgroups(list(candidates), max_t, op_matrix)
        if score > best_score:
            best_score = score
            best_subs = subs

    return best_subs
