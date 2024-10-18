from copy import deepcopy
import itertools


def Greedy(D, subgroups: list, max_t: float):
    if len(subgroups) <= D.k:
        return subgroups

    subgroups = sorted(subgroups, key=lambda x: x["t_est"], reverse=True)

    total_subgroups = len(subgroups)
    percentile_index = int(0.9 * total_subgroups)
    bottom_10th_percentile = subgroups[percentile_index]["t_est"]

    top_subs = deepcopy(subgroups[: D.k])

    top_subs, min_score, min_score_i, _ = D.compute_scores_for_subgroups(
        top_subs, max_t
    )

    theta = bottom_10th_percentile
    i = D.k + 1

    while i < total_subgroups and subgroups[i]["t_est"] > theta:
        score_si = D.compute_score_for_subgroup(
            top_subgroups=top_subs, s=subgroups[i], max_t=max_t
        )
        if score_si > min_score:
            top_subs.pop(min_score_i)
            top_subs.append(subgroups[i])
            top_subs, min_score, min_score_i, _ = D.compute_scores_for_subgroups(
                top_subs, max_t
            )
        i += 1

    return top_subs


def Exhaustive(D, subgroups: list, max_t: float):
    best_score = 0
    best_subs = None

    for candidates in itertools.combinations(subgroups, D.k):
        subs, _, _, score = D.compute_scores_for_subgroups(list(candidates), max_t)
        if score > best_score:
            best_score = score
            best_subs = subs

    return best_subs
