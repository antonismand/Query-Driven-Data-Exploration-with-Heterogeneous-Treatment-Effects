def Greedy(D, subgroups: list):
    if len(subgroups) <= D.k:
        return subgroups

    subgroups = sorted(subgroups, key=lambda x: x["t_est"], reverse=True)

    total_subgroups = len(subgroups)
    percentile_index = int(0.9 * total_subgroups)
    bottom_10th_percentile = subgroups[percentile_index]["t_est"]

    top_subs = subgroups[: D.k].copy()

    top_subs, max_t, max_op, min_score, min_score_i = D.compute_topK_scores(top_subs)

    theta = bottom_10th_percentile
    i = D.k + 1

    while i < total_subgroups and subgroups[i]["t_est"] > theta:
        score_si = D.compute_score_for_subgroup(
            top_subgroups=top_subs, s=subgroups[i], max_t=max_t, max_op=max_op
        )
        if score_si > min_score:
            top_subs.pop(min_score_i)
            top_subs.append(subgroups[i])
            top_subs, max_t, max_op, min_score, min_score_i = D.compute_topK_scores(
                top_subs
            )
        i += 1

    return top_subs
