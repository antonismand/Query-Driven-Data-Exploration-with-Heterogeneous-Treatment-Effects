from copy import deepcopy
import itertools
from math import comb
from time import time

import pandas as pd
from tqdm import tqdm


class Scanner:
    def __init__(self, valid_subs: list, op_matrix: list, D):
        if D.p is None:
            raise ValueError("no P given")

        self.valid_subs = deepcopy(valid_subs)
        self.n_subs = len(valid_subs)
        self.op_matrix = op_matrix
        self.top_subs = []
        self.max_t = 0
        self.D = D

    def scan(self):
        pass

    def get_topK(self, w=0.5, k=5):
        self.w = w
        self.k = k

        start = time()
        self.scan()
        scan_time = round(time() - start, 2)

        for s in self.top_subs:
            s["t"] = self.D.CATE(s["condition"])
            s["scan_method"] = type(self).__name__
            s["scan_time"] = scan_time
            s["algorithm"] += f" ({type(self).__name__})"
            s["valid_options"] = self.n_subs
            # s['true_score'] = w * s["t"] / self.max_t + (1 - w) * s["overlap_penalty"]

        self.compute_scores_for_top_subs()
        return pd.DataFrame(self.top_subs)

    def get_score_for_sub(self, top_subs: list, s: dict):
        overlap = (
            sum([self.op_matrix[top_sub["id"]][s["id"]] for top_sub in top_subs])
            / self.k
        )
        return self.w * s["t_est"] / self.max_t + (1 - self.w) * (1 - overlap)

    def get_scores_for_subs(self, subs: list):

        total_score = 0
        min_score = 999999999
        min_score_idx = -1

        for i, s1 in enumerate(subs):
            overlap = 0
            for j, s2 in enumerate(subs):
                overlap += self.op_matrix[s1["id"]][s2["id"]]

            overlap /= self.k - 1
            score = self.w * s1["t_est"] / self.max_t + (1 - self.w) * (1 - overlap)
            total_score += score
            if score < min_score:
                min_score = score
                min_score_idx = i

        total_score /= self.k
        return total_score, min_score, min_score_idx

    def compute_scores_for_top_subs(self):
        total_score = 0
        for s1 in self.top_subs:
            overlap = 0
            for s2 in self.top_subs:
                overlap += self.op_matrix[s1["id"]][s2["id"]]

            overlap /= self.k - 1
            s1["overlap"] = overlap
            s1["score"] = self.w * s1["t_est"] / self.max_t + (1 - self.w) * (
                1 - overlap
            )
            total_score += s1["score"]
        print("Total score:", total_score / self.k)


class Greedy(Scanner):
    def __init__(self, valid_subs: list, op_matrix: list, D):
        super().__init__(valid_subs, op_matrix, D)

    def scan(self):
        if self.n_subs <= self.k:
            return self.valid_subs

        subs = sorted(self.valid_subs, key=lambda x: x["t_est"], reverse=True)
        self.max_t = subs[0]["t_est"]

        percentile_index = int(0.9 * self.n_subs)

        self.top_subs = deepcopy(subs[: self.k])

        _, min_score, min_score_idx = self.get_scores_for_subs(self.top_subs)

        for sub in subs[self.k + 1 : percentile_index]:
            sub_score = self.get_score_for_sub(top_subs=self.top_subs, s=sub)

            if sub_score > min_score:
                print("Replacing", min_score, "with", sub_score)
                self.top_subs.pop(min_score_idx)
                self.top_subs.append(sub)

                _, min_score, min_score_idx = self.get_scores_for_subs(self.top_subs)
                print("New min score", min_score)


class Exhaustive(Scanner):
    def __init__(self, valid_subs: list, op_matrix: list, D):
        super().__init__(valid_subs, op_matrix, D)

    def scan(self):
        best_score = 0

        subs = sorted(self.valid_subs, key=lambda x: x["t_est"], reverse=True)
        self.max_t = subs[0]["t_est"]

        percentile_index = int(0.8 * self.n_subs)
        pruned_subgroups = subs[:percentile_index]

        for candidates in tqdm(
            itertools.combinations(pruned_subgroups, self.k),
            total=comb(len(pruned_subgroups), self.k),
        ):
            score, _, _ = self.get_scores_for_subs(candidates)
            if score > best_score:
                print("NEW score:", score, "previous:", best_score)
                best_score = score
                self.top_subs = candidates
