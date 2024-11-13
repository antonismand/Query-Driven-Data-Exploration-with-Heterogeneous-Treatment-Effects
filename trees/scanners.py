from copy import deepcopy
import itertools
from math import comb
from time import time

import pandas as pd
from tqdm import tqdm


class Scanner:
    def __init__(self, valid_subs: list, op_matrix: list, D):

        self.valid_subs = deepcopy(valid_subs)
        self.n_subs = len(valid_subs)
        self.op_matrix = op_matrix
        self.top_subs = []
        self.max_t = 0
        self.D = D
        self.name = type(self).__name__

    def scan(self):
        pass

    def get_topK(self, w=0.5, k=5):
        if self.D.p is None:
            raise ValueError("no P given")

        self.w = w
        self.k = k

        start = time()
        self.scan()
        scan_time = round(time() - start, 3)

        for s in self.top_subs:
            s["t"] = self.D.CATE(s["condition"])
            s["scan_method"] = self.name
            s["scan_time"] = scan_time
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

    def get_worst_overlap(self, subs: list):
        worst_overlap = 0
        worst_overlap_idx = -1
        for i, s1 in enumerate(subs):
            overlap = sum([self.op_matrix[s1["id"]][s2["id"]] for s2 in subs])

            if overlap > worst_overlap:
                worst_overlap = overlap
                worst_overlap_idx = i

        return worst_overlap / (self.k - 1), worst_overlap_idx

    def are_groups_valid(self, groups: list, threshold: float):
        for s1 in groups:
            for s2 in groups:
                if self.op_matrix[s1["id"]][s2["id"]] > threshold:
                    return False
        return True

    # def compute_overlap_for_sub(self, top_subs: list, s: dict):
    #     overlap = (
    #         sum([self.op_matrix[top_sub["id"]][s["id"]] for top_sub in top_subs])
    #         / self.k
    #     )
    #     return overlap

    def get_scores_for_subs(self, subs: list):

        total_score = 0
        min_score = 999999999
        min_score_idx = -1

        for i, s1 in enumerate(subs):
            overlap = sum([self.op_matrix[s1["id"]][s2["id"]] for s2 in subs])
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
        # print("Total score:", total_score / self.k)


class ConstrainedJ(Scanner):
    def __init__(self, valid_subs: list, op_matrix: list, D, max_overlap=0.5):
        super().__init__(valid_subs, op_matrix, D)
        self.max_overlap = max_overlap

    def scan(self):
        if self.n_subs <= self.k:
            return self.valid_subs

        subs = sorted(self.valid_subs, key=lambda x: x["t_est"], reverse=True)
        self.max_t = subs[0]["t_est"]

        self.top_subs = deepcopy([subs[0]])

        for i, sub in enumerate(subs[1:]):
            # print("Checking sub", i)
            accepted = True
            for j, selected in enumerate(self.top_subs):
                if self.op_matrix[selected["id"]][sub["id"]] > self.max_overlap:
                    # print(
                    #     "overlap of sub",
                    #     i,
                    #     "with selected",
                    #     j,
                    #     self.op_matrix[selected["id"]][sub["id"]],
                    #     ">",
                    #     self.max_overlap,
                    # )
                    accepted = False
                    break

            if accepted:
                self.top_subs.append(sub)
                if len(self.top_subs) == self.k:
                    return self.top_subs

        if len(self.top_subs) < self.k:
            raise ValueError(
                f"Not enough valid subgroups to satisfy max_overlap= {self.max_overlap}"
            )
        return self.top_subs


class ConstrainedT(Scanner):
    def __init__(self, valid_subs: list, op_matrix: list, D, percentile=0.7):
        super().__init__(valid_subs, op_matrix, D)
        self.percentile = percentile

    def scan(self):
        if self.n_subs <= self.k:
            return self.valid_subs

        subs = sorted(self.valid_subs, key=lambda x: x["t_est"], reverse=True)
        self.max_t = subs[0]["t_est"]

        percentile_index = int(self.percentile * self.n_subs)

        self.top_subs = deepcopy(subs[: self.k])

        worst_overlap, worst_overlap_idx = self.get_worst_overlap(self.top_subs)

        for sub in subs[self.k + 1 : percentile_index]:
            removed_sub = self.top_subs.pop(worst_overlap_idx)
            self.top_subs.append(sub)

            new_overlap, _ = self.get_worst_overlap(self.top_subs)

            if new_overlap > worst_overlap:
                # print("Reverting", new_worst_overlap, "with", worst_overlap)
                self.top_subs.pop(-1)
                self.top_subs.append(removed_sub)

            else:
                worst_overlap, worst_overlap_idx = self.get_worst_overlap(self.top_subs)
                # print("New worst overlap", worst_overlap)


class Weighted(Scanner):
    def __init__(self, valid_subs: list, op_matrix: list, D, percentile=0.7):
        super().__init__(valid_subs, op_matrix, D)
        self.percentile = percentile

    def scan(self):
        if self.n_subs <= self.k:
            return self.valid_subs

        subs = sorted(self.valid_subs, key=lambda x: x["t_est"], reverse=True)
        self.max_t = subs[0]["t_est"]

        percentile_index = int(self.percentile * self.n_subs)

        self.top_subs = deepcopy(subs[: self.k])

        _, min_score, min_score_idx = self.get_scores_for_subs(self.top_subs)

        for sub in subs[self.k + 1 : percentile_index]:
            removed_sub = self.top_subs.pop(min_score_idx)
            self.top_subs.append(sub)

            _, new_score, _ = self.get_scores_for_subs(self.top_subs)

            if new_score < min_score:
                # print("Reverting", new_score, "with", min_score)
                self.top_subs.pop(-1)
                self.top_subs.append(removed_sub)
            else:
                _, min_score, min_score_idx = self.get_scores_for_subs(self.top_subs)
                # print("New min score", min_score)


class ExhaustiveWeighted(Scanner):
    def __init__(self, valid_subs: list, op_matrix: list, D, percentile=0.7):
        super().__init__(valid_subs, op_matrix, D)
        self.percentile = percentile

    def scan(self):
        best_score = 0

        subs = sorted(self.valid_subs, key=lambda x: x["t_est"], reverse=True)
        self.max_t = subs[0]["t_est"]

        percentile_index = int(self.percentile * self.n_subs)
        pruned_subgroups = subs[:percentile_index]

        for candidates in tqdm(
            itertools.combinations(pruned_subgroups, self.k),
            total=comb(len(pruned_subgroups), self.k),
        ):
            score, _, _ = self.get_scores_for_subs(candidates)
            if score > best_score:
                # print("NEW score:", score, "previous:", best_score)
                best_score = score
                self.top_subs = candidates


class ExhaustiveT(Scanner):
    def __init__(
        self, valid_subs: list, op_matrix: list, D, percentile=0.7, max_overlap=0.5
    ):
        super().__init__(valid_subs, op_matrix, D)
        self.percentile = percentile
        self.max_overlap = max_overlap

    def scan(self):
        best_cate = 0

        subs = sorted(self.valid_subs, key=lambda x: x["t_est"], reverse=True)
        self.max_t = subs[0]["t_est"]

        percentile_index = int(self.percentile * self.n_subs)
        pruned_subgroups = subs[:percentile_index]

        for candidates in tqdm(
            itertools.combinations(pruned_subgroups, self.k),
            total=comb(len(pruned_subgroups), self.k),
        ):

            if self.are_groups_valid(candidates, self.max_overlap):
                cate = sum([c["t_est"] for c in candidates])
                if cate > best_cate:
                    print("NEW CATE:", cate / self.k, "previous:", best_cate / self.k)
                    best_cate = cate
                    self.top_subs = candidates


class ExhaustiveOverlap(Scanner):
    def __init__(self, valid_subs: list, op_matrix: list, D, percentile=0.7):
        super().__init__(valid_subs, op_matrix, D)
        self.percentile = percentile

    def scan(self):
        best_overlap = 999999

        subs = sorted(self.valid_subs, key=lambda x: x["t_est"], reverse=True)
        self.max_t = subs[0]["t_est"]

        percentile_index = int(self.percentile * self.n_subs)
        pruned_subgroups = subs[:percentile_index]

        for candidates in tqdm(
            itertools.combinations(pruned_subgroups, self.k),
            total=comb(len(pruned_subgroups), self.k),
        ):
            for s1 in candidates:
                overlap = sum([self.op_matrix[s1["id"]][s2["id"]] for s2 in candidates])

            if overlap < best_overlap:
                print("NEW overlap:", overlap, "previous:", best_overlap)
                best_overlap = overlap
                self.top_subs = candidates

            if best_overlap == 0:
                return


greedy_scanners = [ConstrainedJ, ConstrainedT, Weighted]
all_scanners = greedy_scanners + [ExhaustiveWeighted, ExhaustiveT, ExhaustiveOverlap]
