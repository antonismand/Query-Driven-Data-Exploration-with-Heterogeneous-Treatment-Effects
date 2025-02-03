from copy import deepcopy
import itertools
from math import comb
from time import time

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
from trees import params
from trees.causal_tree import CT, CTP


class TopK:
    def __init__(self, ct: CT | CTP):
        if ct is not None:
            self.n_subgroups = len(ct.subgroups)
            self.ct = ct

            self.max_t = 0
            self.D = ct.D
            self.name = type(self).__name__

            self.valid_parents = set()
            self.op_matrix = np.ones((self.n_subgroups + 1, self.n_subgroups + 1)) * -1

            for sub in ct.subgroups.values():
                sub["final_condition"] = ct.D.p + " AND " + sub["condition"]

    def scan(self):
        pass

    def get_topK(self, k=params.TOPK.K, min_rows=params.TOPK.N_MIN_ROWS):
        self.k = k
        self.min_rows = min_rows

        if self.D.p is None:
            raise ValueError("no P given")

        start = time()

        recs = self.scan()  # ids

        execution_time = round(time() - start, 3)

        if len(recs) < self.k:
            logger.warning(
                f"{self.ct.algorithm} with {self.name} - could not yield K={self.k} subgroups"
            )
            return pd.DataFrame([])

        final_recs: list[dict] = [self.copy_sub(s) for s in recs]

        max_overlap = 0

        for s in final_recs:
            overlap = self.compute_overlap_for_sub(recs, s["id"], sub_in_subs=True)
            if overlap > max_overlap:
                max_overlap = overlap
            s.update(
                {
                    "t": self.D.CATE(s["condition"]),
                    "t_r": self.D.CATE(s["final_condition"]),
                    "topK_algorithm": self.name,
                    "variant": self.ct.algorithm + " " + self.name,
                    "topK_execution_time": execution_time,
                    "ct_execution_time": self.ct.online_time,
                    "total_execution_time": execution_time + self.ct.online_time,
                    "total_subgroups": self.n_subgroups,
                    "rows": self.D.n_rows(s["final_condition"]),
                    "overlap": overlap,
                    "K": len(recs),
                }
            )
            t_r_error = abs(s["t_r"] - s["t"])
            if t_r_error > 0.15:
                logger.warning(
                    f"{self.ct.algorithm} - {self.name} - |t_r-t|={t_r_error} are different. P: {self.D.p} Subgroup: {s['condition']}"
                )

            s.update(
                {
                    "t_error": abs(s["t"] - s["t_est"]),
                    "t_r_error": t_r_error,
                }
            )

        for s in final_recs:
            s["max_overlap"] = max_overlap

        return pd.DataFrame(final_recs)

    def get_score(self, top_subs: list[int], id: int):  # ID should be in top_subs
        overlap = self.compute_overlap_for_sub(top_subs, id, sub_in_subs=True)
        return self.w * self.get_t(id) / self.max_t + (1 - self.w) * (1 - overlap)

    def is_valid(self, id: int):
        if self.ct.is_online:
            return True

        if not self.D.pp.includes(
            self.get_sub(id)["combined"]
        ):  # check if subgroup is a subset of P
            return False

        # TODO probably all parents are invalid too

        if id in self.valid_parents:
            return True

        r = self.get_sub(id)["final_condition"]
        df = self.D.execute(r)
        if df.shape[0] > self.min_rows:
            self.valid_parents.update(self.get_sub(id)["parents"])
            return True

        return False

    def get_worst_overlap(self, subs: list[int]):
        worst_overlap = 0
        worst_overlap_idx = -1
        for i, s1 in enumerate(subs):
            overlap = sum([self.get_J(s1, s2) for s2 in subs])

            if overlap > worst_overlap:
                worst_overlap = overlap
                worst_overlap_idx = i

        return worst_overlap / (self.k - 1), worst_overlap_idx

    def are_groups_valid(self, groups: list, threshold: float):
        for s1 in groups:
            for s2 in groups:
                if self.get_J(s1, s2) > threshold:
                    return False
        return True

    def compute_overlap_for_sub(self, subs: list[int], id: int, sub_in_subs=True):
        div = self.k - 1 if sub_in_subs else self.k
        overlap = sum([self.get_J(id, s2) for s2 in subs]) / div
        return overlap

    def get_scores_for_subs(self, subs: list[int]):

        total_score = 0
        min_score = 999999999
        min_score_idx = -1

        for i, s1 in enumerate(subs):
            overlap = sum([self.get_J(s1, s2) for s2 in subs])
            overlap /= self.k - 1

            score = self.w * self.get_t(s1) / self.max_t + (1 - self.w) * (1 - overlap)
            total_score += score
            if score < min_score:
                min_score = score
                min_score_idx = i

        total_score /= self.k
        return total_score, min_score, min_score_idx

    def get_sub(self, id: int):
        return self.ct.subgroups[id]

    def copy_sub(self, id: int):
        return deepcopy(self.ct.subgroups[id])

    def get_t(self, id: int):
        return self.ct.subgroups[id]["t_est"]

    # def get_J(self, id1: int, id2: int):
    #     return self.op_matrix[id1][id2]

    def get_J(self, id1: int, id2: int):
        if self.op_matrix[id1][id2] != -1:
            return self.op_matrix[id1][id2]

        sub1 = self.get_sub(id1)
        sub2 = self.get_sub(id2)

        if id1 in sub2["parents"] or id2 in sub1["parents"]:
            overlap = self.D.jaccard_over_preds(
                sub1["final_condition"], sub2["final_condition"]
            )
            self.op_matrix[id1][id2] = overlap
            self.op_matrix[id2][id1] = overlap
            return overlap

        self.op_matrix[id1][id2] = 0
        return 0

    def check_n_subs(self, subs: list[int]):
        if len(subs) < self.k:
            logger.warning(
                f"{self.ct.algorithm} - {self.name} - Subgroups < K. Returning []"
            )
            return False
        return True


class OptRes(TopK):
    def __init__(
        self,
        alg: CT | CTP,
        max_overlap=params.TOPK.MAX_PAIRWISE_OVERLAP,
    ):
        super().__init__(alg)
        self.max_overlap = max_overlap

    def scan(self):
        subs = sorted(self.ct.subgroups, key=lambda x: self.get_t(x), reverse=True)

        while len(subs) > 0 and self.is_valid(subs[0]) == False:
            subs.pop(0)

        if not self.check_n_subs(subs):
            return []

        recs = [subs[0]]

        for sub in subs[1:]:
            logger.trace(f"Checking sub {sub}")
            if not self.is_valid(sub):
                logger.trace(f"Sub {sub} is not valid")
                continue

            accepted = True
            for selected in recs:
                if self.get_J(selected, sub) > self.max_overlap:
                    logger.trace(
                        f"Overlap of candidate {sub} with selected {selected} is {round(self.get_J(selected, sub),2)} > {self.max_overlap}"
                    )
                    accepted = False
                    break

            if accepted:
                recs.append(sub)
                if len(recs) == self.k:
                    return recs

        return recs


class OptOve(TopK):
    def __init__(self, alg: CT, percentile=params.TOPK.PERCENTILE):
        super().__init__(alg)
        self.percentile = percentile

    def scan(self):
        subs = sorted(self.ct.subgroups, key=lambda x: self.get_t(x), reverse=True)

        recs = []
        while len(subs) > 0 and len(recs) != self.k:
            if self.is_valid(subs[0]):
                recs.append(subs[0])
            subs.pop(0)

        if not self.check_n_subs(subs):
            return []

        percentile_index = int(self.percentile * self.n_subgroups)
        worst_overlap, worst_overlap_idx = self.get_worst_overlap(recs)

        for sub in subs[self.k + 1 : percentile_index]:
            logger.trace(f"Checking sub {sub}")
            if self.is_valid(sub):
                removed_sub = recs.pop(worst_overlap_idx)
                recs.append(sub)

                new_overlap, _ = self.get_worst_overlap(recs)

                if new_overlap == 0:
                    break

                if new_overlap > worst_overlap:
                    logger.trace("Reverting", new_overlap, "with", worst_overlap)
                    recs.pop(-1)
                    recs.append(removed_sub)

                else:
                    worst_overlap, worst_overlap_idx = self.get_worst_overlap(recs)
                    logger.trace("New worst overlap", worst_overlap)

        return recs


class Random(TopK):
    def __init__(self, alg: CT):
        super().__init__(alg)

    def scan(self):
        subs = list(self.ct.subgroups.keys())
        recs = []
        while (len(recs) < self.k) and (len(subs) > 0):
            sub = subs.pop(np.random.randint(0, len(subs)))
            if self.is_valid(sub):
                recs.append(sub)

        return recs


class ResOve(TopK):
    def __init__(self, alg: CT, w=params.TOPK.W):
        super().__init__(alg)
        self.w = w

    def scan(self):
        # subs = sorted(self.ct.subgroups, key=lambda x: self.get_t(x), reverse=True)
        subs = list(self.ct.subgroups.keys())[::-1]

        recs = []
        while len(subs) > 0 and len(recs) != self.k:
            if self.is_valid(subs[0]):
                recs.append(subs[0])
            subs.pop(0)

        if not self.check_n_subs(subs):
            return []

        self.max_t = self.get_t(recs[0])
        _, min_score, min_score_idx = self.get_scores_for_subs(recs)

        for sub in subs:
            logger.trace(f"Checking sub {sub}")
            if self.is_valid(sub):
                removed_sub = recs.pop(min_score_idx)
                recs.append(sub)

                _, new_score, _ = self.get_scores_for_subs(recs)

                if new_score < min_score:
                    logger.trace(f"Reverting {new_score} with {min_score}")
                    recs.pop(-1)
                    recs.append(removed_sub)
                else:
                    _, min_score, min_score_idx = self.get_scores_for_subs(recs)
                    logger.debug(f"New score: {new_score}, New min score {min_score}")

        return recs


class ExhaustiveResOve(TopK):
    def __init__(self, alg: CT, w=params.TOPK.W):
        super().__init__(alg)
        self.w = w

    def scan(self):
        best_score = 0
        best_subs = []

        subs = sorted(self.ct.subgroups, key=lambda x: self.get_t(x), reverse=True)
        valid_subs = [s for s in subs if self.is_valid(s)]
        self.max_t = self.get_t(valid_subs[0])

        logger.info(f"Valid subgroups: {len(valid_subs)} out of {len(subs)}")
        valid_subs = valid_subs[0 : params.TOPK.EXHAUSTIVE_TOPK]

        for candidates in tqdm(
            itertools.combinations(valid_subs, self.k),
            total=comb(len(valid_subs), self.k),
        ):
            score, _, _ = self.get_scores_for_subs(candidates)
            if score > best_score:
                logger.debug(f"NEW score:{score} previous: {best_score}")
                best_score = score
                best_subs = candidates[:]

        return best_subs


class ExhaustiveOptRes(TopK):
    def __init__(
        self,
        alg: CT,
        max_overlap=params.TOPK.MAX_PAIRWISE_OVERLAP,
    ):
        super().__init__(alg)
        self.max_overlap = max_overlap

    def scan(self):
        best_cate = 0
        best_subs = []

        subs = sorted(self.ct.subgroups, key=lambda x: self.get_t(x), reverse=True)
        valid_subs = [s for s in subs if self.is_valid(s)]

        logger.info(f"Valid subgroups: {len(valid_subs)} out of {len(subs)}")
        valid_subs = valid_subs[0 : params.TOPK.EXHAUSTIVE_TOPK]

        for candidates in tqdm(
            itertools.combinations(valid_subs, self.k),
            total=comb(len(valid_subs), self.k),
        ):

            if self.are_groups_valid(candidates, self.max_overlap):
                cate = sum([self.get_t(c) for c in candidates])
                if cate > best_cate:
                    logger.debug(
                        f"NEW CATE: {cate / self.k}, previous: {best_cate / self.k}"
                    )
                    best_cate = cate
                    best_subs = candidates[:]

        return best_subs


# class ExhaustiveOptOve(TopK):
#     def __init__(self, alg: CT):
#         super().__init__(alg)

#     def scan(self):
#         best_overlap = 999999

#         subs = sorted(self.ct.subgroups, key=lambda x: self.get_t(x), reverse=True)
#         valid_subs = [s for s in subs if self.is_valid(s)]

#         logger.info(f"Valid subgroups: {len(valid_subs)} out of {len(subs)}")
#         valid_subs = valid_subs[0 : params.TOPK.EXHAUSTIVE_TOPK]

#         for candidates in tqdm(
#             itertools.combinations(valid_subs, self.k),
#             total=comb(len(valid_subs), self.k),
#         ):
#             for s1 in candidates:
#                 overlap = sum([self.get_J(s1, s2) for s2 in candidates])

#             if overlap < best_overlap:
#                 logger.debug(f"NEW overlap:{overlap} previous: {best_overlap}")
#                 best_overlap = overlap
#                 best_subs = candidates[:]

#         return best_subs


class NoOve(TopK):
    def __init__(self, alg: CT):
        super().__init__(alg)

    def scan(self):
        levels = {}

        for id, sub in self.ct.subgroups.items():
            if sub["depth"] not in levels:
                levels[sub["depth"]] = []
            levels[sub["depth"]].append(id)

        best_cate = 0
        best_ids = []
        for level in dict(sorted(levels.items(), reverse=True)):
            lvlsubgroups = levels[level]
            if len(lvlsubgroups) >= self.k:
                lvlsubgroups = sorted(
                    lvlsubgroups, key=lambda x: self.get_t(x), reverse=True
                )
                cate = 0
                valid = 0
                valid_ids = []
                for sub in lvlsubgroups:
                    if self.is_valid(sub):
                        cate += self.get_t(sub)
                        valid += 1
                        valid_ids.append(sub)
                        if valid == self.k:
                            break

                if valid < self.k:
                    continue
                logger.debug(f"Level {level} AVG(CATE): {round(cate / self.k, 2)}")
                if cate > best_cate:
                    best_cate = cate
                    best_ids = valid_ids[:]

        return best_ids


# class LastLevel(TopK):
#     def __init__(self, alg: CT):
#         super().__init__(alg)

#     def scan(self):
#         levels = {}
#         level = 0

#         for id in self.valid_subs:
#             sub = self.get_sub(id)
#             if sub["depth"] not in levels:
#                 levels[sub["depth"]] = []
#                 level = max(level, sub["depth"])
#             levels[sub["depth"]].append(sub)

#         level_keys = sorted(levels.keys())

#         while len(levels[level]) < self.k and len(level_keys) > 0:
#             level = level_keys.pop()
#             # print("Trying level", level)

#         levels[level] = sorted(levels[level], key=lambda x: x["t_est"], reverse=True)

#         return [c["id"] for c in levels[level][0 : self.k]]


main_no_random = [OptRes, ResOve, NoOve, OptOve]
main_competitors = main_no_random + [Random]
exhaustive = [ExhaustiveResOve, ExhaustiveOptRes]
