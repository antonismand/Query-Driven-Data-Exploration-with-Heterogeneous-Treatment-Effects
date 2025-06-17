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

            if list(ct.subgroups.keys())[-1] != self.n_subgroups:
                raise ValueError("Subgroup size does not match last id")

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

        for s in final_recs:
            s.update(
                {
                    "Top-K Algorithm": self.name,
                    "variant": self.ct.algorithm + " " + self.name,
                    # "topK_execution_time": execution_time,
                    # "ct_execution_time": self.ct.online_time,
                    "Time": execution_time + self.ct.online_time,
                    "Total Subgroups": self.n_subgroups,
                    "K": len(recs),
                    "Diversity": self.get_diversity(recs),
                }
            )

            if self.D.mode in [71, 72]:
                s.update({"t_r": s["t_est"]})
            else:
                max_overlap = 0
                overlap = self.compute_overlap_for_sub(recs, s["id"], sub_in_subs=True)
                if overlap > max_overlap:
                    max_overlap = overlap

                t = self.D.CATE(s["condition"])
                t_r = self.D.CATE(s["final_condition"])
                t_r_error = abs(t_r - t)

                s.update(
                    {
                        "t": t,
                        "t_r": t_r,
                        "Overlap": overlap,
                        "Number of rows": self.D.n_rows(s["final_condition"]),
                        "t_error": abs(t - s["t_est"]),
                        "t_r_error": t_r_error,
                    }
                )

                if t_r_error > 0.15:
                    logger.warning(
                        f"{self.ct.algorithm} - {self.name} - |t_r-t|={t_r_error} are different. P: {self.D.p} Subgroup: {s['condition']}"
                    )

        if self.D.mode not in [71, 72]:
            for s in final_recs:
                s["Max Overlap"] = max_overlap

        return pd.DataFrame(final_recs)

    # def get_score(self, top_subs: list[int], id: int):  # ID should be in top_subs
    #     overlap = self.compute_overlap_for_sub(top_subs, id, sub_in_subs=True)
    #     return self.w * self.get_t(id) / self.max_t + (1 - self.w) * (1 - overlap)

    def is_valid(self, id: int):
        if self.ct.is_online:
            return True

        if not self.D.pp.includes(
            self.get_sub(id)["combined"]
        ):  # check if subgroup is a subset of P
            return False

        if id in self.valid_parents:
            return True

        r = self.get_sub(id)["final_condition"]
        df = self.D.execute(r)
        if df.shape[0] > self.min_rows:
            self.valid_parents.update(self.get_sub(id)["parents"])
            return True

        return False

    def get_worst_diversity(self, subs: list[int]):
        worst_diversity = 1
        worst_diversity_idx = -1
        for i, s in enumerate(subs):
            diversity = self.get_div_of_sub(sub=s, subs=subs)

            if diversity < worst_diversity:
                worst_diversity = diversity
                worst_diversity_idx = i

        return worst_diversity, worst_diversity_idx

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

            diversity = self.get_div_of_sub(s1, subs)

            score = self.w * self.get_t(s1) / self.max_t + (1 - self.w) * diversity
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

    def get_dissimilarity(self, s1: int, s2: int):
        return self.D.dissimilarity(
            self.get_sub(s1)["combined"], self.get_sub(s2)["combined"]
        )

    def get_div_of_sub(self, sub: int, subs: list[int]):
        total_diversity = 0
        n = 0
        for s in subs:
            if sub != s:
                total_diversity += self.get_dissimilarity(sub, s)
                n += 1
        return total_diversity / n

    def get_diversity(self, subs: list[int]):
        total_diversity = 0
        combs = 0
        for i, s1 in enumerate(subs):
            for s2 in subs[i + 1 :]:
                dissimilarity = self.get_dissimilarity(s1, s2)
                # sub1 = self.get_sub(s1)
                # sub2 = self.get_sub(s2)
                # logger.info(
                #     f"dissimilarity between {sub1['combined']} and {sub2['combined']}: {dissimilarity}"
                # )
                total_diversity += dissimilarity
                combs += 1
        return total_diversity / (self.k * (self.k - 1) / 2)

    def check_n_subs(self, subs: list[int]):
        if len(subs) < self.k:
            logger.warning(
                f"{self.ct.algorithm} - {self.name} - Subgroups < K. Returning []"
            )
            return False
        return True


class DiCoR(TopK):
    def __init__(
        self,
        alg: CT | CTP,
        min_diversity=params.TOPK.MIN_DIVERSITY,
    ):
        super().__init__(alg)
        self.min_diversity = min_diversity

    def scan(self):
        subs = sorted(self.ct.subgroups, key=lambda x: self.get_t(x), reverse=True)

        while len(subs) > 0 and self.is_valid(subs[0]) == False:
            subs.pop(0)

        if not self.check_n_subs(subs):
            return []

        recs = [subs[0]]

        for sub in subs[1:]:
            # logger.trace(f"Checking sub {sub}")
            if not self.is_valid(sub):
                logger.trace(f"Sub {sub} is not valid")
                continue

            if self.get_div_of_sub(sub=sub, subs=recs) > self.min_diversity:
                recs.append(sub)
                if len(recs) == self.k:
                    return recs
            else:
                logger.trace(
                    f"Diversity of candidate {sub} with selected recs is {round(self.get_div_of_sub(sub=sub, subs=recs),2)} < {self.min_diversity}"
                )

        return recs


class ReCoD(TopK):
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
        worst_diversity, worst_diversity_idx = self.get_worst_diversity(recs)

        for sub in subs[:percentile_index]:
            # logger.trace(f"Checking sub {sub}")
            if self.is_valid(sub):
                removed_sub = recs.pop(worst_diversity_idx)
                recs.append(sub)

                new_diversity, _ = self.get_worst_diversity(recs)

                if new_diversity < worst_diversity:
                    # logger.trace("Reverting", new_diversity, "with", worst_diversity)
                    recs.pop(-1)
                    recs.append(removed_sub)

                else:
                    worst_diversity, worst_diversity_idx = self.get_worst_diversity(
                        recs
                    )
                    logger.trace(
                        f"{self.name} - Added new candidate, new worst diversity: {worst_diversity} "
                    )

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


class Sort(TopK):
    def __init__(self, alg: CT):
        super().__init__(alg)

    def scan(self):
        subs = sorted(self.ct.subgroups, key=lambda x: self.get_t(x), reverse=True)
        recs = []
        while len(subs) > 0 and len(recs) != self.k:
            if self.is_valid(subs[0]):
                recs.append(subs[0])
            subs.pop(0)

        return recs


class DiRe(TopK):
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
            # logger.trace(f"Checking sub {sub}")
            if self.is_valid(sub):
                removed_sub = recs.pop(min_score_idx)
                recs.append(sub)

                _, new_score, _ = self.get_scores_for_subs(recs)

                if new_score < min_score:
                    # logger.trace(f"Reverting {new_score} with {min_score}")
                    recs.pop(-1)
                    recs.append(removed_sub)
                else:
                    logger.debug(
                        f"{self.name} Old score: {min_score}, New score {new_score}"
                    )
                    _, min_score, min_score_idx = self.get_scores_for_subs(recs)

        return recs


class ExhaustiveDiRe(TopK):
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


class ExhaustiveDiCoR(TopK):
    def __init__(
        self,
        alg: CT,
        min_diversity=params.TOPK.MIN_DIVERSITY,
    ):
        super().__init__(alg)
        self.min_diversity = min_diversity

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
            if self.get_diversity(candidates) > self.min_diversity:
                cate = sum([self.get_t(c) for c in candidates])
                if cate > best_cate:
                    logger.debug(
                        f"NEW CATE: {cate / self.k}, previous: {best_cate / self.k}"
                    )
                    best_cate = cate
                    best_subs = candidates[:]

        return best_subs


class ExhaustiveReCoD(TopK):
    def __init__(self, alg: CT):
        super().__init__(alg)

    def scan(self):
        best_diversity = 0

        subs = sorted(self.ct.subgroups, key=lambda x: self.get_t(x), reverse=True)
        valid_subs = [s for s in subs if self.is_valid(s)]

        logger.info(f"Valid subgroups: {len(valid_subs)} out of {len(subs)}")
        valid_subs = valid_subs[0 : params.TOPK.EXHAUSTIVE_TOPK]

        for candidates in tqdm(
            itertools.combinations(valid_subs, self.k),
            total=comb(len(valid_subs), self.k),
        ):
            diversity = self.get_diversity(candidates)

            if diversity > best_diversity:
                logger.debug(f"NEW diversity:{diversity} previous: {best_diversity}")
                best_diversity = diversity
                best_subs = candidates[:]

        return best_subs


class LoRe(TopK):
    def __init__(self, alg: CT):
        super().__init__(alg)

    def scan(self):
        levels = {}

        for id, sub in self.ct.subgroups.items():
            if sub["Depth"] not in levels:
                levels[sub["Depth"]] = []
            levels[sub["Depth"]].append(id)

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


main_no_random = [LoRe, DiRe, DiCoR, ReCoD]
main_competitors = main_no_random + [Random]
exhaustive_comparison = [
    ExhaustiveDiRe,
    DiRe,
    ExhaustiveDiCoR,
    DiCoR,
    ExhaustiveReCoD,
    ReCoD,
]
