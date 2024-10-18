import re
import sqlglot
from sqlglot.expressions import And, Or, Condition, Paren, GT, LT, GTE, LTE, EQ, Between
from sympy import Interval, oo


class Predicate:
    def __init__(self, p):
        self.conditions = []
        self.or_conditions = []
        parsed = sqlglot.parse_one(f"select * from x where {p}")

        if not "where" in parsed.args:
            raise Exception("No WHERE clause found")

        self.traverse_conditions(parsed.args["where"].this)

        # for i, cond in enumerate(self.conditions, 1):
        #     print(f"Requirement #{i}: {cond}")

    def traverse_conditions(self, expression):
        if isinstance(expression, And):
            self.traverse_conditions(expression.left)
            self.traverse_conditions(expression.right)
        elif isinstance(expression, Or):
            self.or_conditions = []

            self.collect_or_conditions(expression)
            self.conditions.append(self.or_conditions)
        elif isinstance(expression, Paren):
            self.traverse_conditions(expression.this)
        else:
            self.conditions.append([self.parse_condition(expression)])

    def collect_or_conditions(self, expr):
        if isinstance(expr, Or):
            self.collect_or_conditions(expr.left)
            self.collect_or_conditions(expr.right)
        else:
            self.or_conditions.append(self.parse_condition(expr))

    def parse_condition(self, expression):
        if isinstance(expression, Between):
            left = str(expression.this)
            low = float(str(expression.args["low"]))
            high = float(str(expression.args["high"]))
            return {left: Interval(low, high)}

        elif isinstance(expression, Condition):
            left = str(expression.left)
            right = float(str(expression.right))

            if isinstance(expression, GTE):
                return {left: Interval(right, oo)}
            elif isinstance(expression, GT):
                return {left: Interval.Lopen(right, oo)}
            elif isinstance(expression, LTE):
                return {left: Interval(-oo, right)}
            elif isinstance(expression, LT):
                return {left: Interval.Ropen(-oo, right)}
            elif isinstance(expression, (EQ)):
                return {left: Interval(right, right)}
        return str(expression)

    def includes(self, combined: dict[str, Interval]):
        for cond in self.conditions:
            if len(cond) == 1:
                for key, interval in cond[0].items():
                    if key in combined and not combined[key].is_proper_subset(interval):
                        # print(f"{self.combined[key]} not in P: {interval}")
                        return False
            else:
                any_satisfied = False
                any_key_exists = False
                for sub_cond in cond:
                    for key, interval in sub_cond.items():
                        if key in combined:
                            any_key_exists = True
                            if combined[key].is_proper_subset(interval):
                                any_satisfied = True
                                break

                if not any_satisfied and any_key_exists:
                    return False

        return True


def format_interval(interval: Interval):
    lower_bound = "-∞" if interval.start == -oo else round(interval.start, 3)
    upper_bound = "∞" if interval.end == oo else round(interval.end, 3)
    lower_bracket = "[" if interval.left_open is False else "("
    upper_bracket = "]" if interval.right_open is False else ")"
    return f"{lower_bracket}{lower_bound}, {upper_bound}{upper_bracket}"


if __name__ == "__main__":
    predicates_to_test = [
        "a>5 AND b>10",
        "(f1 > 1 OR f2 < 3) AND f4 > 3 AND f5 > 3 AND f1 between 1 and 2 AND (f0 < 4 OR f0 > 5 OR f3 < 3)",
    ]

    for test in predicates_to_test:
        q = "select * from x where " + test
        parsed = Predicate(q)
        print("------", test, "------")
        for i, req in enumerate(parsed.conditions, 1):
            formatted_conditions = []
            for condition in req:
                for feature, interval in condition.items():
                    formatted_conditions.append(
                        f"{feature} ∈ {format_interval(interval)}"
                    )
            joined_conditions = " OR ".join(formatted_conditions)
            pprint = f"Requirement #{i}: {joined_conditions}"
            print(pprint)
