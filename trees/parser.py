import sqlglot
from sqlglot.expressions import And, Or, Condition, Paren, GT, LT, GTE, LTE, EQ, Between


class Predicate:
    def __init__(self, p, D):
        self.conditions = []
        self.or_conditions = []
        self.D = D
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
            return {left: (low, high)}

        elif isinstance(expression, Condition):
            left = str(expression.left)
            right = float(str(expression.right))

            if isinstance(expression, GTE):
                return {left: (right, self.D.min_max[left][1])}
            elif isinstance(expression, GT):
                return {left: (right, self.D.min_max[left][1])}
            elif isinstance(expression, LTE):
                return {left: (self.D.min_max[left][0], right)}
            elif isinstance(expression, LT):
                return {left: (self.D.min_max[left][0], right)}
            elif isinstance(expression, (EQ)):
                return {left: (right, right)}
        return str(expression)

    def includes(self, combined: dict[str, tuple]):
        for cond in self.conditions:
            if len(cond) == 1:
                for attr, interval in cond[0].items():
                    if attr in self.D.hte_features:
                        if attr in combined and not self.D.is_subset(
                            interval, combined[attr]
                        ):
                            # print(f"{self.combined[attr]} not in P: {interval}")
                            return False
                        elif attr not in combined:
                            return False
            else:
                any_satisfied = False
                any_attr_with_HTE = False
                for sub_cond in cond:
                    for attr, interval in sub_cond.items():
                        if attr in self.D.hte_features:
                            any_attr_with_HTE = True
                            if attr in combined and self.D.is_subset(
                                interval, combined[attr]
                            ):
                                any_satisfied = True
                                break

                if not any_satisfied and any_attr_with_HTE:
                    return False

        return True


# def format_interval(interval: Interval):
#     lower_bound = "-∞" if interval.start == -oo else round(interval.start, 3)
#     upper_bound = "∞" if interval.end == oo else round(interval.end, 3)
#     lower_bracket = "[" if interval.left_open is False else "("
#     upper_bracket = "]" if interval.right_open is False else ")"
#     return f"{lower_bracket}{lower_bound}, {upper_bound}{upper_bracket}"
