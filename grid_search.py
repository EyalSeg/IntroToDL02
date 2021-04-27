import itertools
import operator

from math import inf


def tune(func, params_dict, direction="minimize"):
    param_sets = (dict(zip(params_dict.keys(), x)) for x in itertools.product(*params_dict.values()))

    if direction.startswith("min"):
        best_score = inf
        better_than = operator.lt
    else:
        best_score = -inf
        better_than = operator.gt

    best_params = None

    for params in param_sets:
        score = func(**params)

        if better_than(score, best_score):
            best_score = score
            best_params = params

    return best_params, best_score


