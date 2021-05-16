import itertools
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np


def tune(func, params_dict, direction="minimize", workers=1):
    param_sets = list((dict(zip(params_dict.keys(), x)) for x in itertools.product(*params_dict.values())))
    call = lambda args: func(**args)

    if workers > 1:
        scores = []
        executor = ThreadPoolExecutor(max_workers=workers)
        results_gen = executor.map(call, param_sets)

        for res in results_gen:
            scores.append(res)
    else:
        scores = list(map(call, param_sets))

    if direction.startswith("min"):
        idx = np.argmin(scores)
    else:
        idx = np.argmax(scores)

    return param_sets[idx], scores[idx]




