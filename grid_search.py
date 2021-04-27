import itertools
import numpy as np
import marshal
import types

from multiprocessing import Pool


def tune(func, params_dict, direction="minimize", workers=1):
    param_sets = list((dict(zip(params_dict.keys(), x)) for x in itertools.product(*params_dict.values())))
    args = [(marshal.dumps(func.__code__), parameters) for parameters in param_sets]

    pool = Pool(workers)
    results = pool.map(evalaute, args)

    if direction.startswith("min"):
        idx = np.argmin(results)
    else:
        idx = np.argmax(results)

    return param_sets[idx], results[idx]


def evalaute(args):
    func = types.FunctionType(marshal.loads(args[0]), globals())
    parameters = args[1]
    score = func(**parameters)

    return score



