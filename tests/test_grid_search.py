import pytest

from grid_search import tune


class Test_GridSearch():

    @pytest.mark.parametrize("func,params_dict,dir,expected_params,expected_result", [
        (lambda b, e: b ** e, {'b': [1, 2, 3], 'e': [0, 1, 2]}, "maximize", {'b': 3, 'e': 2}, 3 ** 2),
        (lambda b, e: b ** e, {'b': [2, 3, 4], 'e': [1, 2, 3]}, "minimize", {'b': 2, 'e': 1}, 2)
    ])
    @pytest.mark.parametrize("workers", [1, 2, 4])
    def test(self, func, params_dict, dir, expected_params, expected_result, workers):
        best_params, best_result = tune(func, params_dict, direction=dir, workers=workers)

        assert best_params == expected_params, "wrong params found!"
        assert best_result == expected_result, "wrong result!"
