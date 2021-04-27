import pytest
import torch as T

from torch.utils.data import DataLoader

from data.synthetic_data import SyntheticDataset

csv_file = "synthetic_test_data.csv"

# These are hard-coded for the csv.
# Remember to change these if the csv changes!
expected_length = 10
seq_dim = 1
seq_length = 5

class Test_SyntheticDataset():

    @pytest.fixture()
    def dataset(self):
        return SyntheticDataset(csv_file)

    def test_length(self, dataset):
        assert len(dataset) == expected_length

    @pytest.mark.parametrize("index,expected_item", [
        (0, [0.7237234150137297, 0.13946122270849692, 0.8342476974452179, 0.5342173378336755, 0.33221789790952283]),
        (5, [0.24418778729303547,0.6655126266117972,0.6075295206183552,0.4758907741561006,0.5343760325320923]),
    ])
    def test_item(self, dataset, index, expected_item):
        actual = dataset[index]
        expected = T.tensor(expected_item, dtype=actual.dtype).unsqueeze(-1)

        assert T.allclose(actual, expected)

    @pytest.mark.parametrize("batch_size", [1, 5, 10])
    def test_batch_sizes(self, dataset, batch_size):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        batch = next(iter(loader))

        assert batch.shape == (batch_size, seq_length, seq_dim)

