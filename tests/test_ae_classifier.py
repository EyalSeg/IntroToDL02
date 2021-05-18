import pytest
import torch as T

from lstm_ae import LstmAutoEncoder
from ae_wrappers.ae_classification_wrapper import AutoEncoderClassifier

device = 'cuda:0' if T.cuda.is_available() else 'cpu'


class Test_AE_Classifier():
    @pytest.fixture()
    def ae(self, seq_dim, latent_size, num_layers, n_classes):
        ae = LstmAutoEncoder(seq_dim, latent_size, num_layers).to(device)
        return AutoEncoderClassifier(ae, n_classes)

    @pytest.mark.parametrize("seq_length", [50])
    @pytest.mark.parametrize("latent_size", [20])
    @pytest.mark.parametrize("batch_size", [1, 100])
    @pytest.mark.parametrize("seq_dim", [1, 5])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("n_classes", [1, 10])
    def test_output_shape(self, seq_length, latent_size, batch_size, seq_dim, num_layers, n_classes, ae):
        input = T.randn(batch_size, seq_length, seq_dim).to(device)

        with T.no_grad():
            output = ae.forward(input)
            sequence, predictions = output.output_sequence, output.label_predictions

        assert input.shape == sequence.shape, "Output is not at the same shape of the input!"
        assert predictions.shape == (batch_size, n_classes)

    @pytest.mark.parametrize("seq_length", [50])
    @pytest.mark.parametrize("latent_size", [20])
    @pytest.mark.parametrize("batch_size", [1, 100])
    @pytest.mark.parametrize("seq_dim", [1, 5])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("n_classes", [1, 2, 10])
    def test_predictions_sum_to_one(self, seq_length, latent_size, batch_size, seq_dim, num_layers, n_classes, ae):
        input = T.randn(batch_size, seq_length, seq_dim).to(device)

        with T.no_grad():
            output = ae.forward(input)
            predictions = output.label_predictions

        sums = T.sum(predictions, dim=1)
        assert T.allclose(sums, T.ones(batch_size).to(device)), "Predictions do no sum to one!"

