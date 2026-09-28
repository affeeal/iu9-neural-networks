import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hw5" / "sample"))
import hw5


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.data = TensorDataset(torch.randn(6, 1, 32, 32), torch.arange(6) % 10)
        self.loader = DataLoader(self.data, batch_size=3)
        self.device = torch.device("cpu")

    def test_shapes(self):
        self.assertEqual(hw5.create_model("lenet5")(self.data.tensors[0]).shape, (6, 10))
        # Validate large model shapes without allocating hundreds of MB of weights.
        for name in ("vgg16", "resnet34"):
            with torch.device("meta"):
                model = hw5.create_model(name).eval()
                self.assertEqual(model(torch.empty(2, 3, 32, 32)).shape, (2, 10))

    def test_training_updates_weights(self):
        for optimizer_name in hw5.OPTIMIZERS:
            with self.subTest(optimizer=optimizer_name):
                model = hw5.create_model("lenet5")
                before = model.fc2.weight.detach().clone()
                model.eval()
                optimizer = hw5.create_optimizer(optimizer_name, model.parameters(), 0.01)
                with contextlib.redirect_stdout(io.StringIO()):
                    losses = hw5.train(1, optimizer, model, nn.CrossEntropyLoss(),
                                       self.loader, self.device)
                self.assertTrue(model.training)
                self.assertTrue(torch.isfinite(torch.tensor(losses)).all())
                self.assertFalse(torch.equal(model.fc2.weight, before))

    def test_accuracy_preserves_buffers_and_modes(self):
        model = hw5.create_model("lenet5")
        model.layer1[1].eval()  # An intentionally frozen BatchNorm layer.
        modes = [module.training for module in model.modules()]
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        score = hw5.accuracy(model, self.loader, self.device)
        self.assertEqual(hw5.accuracy(model, self.loader, self.device), score)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        self.assertEqual([module.training for module in model.modules()], modes)
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(value, buffers[name]))

    def test_empty_loader_and_mode_restoration(self):
        model = hw5.create_model("lenet5")
        with self.assertRaises(ValueError):
            hw5.accuracy(model, [], self.device)
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            hw5.train(1, hw5.create_optimizer("sgd", model.parameters(), 0.01),
                      model, nn.CrossEntropyLoss(), [], self.device)

    def test_loss_is_weighted_by_sample_count(self):
        model = nn.Linear(2, 2)
        data = TensorDataset(torch.tensor([[1., 0.], [2., 0.], [9., 1.]]),
                             torch.tensor([0, 1, 1]))
        expected = nn.CrossEntropyLoss()(model(data.tensors[0]), data.tensors[1]).item()
        with contextlib.redirect_stdout(io.StringIO()):
            losses = hw5.train(1, torch.optim.SGD(model.parameters(), lr=0),
                               model, nn.CrossEntropyLoss(), DataLoader(data, batch_size=2),
                               self.device)
        self.assertAlmostEqual(losses[0], expected, places=6)

    def test_experiments_start_from_same_weights(self):
        initial = []
        factory = hw5.create_model

        def capture(name):
            model = factory(name)
            initial.append(model.fc2.weight.detach().clone())
            return model

        with patch.object(hw5, "create_model", side_effect=capture):
            with contextlib.redirect_stdout(io.StringIO()):
                for name in ("sgd", "adam"):
                    hw5.run_experiment("lenet5", name, self.data, self.data,
                                       epochs=1, batch_size=3, learning_rate=0.01,
                                       seed=42, device=self.device)
        self.assertTrue(torch.equal(initial[0], initial[1]))

    def test_cli_validation_precedes_dataset_access(self):
        with patch.object(hw5, "load_datasets") as datasets:
            for arguments in (["--epochs", "0"], ["--batch-size", "-1"],
                              ["--learning-rate", "nan"], ["--seed", "-1"]):
                with self.subTest(arguments=arguments):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as exit_context:
                            hw5.main(arguments)
                    self.assertNotEqual(exit_context.exception.code, 0)
            datasets.assert_not_called()

    def test_shared_transform_and_explicit_download(self):
        for model, dataset in (("lenet5", "MNIST"), ("resnet34", "CIFAR10")):
            with patch.object(hw5.datasets, dataset) as constructor:
                hw5.load_datasets(model, Path("/unused"))
                train, test = constructor.call_args_list
                self.assertIs(train.kwargs["transform"], test.kwargs["transform"])
                self.assertFalse(train.kwargs["download"])
                self.assertFalse(test.kwargs["download"])


if __name__ == "__main__":
    unittest.main()
