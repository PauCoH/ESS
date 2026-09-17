import random
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ess import ESSConfig
from ess.artifacts import CheckpointSelector
from ess.models import ControllerInput, create_model
from ess.reproducibility import seed_everything
from ess.scheduling import StageLRScheduler
from ess.training import apply_accumulated_gradients


class ESSTests(unittest.TestCase):
    def test_workspace_is_set_before_torch_import(self):
        script = '''
import builtins
import os
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name == 'torch':
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') in {':4096:8', ':16:8'}
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
import ess
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == EXPECTED
'''
        for existing in (None, ':16:8'):
            environment = os.environ.copy()
            environment.pop('CUBLAS_WORKSPACE_CONFIG', None)
            if existing is not None:
                environment['CUBLAS_WORKSPACE_CONFIG'] = existing
            result = subprocess.run(
                [sys.executable, '-c', script.replace('EXPECTED', repr(existing or ':4096:8'))],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_workspace_has_actionable_error(self):
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ''}):
            with self.assertRaisesRegex(RuntimeError, 'restart the kernel'):
                seed_everything(0, deterministic=True)

    def test_partial_batch_uses_actual_count(self):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for x in (1., 2.):
            model(torch.tensor([[x]])).square().mean().backward()
        apply_accumulated_gradients(model, optimizer, 2, None)
        self.assertAlmostEqual(model.weight.item(), 0.5)

    def test_scheduler_metric_bounds_and_no_restoration(self):
        config = ESSConfig()
        config.lr.plateau_patience = 0
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=config.lr.early)
        scheduler = StageLRScheduler(optimizer, config)
        scheduler.begin_phase('early')
        scheduler.step(1, 10)
        reduced, metric = scheduler.step(2, 1)
        self.assertEqual(metric, 2)
        self.assertEqual(reduced, config.lr.early * 0.5)
        self.assertEqual(scheduler.begin_phase('early'), reduced)
        scheduler.begin_phase('middle')
        self.assertLessEqual(optimizer.param_groups[0]['lr'], config.lr.middle)
        scheduler.begin_phase('late')
        scheduler.step(10, 1)
        reduced, metric = scheduler.step(1, 2)
        self.assertEqual(metric, 2)
        self.assertEqual(reduced, config.lr.late * 0.5)
        for _ in range(30):
            scheduler.step(0, 100)
        self.assertEqual(optimizer.param_groups[0]['lr'], config.lr.minimum)
        scheduler.begin_phase('fine_tuning')
        self.assertEqual(scheduler.metric_name, 'validation_loss')
        self.assertEqual(optimizer.param_groups[0]['lr'], config.lr.minimum)

    def test_early_best_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            selector = CheckpointSelector(tmp, 'ess', .95)
            model = torch.nn.Linear(1, 1)
            selector.consider(model, .01, 1, .1, 'early')
            self.assertIsNone(selector.best)
            self.assertFalse(selector.path.exists())
            selector.consider(model, .5, 2, .96, 'late')
            self.assertEqual(selector.best['epoch'], 2)
            self.assertEqual(selector.monitoring['epoch'], 1)
            self.assertTrue(selector.path.exists())

    def test_config_rejects_unreachable_selection(self):
        ESSConfig().validate()
        with self.assertRaises(ValueError):
            ESSConfig(epochs=2).validate()
        with self.assertRaises(ValueError):
            ESSConfig(architecture='unknown').validate()

    def test_seed_covers_all_random_sources(self):
        def draw():
            seed_everything(19)
            return random.random(), np.random.normal(), torch.rand(3), create_model('LeakyMLP4', 'cpu').state_dict()
        first, second = draw(), draw()
        self.assertEqual(first[:2], second[:2])
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertTrue(all(torch.equal(first[3][key], second[3][key]) for key in first[3]))

    def test_all_architectures_and_window_protocol(self):
        torch.set_num_threads(1)
        for architecture in ('LeakyMLP4', 'MLP8', 'LSTM'):
            with self.subTest(architecture=architecture):
                model = create_model(architecture, 'cpu')
                model.eval()
                first = ControllerInput(architecture, 4, 'cpu')
                second = ControllerInput(architecture, 4, 'cpu')
                for i in range(6):
                    features = np.full(4, i / 10)
                    prediction = first.predict(model, features)
                    self.assertTrue(torch.equal(prediction, second.predict(model, features)))
                    self.assertEqual(prediction.shape, torch.Size([]))
                prediction.backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters()))
                if architecture == 'LSTM':
                    self.assertTrue(np.allclose(list(first.history)[0], .2))


if __name__ == '__main__':
    unittest.main()
