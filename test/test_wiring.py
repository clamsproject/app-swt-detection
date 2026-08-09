"""Wiring tests: configured values must reach the constructed objects.

Regression guards for the v8.x gridsearch bug where the positional
encoding parameters in a training config never reached the
FeatureExtractor: training silently used the class defaults while result
filenames claimed otherwise, so every sweep over those parameters was
inert. These tests read the values back off the constructed objects, so
a broken config-to-consumer path fails loudly instead of shipping.
"""
import logging

import pytest

pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('transformers')

import torch
from torch.utils.data import DataLoader

from conftest import install_mock_backbone

data_loader = install_mock_backbone()

from modeling import train

POS_CONFIG = {
    'img_enc_name': 'mock',
    'pos_length': 6_000_000,
    'pos_unit': 60_000,
    'pos_abs_th_front': 2,
    'pos_abs_th_end': 3,
    'pos_vec_coeff': 0.25,
}


def assert_extractor_matches(fe, cfg):
    assert fe.pos_unit == cfg['pos_unit']
    assert fe.pos_abs_th_front == cfg['pos_abs_th_front']
    assert fe.pos_abs_th_end == cfg['pos_abs_th_end']
    assert fe.pos_vec_coeff == cfg['pos_vec_coeff']


class TestConfigReachesFeatureExtractor:

    def test_training_dataset_forwards_config(self, tmp_path):
        # train.py path: SWTH5Dataset(model_config=configs)
        cfg = dict(POS_CONFIG, num_epochs=1, seed=42)
        ds = train.SWTH5Dataset(str(tmp_path), [], 'mock', 'distorted',
                                model_config=cfg)
        assert_extractor_matches(ds.feat_extr, cfg)

    def test_inference_style_construction_forwards_config(self):
        # classify.py path: FeatureExtractor(**model_config) with the
        # full training yml; extra keys are swallowed, pos params applied
        cfg = dict(POS_CONFIG, num_epochs=8, num_layers=5, dropouts=0.3,
                   learning_rate=1e-4, seed=7, block_guids_train=[])
        fe = data_loader.FeatureExtractor(**cfg)
        assert_extractor_matches(fe, cfg)

    def test_pos_vec_coeff_changes_features(self):
        # two extractors differing only in pos_vec_coeff must produce
        # different feature vectors; when the v8.x wiring bug was live,
        # every coeff produced identical models
        positions = [[60_000, 1_800_000]]
        img = torch.zeros((1, 256))
        outs = []
        for coeff in (0, 0.5):
            fe = data_loader.FeatureExtractor(
                **dict(POS_CONFIG, pos_vec_coeff=coeff))
            fe.pos_vec_lookup = fe.pos_vec_lookup.cpu()
            outs.append(fe.encode_position(positions, img.clone()))
        assert not torch.equal(outs[0], outs[1])
        # coeff 0 means positional encoding fully off
        assert torch.equal(outs[0], torch.zeros((1, 256)))

    def test_unknown_kwarg_is_logged(self, caplog):
        # a typo'd param name (e.g. pos_enc_coeff for pos_vec_coeff)
        # must be visible in the logs, not a silent default fallback
        with caplog.at_level(logging.WARNING, 'modeling.data_loader'):
            fe = data_loader.FeatureExtractor(
                **dict(POS_CONFIG), pos_enc_coeff=0.9)
        assert 'pos_enc_coeff' in caplog.text
        assert fe.pos_vec_coeff == POS_CONFIG['pos_vec_coeff']


class TestSeedDeterminism:

    def test_set_seed_reproducible(self):
        train.set_seed(4)
        a = torch.rand(8)
        train.set_seed(4)
        b = torch.rand(8)
        assert torch.equal(a, b)
        train.set_seed(5)
        c = torch.rand(8)
        assert not torch.equal(a, c)

    def test_seeded_loader_shuffle_reproducible(self):
        # mirrors the train() DataLoader setup: an explicitly seeded
        # generator pins the shuffle order
        data = list(range(64))

        def order(seed):
            g = torch.Generator()
            g.manual_seed(seed)
            dl = DataLoader(data, batch_size=1, shuffle=True,
                            generator=g)
            return [int(b[0]) for b in dl]

        assert order(4) == order(4)
        assert order(4) != order(5)

    def test_train_guid_order_is_deterministic(self):
        # guards the cross-launch nondeterminism found via the
        # stg4/stg5 controls: a bare list(set(...)) follows the
        # per-interpreter PYTHONHASHSEED, so same-seed launches got
        # different batch compositions; the resolved training guid
        # list must be identical however the inputs are ordered
        a = ['g3', 'g1', 'g2', 'g4', 'g5']
        b = list(reversed(a))
        blocked, valid = ['g4'], ['g5']
        assert (train.resolve_train_guids(a, blocked, valid)
                == train.resolve_train_guids(b, blocked, valid)
                == ['g1', 'g2', 'g3'])


if __name__ == '__main__':
    pytest.main([__file__])
