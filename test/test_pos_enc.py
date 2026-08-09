"""Tests for FeatureExtractor.convert_position (positional encoding).

All times are integer milliseconds (CLAMS convention). With the default
``pos_length=6000000`` and ``pos_unit=60000`` the lookup table has one
row per minute, 100 rows total. ``pos_abs_th_front`` / ``pos_abs_th_end``
are unit counts: the first/last N minutes are encoded by absolute
distance from the start/end, everything in between by relative position.
"""
import pytest

pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('transformers')

from conftest import install_mock_backbone

data_loader = install_mock_backbone()

MIN = 60_000  # one pos_unit (a minute) in milliseconds
DIM = 100     # rows in the lookup table with the default length/unit


def prep_extractor(th_front, th_end):
    return data_loader.FeatureExtractor(
        img_enc_name='mock',
        pos_length=6_000_000,
        pos_unit=MIN,
        pos_abs_th_front=th_front,
        pos_abs_th_end=th_end,
        pos_vec_coeff=0.75,
    )


class TestFrontRegion:

    def test_absolute_rows_from_start(self):
        ex = prep_extractor(3, 10)
        tot = 30 * MIN
        assert ex.convert_position(0, tot) == 0
        assert ex.convert_position(MIN - 1, tot) == 0
        assert ex.convert_position(MIN, tot) == 1
        assert ex.convert_position(3 * MIN - 1, tot) == 2

    def test_duration_invariant(self):
        # the same wall-clock offset maps to the same row regardless of
        # how long the video is
        ex = prep_extractor(3, 10)
        for tot in (20 * MIN, 30 * MIN, 90 * MIN):
            assert ex.convert_position(MIN, tot) == 1

    def test_zero_threshold_disables_region(self):
        ex = prep_extractor(0, 10)
        tot = 100 * MIN
        # with no front region, ms 0 falls through to relative encoding
        assert ex.convert_position(0, tot) == 0
        assert ex.convert_position(MIN, tot) == 1  # relative: 1/100


class TestRearRegion:

    def test_absolute_rows_from_end(self):
        ex = prep_extractor(3, 10)
        tot = 30 * MIN
        assert ex.convert_position(tot, tot) == DIM - 1
        assert ex.convert_position(tot - MIN, tot) == DIM - 2
        # 9-and-a-bit minutes from the end, still inside the 10-unit
        # rear region
        assert ex.convert_position(tot - 9 * MIN - 1, tot) == DIM - 10

    def test_duration_invariant(self):
        ex = prep_extractor(3, 10)
        for tot in (20 * MIN, 90 * MIN):
            assert ex.convert_position(tot - MIN, tot) == DIM - 2

    def test_zero_threshold_disables_region(self):
        ex = prep_extractor(3, 0)
        tot = 100 * MIN
        # with no rear region, the last sample encodes relatively (and
        # the final clamp keeps cur == tot inside the table)
        assert ex.convert_position(99 * MIN, tot) == 99
        assert ex.convert_position(tot, tot) == DIM - 1


class TestMiddleRegion:

    def test_relative_position(self):
        ex = prep_extractor(3, 10)
        tot = 30 * MIN
        assert ex.convert_position(15 * MIN, tot) == 50

    def test_relative_only_when_thresholds_zero(self):
        ex = prep_extractor(0, 0)
        tot = 20 * MIN
        assert ex.convert_position(0, tot) == 0
        assert ex.convert_position(5 * MIN, tot) == 25
        assert ex.convert_position(10 * MIN, tot) == 50
        assert ex.convert_position(tot, tot) == DIM - 1


class TestBounds:

    def test_always_in_bounds(self):
        # regression for the v8.8 CUDA device-side assert: the rear
        # branch used to return raw ms, which overran the 100-row table
        # on the downstream gather. 1_957_003 is the video duration that
        # first triggered the crash (duration % 1000 < 10).
        ex = prep_extractor(3, 10)
        for tot in (1, MIN, 30 * MIN, 1_957_003, 200 * MIN):
            for cur in (0, 1, 500, tot // 2, tot - 1, tot):
                p = ex.convert_position(cur, tot)
                assert 0 <= p < DIM, (cur, tot, p)

    def test_v88_crash_sample(self):
        # the exact sampling that crashed v8.8: last 1000ms-grid sample
        # of the 1_957_003ms video lands 3ms before the end
        ex = prep_extractor(3, 10)
        assert ex.convert_position(1_957_000, 1_957_003) == DIM - 1


if __name__ == '__main__':
    pytest.main([__file__])
