import collections
import unittest

from modeling import data_loader

# set up some mock, to avoid loading the full torch-vision model 
DummyImgEncoer = collections.namedtuple('DummyImgEncoer', ['dim'])
data_loader.backbones.model_map = collections.defaultdict(
    lambda: lambda: DummyImgEncoer(256))


class TestPosAbsTh(unittest.TestCase):
    
    @staticmethod
    def prep_extractor(th_front, th_end, cols=100):
        extractor = data_loader.FeatureExtractor(
            img_enc_name="mock_model_name",
            pos_length=6000000,
            pos_unit=60000,
            pos_abs_th_front=th_front,
            pos_abs_th_end=th_end,
            pos_vec_coeff=0.75
        )
        extractor.pos_vec_lookup = extractor.get_sinusoidal_embeddings(cols, extractor.img_encoder.dim)
        return extractor

    def test_convert_position(self):
        extractor = self.prep_extractor(10, 10, 100)
        
        # test if cur is in "front" enough 
        cur_time = 5
        tot_time = 30
        self.assertEqual(extractor.convert_position(cur_time, tot_time),cur_time) 
        
        # test the end side as well
        cur_time = 25
        tot_time = 30
        self.assertEqual(extractor.convert_position(cur_time, tot_time),cur_time)
        
        # then test if cur is in the middle part
        cur_time = 15
        tot_time = 30
        self.assertEqual(extractor.convert_position(cur_time, tot_time), 50)
        
        # what if the input is much longer than the position matrix?
        cur_time = 100
        tot_time = 200
        self.assertEqual(extractor.convert_position(cur_time, tot_time), 50)
        
        cur_time = 5
        tot_time = 200
        self.assertEqual(extractor.convert_position(cur_time, tot_time), cur_time)
        # near-end branch returns raw `cur` (195), which is out of the 100-row
        # table; the v8.9 clamp caps it at the last valid row (dim - 1 = 99)
        # rather than returning an OOB index that would assert on the GPU gather.
        cur_time = 195
        tot_time = 200
        self.assertEqual(extractor.convert_position(cur_time, tot_time), 99)

    def test_convert_position_always_in_bounds(self):
        # Regression for the v8.8 CUDA OOB crash: the near-end branch returned
        # raw ms, which the downstream `pos_vec_lookup[...]` gather asserted on.
        # The clamp must keep the index in [0, dim) for any (cur, tot).
        # 1_957_003 is the Jack_Taylor duration that first triggered it.
        extractor = self.prep_extractor(10, 10, 100)
        dim = extractor.pos_vec_lookup.shape[0]
        for tot in (1, 60_000, 1_800_000, 1_957_003):
            for cur in (0, 1, 500, tot - 1, tot):
                p = extractor.convert_position(cur, tot)
                self.assertTrue(0 <= p < dim, (cur, tot, p, dim))

    @unittest.skip("Some extreme edge cases")
    def test_convert_position_edgecases(self):
        # full matrix is covered just by thresholds
        extractor = self.prep_extractor(5, 15, 20)
        cur_time = 50
        tot_time = 100
        self.assertEqual(extractor.convert_position(cur_time, tot_time), 10)
        
        # full matrix is shorter than thresholds
        extractor = self.prep_extractor(40, 50, 80)
        cur_time = 60
        tot_time = 80
        self.assertEqual(extractor.convert_position(cur_time, tot_time), 40)
        
        
if __name__ == '__main__':
    unittest.main()
