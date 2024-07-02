import unittest
from modeling import data_loader
import torch

extractor = data_loader.FeatureExtractor(
        img_enc_name="convnext_lg",
        pos_enc_name="sinusoidal-add",
        pos_enc_dim=256,
        input_length=6000000,
        pos_unit=60000,
        pos_abs_th_front=5,
        pos_abs_th_end=10,
        pos_vec_coeff=0.75
    )

extractor.pos_vec_lookup = extractor.get_sinusoidal_embeddings(100, extractor.img_encoder.dim)


class TestPosAbsTh(unittest.TestCase):
    def test_cur_time_before(self):
        img_vec = torch.ones(extractor.img_encoder.dim)
        pos_vec = extractor.pos_vec_lookup[3] * extractor.pos_vec_coeff
        vector = extractor.encode_position(3, 20, img_vec)
        self.assertTrue(torch.equal(vector, torch.add(img_vec, pos_vec)))

    def test_cur_time_after(self):
        img_vec = torch.ones(extractor.img_encoder.dim)
        pos_vec = extractor.pos_vec_lookup[12] * extractor.pos_vec_coeff
        vector = extractor.encode_position(12, 20, img_vec)
        self.assertTrue(torch.equal(vector, torch.add(img_vec, pos_vec)))

    def test_cur_time_between(self):
        img_vec = torch.ones(extractor.img_encoder.dim)
        pos_lookup_col = 8 * extractor.pos_vec_lookup.shape[0] // 20
        pos_vec = extractor.pos_vec_lookup[pos_lookup_col] * extractor.pos_vec_coeff
        vector = extractor.encode_position(8, 20, img_vec)
        self.assertTrue(torch.equal(vector, torch.add(img_vec, pos_vec)))

    def test_cur_time_th_front(self):
        img_vec = torch.ones(extractor.img_encoder.dim)
        pos_lookup_col = 5 * extractor.pos_vec_lookup.shape[0] // 20
        pos_vec = extractor.pos_vec_lookup[pos_lookup_col] * extractor.pos_vec_coeff
        vector = extractor.encode_position(5, 20, img_vec)
        self.assertTrue(torch.equal(vector, torch.add(img_vec, pos_vec)))

    def test_cur_time_th_end(self):
        img_vec = torch.ones(extractor.img_encoder.dim)
        pos_lookup_col = 10 * extractor.pos_vec_lookup.shape[0] // 20
        pos_vec = extractor.pos_vec_lookup[pos_lookup_col] * extractor.pos_vec_coeff
        vector = extractor.encode_position(10, 20, img_vec)
        self.assertTrue(torch.equal(vector, torch.add(img_vec, pos_vec)))


if __name__ == '__main__':
    unittest.main()
