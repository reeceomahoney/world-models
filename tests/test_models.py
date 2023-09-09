import unittest

import numpy as np
import torch
from PIL import Image

import dreamer


class TestModels(unittest.TestCase):
    def test_downsample(self):
        arr = np.random.rand(500, 500, 3)
        frame_image = Image.fromarray(arr, mode="RGB")
        frame_image = frame_image.resize((64, 64))
        frame = np.array(frame_image)
        self.assertEqual(frame.shape, (64, 64, 3))

    def test_conv_encoder(self):
        # (time, batch, h, w, ch)
        img = torch.randn(16, 32, 64, 64, 3).to("cuda")
        encoder = dreamer.models.ConvEncoder((64, 64, 3))
        out = encoder(img)
        self.assertEqual(out.shape, (16, 32, 256 * 4 * 4))

    def test_conv_decoder(self):
        # (time, batch, h, w, ch)
        feat_size = 1024 + 255
        state = torch.randn(16, 32, feat_size).to("cuda")
        decoder = dreamer.models.ConvDecoder(feat_size)
        out = decoder(state).sample()
        self.assertEqual(out.shape, (16, 32, 64, 64, 3))

    def test_mlp(self):
        in_dim = 64
        out_dim = 1024
        hidden_layers = [512, 512]

        for d in ["normal", "binary", "categorical"]:
            x = torch.randn(16, in_dim).to("cuda")
            dist = dreamer.models.MLP(in_dim, out_dim, hidden_layers, dist=d)(x)
            x = dist.sample()
            dist.log_prob(x)
            dist.entropy()
            dist.mode
            self.assertEqual(x.shape, (16, out_dim))


if __name__ == "__main__":
    unittest.main()
