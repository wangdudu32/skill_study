"""测试模型计算和训练。"""

import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from data import (
    BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE,
    ReverseDataset, collate_batch, decode_numbers, encode_numbers,
)
from train import evaluate, train_epoch
from transformer import (
    LayerNorm, MultiHeadAttention, SinusoidalPositionalEncoding,
    Transformer, TransformerConfig, greedy_decode,
    make_causal_mask, make_padding_mask, scaled_dot_product_attention,
)


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.config = TransformerConfig(
            src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
            d_model=32, n_heads=4, num_encoder_layers=1, num_decoder_layers=1,
            d_ff=64, dropout=0.0, max_len=16,
        )

    def test_attention_matches_hand_calculation(self):
        q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
        k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
        v = torch.tensor([[[[10.0, 0.0], [0.0, 20.0]]]], dtype=torch.float64)
        output, weights = scaled_dot_product_attention(q, k, v)
        p = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
        torch.testing.assert_close(weights, torch.tensor([[[[p, 1 - p]]]], dtype=q.dtype))
        torch.testing.assert_close(output, torch.tensor([[[[10 * p, 20 * (1 - p)]]]], dtype=q.dtype))

    def test_masked_key_and_value_cannot_affect_attention(self):
        q = torch.randn(2, 3, 4, 8)
        k, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 8)
        mask = torch.tensor([True, True, True, False, False])[None, None, None, :]
        before, weights = scaled_dot_product_attention(q, k, v, mask)
        k[:, :, 3:] = 10000
        v[:, :, 3:] = -10000
        after, _ = scaled_dot_product_attention(q, k, v, mask)
        torch.testing.assert_close(before, after)
        self.assertTrue((weights[..., 3:] == 0).all())
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 3, 4))

    def test_fully_masked_rows_have_zero_weights_and_finite_gradients(self):
        q = torch.randn(2, 2, 3, 4, requires_grad=True)
        k = torch.randn(2, 2, 5, 4, requires_grad=True)
        v = torch.randn(2, 2, 5, 4, requires_grad=True)
        mask = torch.ones(2, 1, 3, 5, dtype=torch.bool)
        mask[0, :, 1, :] = False
        output, weights = scaled_dot_product_attention(q, k, v, mask)
        self.assertTrue((output[0, :, 1] == 0).all())
        self.assertTrue((weights[0, :, 1] == 0).all())
        output.square().sum().backward()
        for tensor in (q, k, v):
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_attention_rejects_ambiguous_numeric_mask(self):
        x = torch.randn(1, 1, 2, 4)
        with self.assertRaisesRegex(TypeError, "bool"):
            scaled_dot_product_attention(x, x, x, torch.ones(2, 2))

    def test_multihead_cross_attention_allows_different_lengths(self):
        attention = MultiHeadAttention(32, 4, dropout=0)
        query = torch.randn(2, 3, 32, requires_grad=True)
        memory = torch.randn(2, 7, 32, requires_grad=True)
        output = attention(query, memory, memory)
        self.assertEqual(output.shape, (2, 3, 32))
        output.square().sum().backward()
        for parameter in attention.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(memory.grad.abs().sum().item(), 0)

    def test_layer_norm_matches_pytorch_values_and_gradients(self):
        layer = LayerNorm(7).double()
        with torch.no_grad():
            layer.weight.normal_()
            layer.bias.normal_()
        x = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
        actual = layer(x)
        expected = F.layer_norm(x, (7,), layer.weight, layer.bias, layer.eps)
        torch.testing.assert_close(actual, expected)
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (x, layer.weight, layer.bias), upstream)
        expected_grads = torch.autograd.grad(expected, (x, layer.weight, layer.bias), upstream)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_sinusoidal_positions_and_length_limit(self):
        position = SinusoidalPositionalEncoding(5, max_len=4, dropout=0)
        output = position(torch.zeros(2, 4, 5))
        torch.testing.assert_close(output[0, 0], torch.tensor([0., 1., 0., 1., 0.]))
        self.assertAlmostEqual(output[0, 1, 0].item(), math.sin(1), places=6)
        self.assertAlmostEqual(output[0, 1, 1].item(), math.cos(1), places=6)
        self.assertFalse(position.pe.requires_grad)
        self.assertIn("pe", position.state_dict())
        with self.assertRaisesRegex(ValueError, "max_len"):
            position(torch.zeros(1, 5, 5))

    def test_masks_have_correct_broadcasting_and_direction(self):
        tokens = torch.tensor([[BOS_ID, 4, PAD_ID], [BOS_ID, 5, 6]])
        padding = make_padding_mask(tokens, PAD_ID)
        self.assertEqual(padding.shape, (2, 1, 1, 3))
        causal = make_causal_mask(3)
        expected = torch.tensor([[True, False, False], [True, True, False], [True, True, True]])
        torch.testing.assert_close(causal[0, 0], expected)
        combined = padding & causal
        self.assertFalse(combined[0, 0, :, 2].any())
        self.assertTrue(combined[1, 0, 2, 2])

    def test_future_target_tokens_do_not_change_past_logits(self):
        model = Transformer(self.config).eval()
        src = torch.tensor([[3, 4, 5, EOS_ID]])
        tgt = torch.tensor([[BOS_ID, 5, 4, 3]])
        changed = tgt.clone()
        changed[:, 2:] = torch.tensor([10, 11])
        with torch.no_grad():
            before, after = model(src, tgt), model(src, changed)
        torch.testing.assert_close(before[:, :2], after[:, :2], atol=1e-6, rtol=1e-5)
        self.assertFalse(torch.allclose(before[:, 2:], after[:, 2:]))

    def test_parallel_decoder_matches_each_prefix(self):
        model = Transformer(self.config).eval()
        src = torch.tensor([[3, 7, EOS_ID]])
        tgt = torch.tensor([[BOS_ID, 7, 3]])
        with torch.no_grad():
            full = model(src, tgt)
            for length in range(1, tgt.size(1) + 1):
                prefix = model(src, tgt[:, :length])
                torch.testing.assert_close(full[:, length - 1], prefix[:, -1], atol=1e-6, rtol=1e-5)

    def test_padding_does_not_change_real_token_logits(self):
        model = Transformer(self.config).eval()
        src = torch.tensor([[4, 8, EOS_ID]])
        tgt = torch.tensor([[BOS_ID, 8, 4]])
        padded_src = F.pad(src, (0, 3), value=PAD_ID)
        padded_tgt = F.pad(tgt, (0, 2), value=PAD_ID)
        with torch.no_grad():
            expected = model(src, tgt)
            # 改动 PAD 向量后，正常位置的结果应该不变
            model.src_embedding.weight[PAD_ID].fill_(1000)
            model.tgt_embedding.weight[PAD_ID].fill_(-1000)
            actual = model(padded_src, padded_tgt)
        torch.testing.assert_close(expected, actual[:, :tgt.size(1)], atol=1e-6, rtol=1e-5)

    def test_whole_model_backward_reaches_every_parameter(self):
        model = Transformer(self.config)
        src, tgt, labels = collate_batch([ReverseDataset(1, seed=1)[0], ReverseDataset(1, seed=2)[0]])
        logits = model(src, tgt)
        self.assertEqual(logits.shape, (*tgt.shape, VOCAB_SIZE))
        F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1), ignore_index=PAD_ID).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertTrue((model.src_embedding.weight.grad[PAD_ID] == 0).all())
        self.assertTrue((model.tgt_embedding.weight.grad[PAD_ID] == 0).all())

    def test_data_shift_padding_and_seed(self):
        first = ([4, 5, EOS_ID], [BOS_ID, 5, 4, EOS_ID])
        second = ([6, EOS_ID], [BOS_ID, 6, EOS_ID])
        src, decoder_input, labels = collate_batch([first, second])
        self.assertEqual(src.tolist(), [[4, 5, EOS_ID], [6, EOS_ID, PAD_ID]])
        self.assertEqual(decoder_input.tolist(), [[BOS_ID, 5, 4], [BOS_ID, 6, EOS_ID]])
        self.assertEqual(labels.tolist(), [[5, 4, EOS_ID], [6, EOS_ID, PAD_ID]])
        self.assertEqual(ReverseDataset(4, seed=11).samples, ReverseDataset(4, seed=11).samples)
        self.assertEqual(decode_numbers(encode_numbers([0, 9, 2]) + [EOS_ID, PAD_ID]), [0, 9, 2])
        with self.assertRaises(ValueError):
            encode_numbers([10])

    def test_greedy_decode_handles_independent_eos_and_restores_mode(self):
        model = Transformer(self.config).train()
        src = torch.tensor([[4, EOS_ID], [6, EOS_ID]])

        def scripted_logits(tgt, memory, src_mask):
            self.assertFalse(model.training)
            logits = torch.zeros(2, tgt.size(1), VOCAB_SIZE)
            logits[0, -1, EOS_ID] = 10
            logits[1, -1, 6 if tgt.size(1) == 1 else EOS_ID] = 10
            return logits

        with patch.object(model, "decode", side_effect=scripted_logits) as decode:
            output = greedy_decode(model, src, max_new_tokens=5)
        self.assertEqual(output.tolist(), [[BOS_ID, EOS_ID, PAD_ID], [BOS_ID, 6, EOS_ID]])
        self.assertEqual(decode.call_count, 2)
        self.assertTrue(model.training)

    def test_checkpoint_round_trip(self):
        model = Transformer(self.config).eval()
        src, tgt, _ = collate_batch([ReverseDataset(1)[0]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model_config": asdict(self.config), "model_state_dict": model.state_dict()}, path)
            checkpoint = torch.load(path, weights_only=True, map_location="cpu")
            restored = Transformer(TransformerConfig(**checkpoint["model_config"])).eval()
            restored.load_state_dict(checkpoint["model_state_dict"])
            with torch.no_grad():
                torch.testing.assert_close(model(src, tgt), restored(src, tgt), atol=0, rtol=0)

    def test_can_overfit_a_tiny_batch_and_generate_exact_sequences(self):
        samples = []
        for digits in ([1, 2, 3], [8, 0], [4, 4, 9], [9, 7, 5, 2]):
            tokens = encode_numbers(digits)
            samples.append((tokens + [EOS_ID], [BOS_ID] + tokens[::-1] + [EOS_ID]))
        loader = [collate_batch(samples)]
        model = Transformer(self.config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial = evaluate(model, loader, torch.device("cpu"))
        for _ in range(150):
            train_epoch(model, loader, optimizer, torch.device("cpu"))
        final = evaluate(model, loader, torch.device("cpu"))
        self.assertLess(final["loss"], initial["loss"] * 0.05)
        self.assertEqual(final["sequence_accuracy"], 1.0)

    def test_invalid_architecture_fails_early(self):
        with self.assertRaisesRegex(ValueError, "整除"):
            TransformerConfig(VOCAB_SIZE, VOCAB_SIZE, d_model=30, n_heads=4)
        with self.assertRaises(ValueError):
            TransformerConfig(VOCAB_SIZE, VOCAB_SIZE, dropout=1.0)
        with self.assertRaises(ValueError):
            TransformerConfig(VOCAB_SIZE, VOCAB_SIZE, num_encoder_layers=0)


if __name__ == "__main__":
    unittest.main()
