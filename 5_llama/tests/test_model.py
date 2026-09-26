"""测试模型计算、缓存和训练。"""

import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from generate import generate_tokens, load_checkpoint, sample_next_token
from model import GroupedQueryAttention, MiniLlama, ModelConfig, RMSNorm, apply_rope, rope_angles
from tokenizer import CharTokenizer
from train import evaluate, sample_batch


class LlamaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)
        self.config = ModelConfig(vocab_size=24, dim=32, n_layers=2, n_heads=4,
                                  n_kv_heads=2, hidden_dim=64, max_seq_len=32)
        self.model = MiniLlama(self.config)
        self.tokens = torch.randint(0, self.config.vocab_size, (2, 11))

    def test_invalid_configs(self):
        for values in ({"dim": 30}, {"dim": 12}, {"n_kv_heads": 3}, {"n_layers": 0},
                       {"max_seq_len": 0}, {"hidden_dim": -1}, {"rope_theta": float("nan")},
                       {"norm_eps": 0}, {"n_heads": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ModelConfig(**(asdict(self.config) | values))

    def test_rmsnorm_matches_pytorch_reference(self):
        norm = RMSNorm(32)
        with torch.no_grad():
            norm.weight.uniform_(0.5, 1.5)
        x = torch.randn(2, 7, 32) + 3  # 让均值偏离 0，检查有没有误写成 LayerNorm
        expected = F.rms_norm(x, (32,), norm.weight, norm.eps)
        torch.testing.assert_close(norm(x), expected)

    def test_rope_known_rotation_and_norm(self):
        x = torch.tensor([[[[1., 0., 1., 0.], [1., 0., 1., 0.]]]])
        cos, sin = rope_angles(torch.tensor([0, 1]), 4, 100.0)
        result = apply_rope(x, cos, sin)
        torch.testing.assert_close(result[:, :, 0], x[:, :, 0])
        expected = torch.tensor([math.cos(1), math.sin(1), math.cos(0.1), math.sin(0.1)])
        torch.testing.assert_close(result[0, 0, 1], expected)
        torch.testing.assert_close(result.norm(dim=-1), x.norm(dim=-1))

    def test_rope_relative_position(self):
        q, k = torch.randn(2, 4, 1, 8), torch.randn(2, 4, 1, 8)

        def inner_product(q_pos, k_pos):
            rq = apply_rope(q, *rope_angles(torch.tensor([q_pos]), 8, 10000.0))
            rk = apply_rope(k, *rope_angles(torch.tensor([k_pos]), 8, 10000.0))
            return (rq * rk).sum(-1)

        torch.testing.assert_close(inner_product(2, 5), inner_product(9, 12), atol=2e-6, rtol=2e-6)

    def test_gqa_matches_reference_attention_per_head(self):
        attention = GroupedQueryAttention(self.config)
        x = torch.randn(2, 5, 32)
        cos, sin = rope_angles(torch.arange(5), 8, self.config.rope_theta)
        blocked = torch.ones(5, 5, dtype=torch.bool).triu(1)
        actual, _ = attention(x, cos, sin, blocked)
        q = apply_rope(attention.q_proj(x).view(2, 5, 4, 8).transpose(1, 2), cos, sin)
        k = apply_rope(attention.k_proj(x).view(2, 5, 2, 8).transpose(1, 2), cos, sin)
        v = attention.v_proj(x).view(2, 5, 2, 8).transpose(1, 2)
        # 用 PyTorch 的注意力函数逐头对照，检查 KV 头有没有分对
        heads = [F.scaled_dot_product_attention(q[:, head], k[:, head // 2], v[:, head // 2],
                                                attn_mask=~blocked) for head in range(4)]
        expected = attention.o_proj(torch.stack(heads, dim=2).reshape(2, 5, 32))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_future_tokens_do_not_change_prefix(self):
        changed = self.tokens.clone()
        changed[:, 6:] = (changed[:, 6:] + 1) % self.config.vocab_size
        original = self.model(self.tokens).logits
        altered = self.model(changed).logits
        torch.testing.assert_close(original[:, :6], altered[:, :6], atol=0, rtol=0)
        self.assertGreater((original[:, 6:] - altered[:, 6:]).abs().max().item(), 1e-4)

    @torch.inference_mode()
    def test_cache_matches_full_sequence_for_mha_gqa_mqa(self):
        for kv_heads in (4, 2, 1):
            model = MiniLlama(ModelConfig(**(asdict(self.config) | {"n_kv_heads": kv_heads}))).eval()
            expected = model(self.tokens).logits
            for chunks in ((1,) * 11, (4, 3, 4), (11,)):
                with self.subTest(kv_heads=kv_heads, chunks=chunks):
                    cache, start, pieces = None, 0, []
                    for count in chunks:
                        output = model(self.tokens[:, start:start + count], past_key_values=cache, use_cache=True)
                        cache = output.past_key_values
                        start += count
                        pieces.append(output.logits)
                        self.assertEqual(cache[0][0].shape, (2, kv_heads, start, 8))
                    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, atol=2e-6, rtol=2e-5)

    @torch.inference_mode()
    def test_cache_is_explicit_and_not_mutated(self):
        prefix = self.model(self.tokens[:, :4], use_cache=True)
        original_key = prefix.past_key_values[0][0].clone()
        first = self.model(self.tokens[:, 4:7], past_key_values=prefix.past_key_values, use_cache=True)
        second = self.model(self.tokens[:, 4:7], past_key_values=prefix.past_key_values, use_cache=True)
        torch.testing.assert_close(first.logits, second.logits, atol=0, rtol=0)
        torch.testing.assert_close(prefix.past_key_values[0][0], original_key, atol=0, rtol=0)
        self.assertEqual(prefix.past_key_values[0][0].shape[2], 4)
        torch.testing.assert_close(self.model(self.tokens).logits[:, 4:7], first.logits)

    @torch.inference_mode()
    def test_invalid_cache_rejected(self):
        cache = self.model(self.tokens[:, :3], use_cache=True).past_key_values
        bad_length = (cache[0], (cache[1][0][:, :, :2], cache[1][1][:, :, :2]))
        bad_dtype = tuple((k.double(), v.double()) for k, v in cache)
        for invalid in (cache[:1], bad_length, bad_dtype):
            with self.subTest(cache_shape=invalid[0][0].shape), self.assertRaises(ValueError):
                self.model(self.tokens[:, 3:4], past_key_values=invalid, use_cache=True)
        with self.assertRaises(ValueError):
            self.model(self.tokens[:1, 3:4], past_key_values=cache, use_cache=True)
        with self.assertRaises(ValueError):
            self.model(self.tokens[:, 3:4], past_key_values=cache)

    def test_cache_requires_inference(self):
        with self.assertRaises(ValueError):
            self.model(self.tokens, use_cache=True)

    def test_context_limit_and_inputs(self):
        for invalid in (torch.zeros(1, 33, dtype=torch.long), torch.zeros(1, 0, dtype=torch.long),
                        torch.zeros(11, dtype=torch.long), self.tokens.float()):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                self.model(invalid)
        with self.assertRaises(ValueError):
            self.model(self.tokens, self.tokens[:, :-1])
        with torch.inference_mode():
            tokens = torch.zeros(1, 32, dtype=torch.long)
            cache = self.model(tokens[:, :31], use_cache=True).past_key_values
            self.model(tokens[:, 31:], past_key_values=cache, use_cache=True)
            with self.assertRaises(ValueError):
                self.model(tokens[:, :2], past_key_values=cache, use_cache=True)

    def test_all_parameters_receive_finite_gradients(self):
        output = self.model(self.tokens[:, :-1], self.tokens[:, 1:])
        self.assertEqual(output.logits.shape, (2, 10, 24))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_fixed_batch_overfits(self):
        stream = torch.tensor([[1, 2, 3, 4] * 5], dtype=torch.long)
        x, y = stream[:, :-1], stream[:, 1:]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        initial = self.model(x, y).loss.item()
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            loss = self.model(x, y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        final = self.model(x, y).loss.item()
        self.assertLess(final, initial * 0.1)
        self.assertLess(final, 0.1)

    def test_batch_targets_are_shifted_once(self):
        data = torch.arange(100)
        x, y = sample_batch(data, 8, 16, torch.Generator().manual_seed(0), "cpu")
        torch.testing.assert_close(y, x + 1)
        torch.testing.assert_close(x[:, 1:], y[:, :-1])
        with self.assertRaises(ValueError):
            sample_batch(data[:16], 1, 16, torch.Generator(), "cpu")

    def test_evaluate_restores_training_mode(self):
        self.model.train()
        value = evaluate(self.model, [(self.tokens[:, :-1], self.tokens[:, 1:])])
        self.assertTrue(math.isfinite(value))
        self.assertTrue(self.model.training)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_generation_cache_parity_and_mode(self):
        prompt = self.tokens[:, :4]
        cached = generate_tokens(self.model, prompt, 8, temperature=0, use_cache=True)
        full = generate_tokens(self.model, prompt, 8, temperature=0, use_cache=False)
        torch.testing.assert_close(cached, full, atol=0, rtol=0)
        torch.testing.assert_close(cached[:, :4], prompt, atol=0, rtol=0)
        self.assertEqual(cached.shape, (2, 12))
        self.assertTrue(self.model.training)
        torch.testing.assert_close(generate_tokens(self.model, prompt, 0), prompt)

    def test_generation_input_limits(self):
        for kwargs in ({"max_new_tokens": -1}, {"max_new_tokens": 40}, {"temperature": -1},
                       {"temperature": float("nan")}, {"top_k": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_tokens(self.model, self.tokens, **kwargs)
        with self.assertRaises(ValueError):
            generate_tokens(self.model, self.tokens[:, :0])

    def test_top_k_sampling(self):
        logits = torch.tensor([[1., 9., 3., 8.]])
        for _ in range(30):
            self.assertIn(sample_next_token(logits, top_k=2).item(), (1, 3))
        self.assertEqual(sample_next_token(logits, temperature=0).item(), 1)
        self.assertEqual(sample_next_token(logits, top_k=1).item(), 1)
        self.assertLess(sample_next_token(logits, top_k=100).item(), 4)
        self.assertLess(sample_next_token(logits, top_k=None).item(), 4)

    def test_tokenizer_round_trip_and_unknown(self):
        text = "你好，LLaMA！\n🙂 "
        tokenizer = CharTokenizer.from_text(text)
        restored = CharTokenizer.from_dict(tokenizer.to_dict())
        self.assertEqual(restored.decode(restored.encode(text)), text)
        self.assertEqual(restored.encode("陌"), [0])
        self.assertEqual(restored.decode([0]), "[UNK]")
        self.assertEqual(restored.encode(text), tokenizer.encode(text))
        with self.assertRaises(ValueError):
            CharTokenizer(["a", "a"])
        with self.assertRaises(ValueError):
            restored.decode([restored.vocab_size])

    def test_checkpoint_round_trip(self):
        tokenizer = CharTokenizer.from_text("春天来了，小猫坐在窗边。\n")
        model = MiniLlama(ModelConfig(**(asdict(self.config) | {"vocab_size": tokenizer.vocab_size})))
        tokens = torch.tensor([tokenizer.encode("春天来了，")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"format_version": 1, "config": asdict(model.config),
                        "tokenizer": tokenizer.to_dict(), "model_state_dict": model.state_dict()}, path)
            loaded, loaded_tokenizer = load_checkpoint(path)
            torch.testing.assert_close(model(tokens).logits, loaded(tokens).logits, atol=0, rtol=0)
            self.assertEqual(loaded_tokenizer.encode("春天来了，"), tokens[0].tolist())
            self.assertFalse(loaded.training)


if __name__ == "__main__":
    unittest.main()
