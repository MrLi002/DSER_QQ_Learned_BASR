"""CPU tests: python -m unittest discover -s tests -v (no data or weights)."""
import copy
import gc
import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from model.selective_basr import (
    temporal_event_features, select_topk_patches, DifficultyHead,
    TargetGridRetriever, LearnedDetailsSup, basr_auxiliary_losses)
from model.checkpoints import (load_network_state, configure_inference_args,
                               validate_resume_args, checkpoint_revision)
from model.ourNet import MyNet

torch.set_num_threads(2)


def args(mode="learned", **extra):
    settings = dict(bins=8, beta=0.1, mask_patch_size=32, basr_mode=mode,
                    repair_ratio=0.25, repair_residual_radius=2., repair_chunk_size=17)
    settings.update(extra)
    return SimpleNamespace(**settings)


class SelectionTests(unittest.TestCase):
    def test_topk_independent_per_image_and_padding(self):
        error = torch.zeros(2, 1, 8, 8)
        error[0, :, :4, :4] = 1
        error[1, :, 4:, 4:] = 1
        mask, _ = select_topk_patches(error, 4, .25)
        self.assertTrue(torch.equal(mask, error))
        valid = torch.ones_like(error)
        valid[:, :, :4, :4] = 0
        masked, _ = select_topk_patches(error, 4, 1, valid)
        self.assertTrue(torch.equal(masked, valid))
        zero, _ = select_topk_patches(error, 4, 0)
        self.assertEqual(zero.sum().item(), 0)

    def test_partial_patches_and_all_invalid(self):
        e = torch.ones(1, 1, 7, 9)
        mask, scores = select_topk_patches(e, 4, 1)
        self.assertEqual(mask.shape, e.shape)
        self.assertEqual(scores.shape[-2:], (2, 3))
        self.assertTrue(torch.equal(mask, e))
        mask, _ = select_topk_patches(e, 4, 1, torch.zeros_like(e))
        self.assertEqual(mask.sum().item(), 0)

    def test_empty_events_and_time_order(self):
        empty = torch.zeros(1, 8, 8, 8)
        early, late = empty.clone(), empty.clone()
        early[:, 0] = 1
        late[:, -1] = 1
        features = temporal_event_features(early, late)
        self.assertEqual(features.shape, (1, 7, 8, 8))
        self.assertTrue(torch.isfinite(temporal_event_features(empty, empty)).all())
        self.assertEqual(features[:, 1].mean().item(), 0)
        self.assertEqual(features[:, 4].mean().item(), 1)

    def test_difficulty_training_detaches_coarse_target(self):
        head = DifficultyHead()
        coarse = torch.rand(1, 3, 8, 8, requires_grad=True)
        image = torch.rand_like(coarse)
        predicted = head(image, image, coarse, torch.zeros(1, 7, 8, 8))
        aux = dict(difficulty=predicted, stages=[])
        losses = basr_auxiliary_losses(aux, coarse, image, patch_size=4)
        losses["difficulty"].backward()
        self.assertIsNone(coarse.grad)
        self.assertGreater(head.net[-1].bias.grad.abs().sum().item(), 0)


class RetrievalTests(unittest.TestCase):
    def test_offset_order_matches_torchvision_and_target_grid(self):
        c, h, w = 4, 6, 7
        module = TargetGridRetriever(c, residual_radius=0, chunk_size=1)
        with torch.no_grad():
            module.value.weight.copy_(torch.eye(c).view(c, c, 1, 1))
            module.value.bias.zero_()
        source = torch.arange(h * w).float().view(1, 1, h, w).repeat(1, c, 1, 1)
        rgb = source[:, :3] / (h * w)
        offsets = torch.full((1, 18, h, w), 1000.)
        offsets[:, 8] = 1  # center sample: dy=1
        offsets[:, 9] = 2  # center sample: dx=2
        selected = torch.zeros(1, 1, h, w)
        selected[:, :, 2, 2] = 1
        support, colors, valid, _ = module(source, source, rgb, offsets,
                                           torch.zeros(1, 7, h, w), selected)
        modulation = torch.zeros(1, 9, h, w)
        modulation[:, 4] = 1
        reference = deform_conv2d(source[:, :1], offsets, torch.ones(1, 1, 3, 3),
                                  padding=(1, 1), mask=modulation)
        self.assertAlmostEqual(support[0, 0, 2, 2].item(), reference[0, 0, 2, 2].item(), places=5)
        self.assertAlmostEqual(support[0, 0, 2, 2].item(), source[0, 0, 3, 4].item(), places=5)
        self.assertAlmostEqual(colors[0, 0, 2, 2].item(), rgb[0, 0, 3, 4].item(), places=5)
        self.assertEqual(valid.sum().item(), 1)

    def _block_inputs(self):
        return dict(target=torch.rand(1, 4, 8, 8), source0=torch.rand(1, 4, 8, 8),
                    source1=torch.rand(1, 4, 8, 8), base_high=torch.rand(1, 8, 16, 16),
                    offsets0=torch.zeros(1, 18, 8, 8), offsets1=torch.zeros(1, 18, 8, 8),
                    temporal=torch.rand(1, 7, 16, 16), selected=torch.zeros(1, 1, 16, 16),
                    difficulty=torch.rand(1, 1, 16, 16), rgb0=torch.rand(1, 3, 16, 16),
                    rgb1=torch.rand(1, 3, 16, 16), base_rgb=torch.rand(1, 3, 8, 8))

    def test_empty_selection_and_outside_region_preserved(self):
        block = LearnedDetailsSup(4, 8, chunk_size=5)
        torch.nn.init.normal_(block.to_high.weight, std=.1)
        inputs = self._block_inputs()
        out, _ = block(**inputs)
        self.assertTrue(torch.equal(out, inputs["base_high"]))
        inputs["selected"][:, :, :8, :8] = 1
        out, aux = block(**inputs)
        outside = (inputs["selected"] == 0).expand_as(out)
        self.assertTrue(torch.equal(out[outside], inputs["base_high"][outside]))
        self.assertTrue(torch.allclose(aux["weights"].sum(1), torch.ones(1, 8, 8)))
        self.assertGreater((out - inputs["base_high"]).abs().sum().item(), 0)

    def test_no_source_support_falls_back_without_nan(self):
        block = LearnedDetailsSup(4, 8)
        torch.nn.init.normal_(block.to_high.weight, std=.1)
        inputs = self._block_inputs()
        inputs["selected"].fill_(1)
        inputs["offsets0"].fill_(1000)
        inputs["offsets1"].fill_(1000)
        out, aux = block(**inputs)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.equal(aux["weights"][:, 2], torch.ones(1, 8, 8)))
        self.assertTrue(torch.equal(out, inputs["base_high"]))

    def test_source_padding_is_not_used(self):
        module = TargetGridRetriever(4)
        source = torch.ones(1, 4, 8, 8)
        out, _, valid, _ = module(source, source, source[:, :3],
            torch.zeros(1, 18, 8, 8), torch.zeros(1, 7, 8, 8),
            torch.ones(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
        self.assertEqual(out.abs().sum().item(), 0)
        self.assertEqual(valid.sum().item(), 0)

    def test_chunked_retrieval_matches_unchunked(self):
        a = TargetGridRetriever(4, chunk_size=3)
        b = copy.deepcopy(a)
        b.chunk_size = 10000
        inputs = (torch.rand(1, 4, 8, 8), torch.rand(1, 4, 8, 8),
                  torch.rand(1, 3, 8, 8), torch.rand(1, 18, 8, 8) * .1,
                  torch.rand(1, 7, 8, 8), torch.ones(1, 1, 8, 8))
        for left, right in zip(a(*inputs), b(*inputs)):
            self.assertTrue(torch.allclose(left, right, atol=1e-6))

    def test_retrieval_learns_after_zero_projection_initialization(self):
        block = LearnedDetailsSup(4, 8, chunk_size=32)
        inputs = self._block_inputs()
        inputs["selected"].fill_(1)
        optimizer = torch.optim.SGD(block.parameters(), lr=.1)
        for _ in range(2):
            optimizer.zero_grad()
            out, _ = block(**inputs)
            out.square().mean().backward()
            optimizer.step()
        for name in ("retriever.query.weight", "retriever.value.weight",
                     "retriever.offset_residual.2.weight", "decision.2.weight"):
            gradient = dict(block.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)


class IntegrationTests(unittest.TestCase):
    def test_full_forward_backward_and_no_gt_input_leak(self):
        from model.loss import MaskedSSIMLoss
        from train_bsergb import compute_losses
        model = MyNet(args()).eval()
        imgs = torch.rand(1, 9, 64, 64)
        voxels = torch.randn(1, 24, 64, 64)
        mask = torch.ones(1, 1, 64, 64)
        with torch.no_grad():
            first = model(imgs, voxels, mask, 8, return_aux=True)
            changed = imgs.clone()
            changed[:, 3:6] = torch.rand_like(changed[:, 3:6])
            second = model(changed, voxels, mask, 8, return_aux=True)
            for left, right in zip(first[3], second[3]):
                self.assertTrue(torch.equal(left, right))
            self.assertTrue(torch.equal(first[6]["difficulty"], second[6]["difficulty"]))
        del first, second
        model.train()
        output = model(imgs, voxels, mask, 8, return_aux=True)
        self.assertEqual([p.shape[-2:] for p in output[3]], [(64, 64), (32, 32), (16, 16)])
        weights = args(lambda_rec=1., lambda_syn=1., lambda_ref=1., lambda_ps=1.,
                       lambda_difficulty=1., lambda_route=.05, lambda_gate=.05)
        class PerceptualStub(torch.nn.Module):
            def forward(self, a, b):
                return (a - b).square().mean((1, 2, 3), keepdim=True)
        losses = compute_losses(output, imgs[:, 3:6], mask, PerceptualStub(),
                                MaskedSSIMLoss(), weights)
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
        losses["total"].backward()
        for name in ("event_rec.conv3.weight", "TNet.lastconv.weight",
                     "difficulty_head.net.4.bias", "sys.refine_block0.to_high.weight",
                     "sys.refine_block0.decision.2.bias"):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)
        del model, output, losses
        gc.collect()


class V2CorrectionTests(unittest.TestCase):
    def _inputs(self, selected=True):
        inputs = RetrievalTests()._block_inputs()
        inputs["selected"].fill_(int(selected))
        inputs["modulation0"] = torch.randn(1, 9, 8, 8, requires_grad=True)
        inputs["modulation1"] = torch.randn(1, 9, 8, 8, requires_grad=True)
        inputs["offsets0"].requires_grad_()
        inputs["offsets1"].requires_grad_()
        return inputs

    def test_residual_receives_coarse_aligned_source(self):
        retriever = TargetGridRetriever(4, residual_radius=0)
        source = torch.arange(56).float().view(1, 1, 7, 8).repeat(1, 4, 1, 1)
        offsets = torch.full((1, 18, 7, 8), 1000.)
        offsets[:, 8] = 0
        offsets[:, 9] = 2
        selected = torch.zeros(1, 1, 7, 8)
        selected[:, :, 3, 2] = 1
        captured = []
        handle = retriever.offset_residual[0].register_forward_pre_hook(
            lambda module, values: captured.append(values[0].detach()))
        retriever(torch.zeros_like(source), source, source[:, :3] / 56,
                  offsets, torch.zeros(1, 7, 7, 8), selected)
        handle.remove()
        aligned = captured[0][:, 4:8, 3, 2]
        self.assertTrue(torch.allclose(aligned, source[:, :, 3, 4], atol=1e-5))
        self.assertFalse(torch.allclose(aligned, source[:, :, 3, 2]))
        self.assertEqual(captured[0][0, -1, 3, 2].item(), 1)

    def test_direct_rgb_loss_trains_retrieval_at_initialization(self):
        torch.manual_seed(9)
        block = LearnedDetailsSup(4, 8, chunk_size=16)
        inputs = self._inputs()
        _, stage = block(**inputs)
        aux = dict(difficulty=torch.rand(1, 1, 16, 16, requires_grad=True), stages=[stage])
        losses = basr_auxiliary_losses(aux, torch.rand(1, 3, 16, 16),
                                       torch.rand(1, 3, 16, 16), patch_size=4)
        # No main, routing or gate objective participates in this backward.
        losses["retrieve"].backward()
        for name in ("retriever.query.weight", "retriever.key.weight",
                     "retriever.offset_residual.2.weight",
                     "retriever.modulation_projection.weight", "retriever.prior_logit"):
            gradient = dict(block.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)
        self.assertIsNone(block.to_high.weight.grad)
        self.assertIsNone(block.decision[-1].weight.grad)
        value_gradient = block.retriever.value.weight.grad
        self.assertTrue(value_gradient is None or value_gradient.abs().sum().item() == 0)
        self.assertIsNone(inputs["offsets0"].grad)
        self.assertIsNone(inputs["modulation0"].grad)

    def test_decision_sees_rgb_and_rgb_differences(self):
        block = LearnedDetailsSup(4, 8)
        inputs = self._inputs()
        captured = []
        handle = block.decision[0].register_forward_pre_hook(
            lambda module, values: captured.append(values[0].detach()))
        _, stage = block(**inputs)
        handle.remove()
        evidence = captured[0][:, 3 * 4 + 12:]
        expected = torch.cat((stage["rgb0"], stage["rgb1"], stage["base_rgb"],
                              (stage["rgb0"] - stage["base_rgb"]).abs(),
                              (stage["rgb1"] - stage["base_rgb"]).abs()), 1)
        self.assertEqual(evidence.shape[1], 15)
        self.assertTrue(torch.equal(evidence, expected.detach()))

    def test_signed_modulation_prior_is_finite_and_optional(self):
        module = TargetGridRetriever(4, residual_radius=0)
        source = torch.rand(1, 4, 8, 8)
        arguments = (source, source, source[:, :3], torch.zeros(1, 18, 8, 8),
                     torch.zeros(1, 7, 8, 8), torch.ones(1, 1, 8, 8))
        modulation = torch.randn(1, 9, 8, 8) * 10 - 5
        prior = module._modulation_prior(modulation, source)
        self.assertTrue(torch.isfinite(prior).all())
        self.assertTrue(torch.isfinite(module(*arguments, modulation=modulation)[0]).all())
        self.assertEqual(module._modulation_prior(None, source).abs().sum().item(), 0)

    def test_rgb_loss_skips_invalid_source_and_all_invalid(self):
        rgb0 = torch.ones(1, 3, 4, 4, requires_grad=True)
        rgb1 = torch.ones(1, 3, 4, 4, requires_grad=True) * .5
        stage = dict(rgb0=rgb0, rgb1=rgb1, base_rgb=torch.zeros(1, 3, 4, 4),
                     selected=torch.ones(1, 1, 4, 4), valid0=torch.zeros(1, 1, 4, 4),
                     valid1=torch.ones(1, 1, 4, 4), route_logits=torch.zeros(1, 3, 4, 4),
                     gate_logits=torch.zeros(1, 1, 4, 4), weights=torch.ones(1, 3, 4, 4) / 3,
                     gate=torch.ones(1, 1, 4, 4) / 2)
        aux = dict(difficulty=torch.zeros(1, 1, 4, 4), stages=[stage])
        losses = basr_auxiliary_losses(aux, torch.zeros(1, 3, 4, 4),
                                       torch.zeros(1, 3, 4, 4), patch_size=4)
        self.assertAlmostEqual(losses["retrieve"].item(), .499001, places=5)
        losses["retrieve"].backward()
        self.assertEqual(rgb0.grad.abs().sum().item(), 0)
        stage["valid1"].zero_()
        losses = basr_auxiliary_losses(aux, torch.zeros(1, 3, 4, 4),
                                       torch.zeros(1, 3, 4, 4), patch_size=4)
        self.assertEqual(losses["retrieve"].item(), 0)
        self.assertEqual(losses["support_valid"].item(), 0)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_usage_metrics_ignore_unselected_pixels(self):
        block = LearnedDetailsSup(4, 8)
        inputs = self._inputs()
        inputs["selected"][:, :, 8:] = 0
        _, stage = block(**inputs)
        losses = basr_auxiliary_losses(
            dict(difficulty=inputs["difficulty"], stages=[stage]),
            torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16), patch_size=4)
        self.assertAlmostEqual(losses["source_usage"].item(), .21301396, places=5)
        self.assertAlmostEqual(losses["gate_mean"].item(), .11920292, places=5)
        self.assertAlmostEqual(losses["effective_usage"].item(), .0253919, places=5)
        self.assertEqual(losses["support_valid"].item(), 1)
        self.assertFalse(losses["gate_mean"].requires_grad)

    def test_v1_migration_is_explicit_and_preserves_difficulty(self):
        model = MyNet(args())
        state = dict(model.state_dict())
        v1 = {key: value for key, value in state.items()
              if ".retriever.modulation_projection." not in key and not key.endswith(".retriever.prior_logit")}
        self.assertEqual(checkpoint_revision(v1), 1)
        with self.assertRaisesRegex(RuntimeError, "modes differ"):
            load_network_state(model, v1)
        report = load_network_state(model, v1, allow_basr_migration=True)
        self.assertEqual((report["source_revision"], report["target_revision"]), (1, 2))
        self.assertTrue(all(key.startswith(("sys.refine_block0.", "sys.refine_block1."))
                            for key in report["missing"]))
        with self.assertRaisesRegex(ValueError, "V1"):
            configure_inference_args(SimpleNamespace(), v1)
        with self.assertRaisesRegex(ValueError, "V1"):
            validate_resume_args(args(), dict(net=v1))
        broken = {key: value for key, value in v1.items() if key != "difficulty_head.net.0.weight"}
        with self.assertRaisesRegex(RuntimeError, "Backbone/checkpoint mismatch"):
            load_network_state(model, broken, allow_basr_migration=True)
        del model, state, v1
        gc.collect()

    def test_original_checkpoint_migration_and_strict_backbone(self):
        original = MyNet(args("original"))
        state = original.state_dict()
        learned = MyNet(args())
        report = load_network_state(learned, state, allow_basr_migration=True)
        self.assertTrue(report["migrated"])
        self.assertTrue(torch.equal(learned.event_rec.conv1.weight, original.event_rec.conv1.weight))
        self.assertTrue(all(k.startswith(("difficulty_head.", "sys.refine_block0.",
                                         "sys.refine_block1.")) for k in report["missing"]))
        broken = {k: v for k, v in state.items() if k != "event_rec.conv1.weight"}
        with self.assertRaisesRegex(RuntimeError, "Backbone/checkpoint mismatch"):
            load_network_state(learned, broken, allow_basr_migration=True)
        configured = configure_inference_args(SimpleNamespace(), {"net": learned.state_dict(),
                                           "args": dict(repair_ratio=.5, bins=8)})
        self.assertEqual(configured.basr_mode, "learned")
        self.assertEqual(configured.repair_ratio, .5)
        del original, learned, state
        gc.collect()


if __name__ == "__main__":
    unittest.main()
