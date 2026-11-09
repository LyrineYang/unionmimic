"""Contracts for causal compact context and differentiable observation noise."""
import io
import unittest

import torch

from unionmimic.geometry_context import PositionErrorBridge, PositionReadout


class GeometryContextTests(unittest.TestCase):
    def test_history_order_and_checkpoint_input_gradients(self):
        # Select oldest history, newest history, and sensor feature independently.
        weight = torch.zeros(5, 3)
        weight[0, 0] = weight[3, 1] = weight[4, 2] = 1
        head = PositionReadout(torch.zeros(5), torch.ones(5), weight, torch.zeros(3))
        buf = io.BytesIO(); torch.save(head.checkpoint(), buf); buf.seek(0)
        restored = PositionReadout.from_checkpoint(torch.load(buf))
        history = torch.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
        embed = torch.tensor([[5.]], requires_grad=True)
        value = restored(history, embed)
        torch.testing.assert_close(value, torch.tensor([[1., 4., 5.]]))
        value.sum().backward()
        torch.testing.assert_close(history.grad, torch.tensor([[[1., 0.], [0., 1.]]]))
        torch.testing.assert_close(embed.grad, torch.ones_like(embed))
        self.assertEqual(list(restored.parameters()), [])

    def test_bridge_conditional_mean_and_gradient_through_horizon(self):
        bridge = PositionErrorBridge(torch.zeros(3), torch.tensor([.5, .25, 0.]), torch.eye(3)*.01)
        bridge = PositionErrorBridge.from_checkpoint(bridge.checkpoint())
        pos = torch.zeros(1, 3, requires_grad=True)
        error = torch.ones(1, 3)
        for _ in range(4):
            context, error = bridge.step(pos, error, torch.zeros_like(error))
        torch.testing.assert_close(context, torch.tensor([[.0625, .00390625, 0.]]))
        context.sum().backward()
        torch.testing.assert_close(pos.grad, torch.ones_like(pos))
        self.assertEqual(list(bridge.parameters()), [])

    def test_bridge_correlated_noise_orientation(self):
        chol = torch.tensor([[1., 0., 0.], [.5, 2., 0.], [0., .3, 3.]])
        bridge = PositionErrorBridge(torch.zeros(3), torch.zeros(3), chol)
        _, error = bridge.step(torch.zeros(3, 3), torch.zeros(3, 3), torch.eye(3))
        torch.testing.assert_close(error.T @ error, chol @ chol.T)


if __name__ == '__main__':
    unittest.main()
