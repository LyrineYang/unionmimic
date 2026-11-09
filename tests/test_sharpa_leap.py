import math
import unittest

import torch

from unionmimic.sharpa_leap import L, physical_state, object_root_from_physical, SharpaCommandMap


class SharpaLeapContract(unittest.TestCase):
    def test_known_frame_and_world_angular_velocity(self):
        q = torch.arange(22).float()[None]
        base = torch.zeros(1, 13);base[:, :3] = torch.tensor([3., 4., 5.])
        base[:, 5:7] = math.sqrt(.5)  # +90 degrees about world z
        obj = base.clone();obj[:, :3] += torch.tensor([0., 1., 0.])
        obj[:, 7:10] = torch.tensor([0., 2., 0.])
        obj[:, 10:13] = torch.tensor([0., 3., 0.])
        s = physical_state(q, q+1, q+2, obj, base)
        torch.testing.assert_close(s[:, L['p']], torch.tensor([[1., 0., 0.]]))
        torch.testing.assert_close(s[:, L['v']], torch.tensor([[2., 0., 0.]]))
        torch.testing.assert_close(s[:, L['w']], torch.tensor([[3., 0., 0.]]))
        torch.testing.assert_close(s[:, L['quat']], torch.tensor([[1., 0., 0., 0.]]))
        torch.testing.assert_close(s[:, L['q']], q)
        torch.testing.assert_close(s[:, L['cmd']], q+2)
        torch.testing.assert_close(object_root_from_physical(s, base), obj)

    def test_quaternion_sign_and_moving_base_rejection(self):
        q = torch.zeros(2, 22);base = torch.zeros(2, 13);base[:, 6] = 1
        obj = base.clone();obj[0, 6] = -1
        s = physical_state(q,q,q,obj,base)
        torch.testing.assert_close(s[0],s[1])
        base[1, 7] = .1
        with self.assertRaises(ValueError):physical_state(q,q,q,obj,base)

    def test_controller_memory_limits_and_input_gradient(self):
        controller = SharpaCommandMap([-1]*22,[1]*22)
        target = torch.zeros(1,22);command = torch.ones(1,22,requires_grad=True)
        first,a = controller(target, command, torch.zeros_like(target), .1)
        second,b = controller(first, -command, a, .1)
        torch.testing.assert_close(first, torch.full_like(first,.08))
        torch.testing.assert_close(b, torch.full_like(b,-.64))
        torch.testing.assert_close(second, torch.full_like(second,.016))
        first.sum().backward();self.assertTrue((command.grad>0).all())
        limited,_ = controller(torch.ones_like(target), command.detach(), a, .1)
        torch.testing.assert_close(limited,torch.ones_like(limited))


if __name__ == '__main__':unittest.main()
