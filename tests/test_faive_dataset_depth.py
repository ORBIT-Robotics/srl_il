import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch

from srl_il.dataset.faive_dataset import FaiveTrajectorySequenceDataset


class TestFaiveDatasetDepth(unittest.TestCase):
    def test_load_depth_and_color_from_orbit_format(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            h5_path = tmp_path / "episode_0001.h5"

            color = np.random.randint(0, 255, (4, 6, 8, 3), dtype=np.uint8)
            depth_mm = np.array(
                [
                    [[0, 1000], [2000, 3000]],
                    [[4000, 5000], [6000, 7000]],
                    [[8000, 9000], [10000, 11000]],
                    [[12000, 13000], [14000, 15000]],
                ],
                dtype=np.uint16,
            )

            with h5py.File(h5_path, "w") as h5_file:
                h5_file.create_dataset("actions_ikarus_arm", data=np.zeros((4, 12), dtype=np.float32))
                h5_file.create_dataset("actions_ikarus_hand", data=np.zeros((4, 32), dtype=np.float32))
                obs = h5_file.create_group("observations")
                obs.create_dataset("qpos_ikarus_arm", data=np.zeros((4, 12), dtype=np.float32))
                obs.create_dataset("qpos_ikarus_hand", data=np.zeros((4, 32), dtype=np.float32))
                images = obs.create_group("images")
                head = images.create_group("head")
                head.create_dataset("color", data=color)
                head.create_dataset("depth", data=depth_mm)

            ds = FaiveTrajectorySequenceDataset(tmp_path)
            self.assertIn("head/color", ds.traj_data)
            self.assertIn("head/depth", ds.traj_data)

            traj_data, _ = ds[0]
            color_tensor = ds.load(traj_data["head/color"], "head/color")
            depth_tensor = ds.load(traj_data["head/depth"], "head/depth")

            self.assertEqual(tuple(color_tensor.shape), (4, 3, 6, 8))
            self.assertEqual(tuple(depth_tensor.shape), (4, 1, 2, 2))
            self.assertEqual(color_tensor.dtype, torch.float32)
            self.assertEqual(depth_tensor.dtype, torch.float32)

            expected_depth_m = torch.tensor(depth_mm, dtype=torch.float32).unsqueeze(1) / 1000.0
            torch.testing.assert_close(depth_tensor, expected_depth_m)

    def test_load_root_schema_depth_and_color(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            h5_path = tmp_path / "episode_0002.h5"

            # Keep T small for test runtime; field layouts match production schema.
            timesteps = 6
            camera1 = np.random.randint(0, 255, (timesteps, 12, 16, 3), dtype=np.uint8)
            camera2 = np.random.randint(0, 255, (timesteps, 10, 14, 3), dtype=np.uint8)
            depth_mm = np.random.randint(0, 4000, (timesteps, 8, 12), dtype=np.uint16)
            pose_r = np.random.randn(timesteps, 7).astype(np.float32)
            hand_joints = np.random.randn(timesteps, 17).astype(np.float32)
            joint_states = np.random.randn(timesteps, 5).astype(np.float32)
            timestamps = np.arange(timesteps, dtype=np.float64) * 0.05

            with h5py.File(h5_path, "w") as h5_file:
                h5_file.create_dataset("camera1", data=camera1)
                h5_file.create_dataset("camera1_depth", data=depth_mm)
                h5_file.create_dataset("camera2", data=camera2)
                h5_file.create_dataset("poseR", data=pose_r)
                h5_file.create_dataset("hand_joints", data=hand_joints)
                h5_file.create_dataset("joint_states", data=joint_states)
                h5_file.create_dataset("timestamps", data=timestamps)

            ds = FaiveTrajectorySequenceDataset(tmp_path)
            self.assertIn("camera1/color", ds.traj_data)
            self.assertIn("camera2/color", ds.traj_data)
            self.assertIn("camera1_depth", ds.traj_data)
            self.assertIn("joint_states", ds.traj_data)
            self.assertIn("poseR", ds.traj_data)
            self.assertIn("hand_joints", ds.traj_data)
            self.assertIn("timestamps", ds.traj_data)

            traj_data, _ = ds[0]
            color1_tensor = ds.load(traj_data["camera1/color"], "camera1/color")
            color2_tensor = ds.load(traj_data["camera2/color"], "camera2/color")
            depth_tensor = ds.load(traj_data["camera1_depth"], "camera1_depth")

            self.assertEqual(tuple(color1_tensor.shape), (timesteps, 3, 12, 16))
            self.assertEqual(tuple(color2_tensor.shape), (timesteps, 3, 10, 14))
            self.assertEqual(tuple(depth_tensor.shape), (timesteps, 1, 8, 12))
            self.assertEqual(color1_tensor.dtype, torch.float32)
            self.assertEqual(color2_tensor.dtype, torch.float32)
            self.assertEqual(depth_tensor.dtype, torch.float32)

            expected_depth_m = torch.tensor(depth_mm, dtype=torch.float32).unsqueeze(1) / 1000.0
            torch.testing.assert_close(depth_tensor, expected_depth_m)


if __name__ == "__main__":
    unittest.main()
