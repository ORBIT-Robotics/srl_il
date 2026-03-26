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


if __name__ == "__main__":
    unittest.main()
