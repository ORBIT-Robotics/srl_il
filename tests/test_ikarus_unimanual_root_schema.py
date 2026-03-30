import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from srl_il.algo.act import ACT
from srl_il.dataset.faive_dataset import faive_train_val_test
from srl_il.models.common.linear_normalizer import LinearNormalizer


class TestIkarusUnimanualRootSchema(unittest.TestCase):
    def _write_root_schema_episode(self, out_dir: Path, timesteps: int = 24) -> None:
        h5_path = out_dir / "episode_0001.h5"
        with h5py.File(h5_path, "w") as h5_file:
            h5_file.create_dataset(
                "camera1",
                data=np.random.randint(0, 255, (timesteps, 32, 48, 3), dtype=np.uint8),
            )
            h5_file.create_dataset(
                "camera1_depth",
                data=np.random.randint(0, 5000, (timesteps, 20, 32), dtype=np.uint16),
            )
            h5_file.create_dataset(
                "camera2",
                data=np.random.randint(0, 255, (timesteps, 24, 40, 3), dtype=np.uint8),
            )
            h5_file.create_dataset("hand_joints", data=np.random.randn(timesteps, 17).astype(np.float32))
            h5_file.create_dataset("joint_states", data=np.random.randn(timesteps, 5).astype(np.float32))
            h5_file.create_dataset("poseR", data=np.random.randn(timesteps, 7).astype(np.float32))
            h5_file.create_dataset("timestamps", data=np.arange(timesteps, dtype=np.float64) * 0.05)

    def _build_datasets_from_cfg(self, data_directory: Path):
        cfg_path = Path(__file__).resolve().parents[1] / "srl_il" / "cfg" / "ikarus_unimanual.yaml"
        cfg = OmegaConf.load(cfg_path)
        data_cfg = OmegaConf.to_container(cfg.dataset_cfg.data, resolve=True)
        data_cfg.pop("_target_", None)
        data_cfg["data_directory"] = str(data_directory)
        data_cfg["test_fraction"] = 0.0
        data_cfg["val_fraction"] = 0.2
        return faive_train_val_test(**data_cfg)

    def test_config_data_pipeline_smoke(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self._write_root_schema_episode(tmp_path)
            datasets = self._build_datasets_from_cfg(tmp_path)

            batch, mask_batch = datasets.train_data[0]
            expected_keys = {
                "poseR",
                "hand_joints",
                "joint_states",
                "poseR_obs",
                "hand_joints_obs",
                "camera1",
                "camera2",
                "camera1_depth",
            }
            self.assertEqual(set(batch.keys()), expected_keys)
            self.assertEqual(set(mask_batch.keys()), expected_keys)

            self.assertEqual(tuple(batch["poseR"].shape), (20, 7))
            self.assertEqual(tuple(batch["hand_joints"].shape), (20, 17))
            self.assertEqual(tuple(batch["joint_states"].shape), (1, 5))
            self.assertEqual(tuple(batch["poseR_obs"].shape), (1, 7))
            self.assertEqual(tuple(batch["hand_joints_obs"].shape), (1, 17))
            self.assertEqual(tuple(batch["camera1"].shape), (1, 3, 32, 48))
            self.assertEqual(tuple(batch["camera2"].shape), (1, 3, 24, 40))
            self.assertEqual(tuple(batch["camera1_depth"].shape), (1, 1, 20, 32))

    def test_model_generate_io_smoke(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self._write_root_schema_episode(tmp_path)
            datasets = self._build_datasets_from_cfg(tmp_path)
            batch, mask_batch = datasets.train_data[0]

            batch = {k: v.unsqueeze(0) for k, v in batch.items()}
            mask_batch = {k: v.unsqueeze(0) for k, v in mask_batch.items()}

            algo = ACT(
                algo_cfg=dict(
                    device="cpu",
                    target_dims={"poseR": 7, "hand_joints": 17},
                    z_dim=8,
                    T_target=20,
                    T_z=1,
                    encoder_is_causal=False,
                    decoder_is_causal=True,
                    encoder_group_keys=["qpos"],
                    decoder_group_keys=["qpos", "camera1", "camera2", "camera1_depth"],
                    encoder_cfg=dict(
                        d_model=64,
                        nhead=8,
                        num_encoder_layers=2,
                        dim_feedforward=128,
                        dropout=0.0,
                        activation="relu",
                    ),
                    decoder_cfg=dict(
                        d_model=64,
                        nhead=8,
                        num_encoder_layers=2,
                        dim_feedforward=128,
                        dropout=0.0,
                        activation="relu",
                    ),
                ),
                obs_encoder_cfg=dict(
                    output_dim=64,
                    obs_groups_cfg=dict(
                        qpos=dict(
                            datakeys=["joint_states", "poseR_obs", "hand_joints_obs"],
                            encoder_cfg=dict(type="lowdim_concat", input_dim_total=29),
                            posemb_cfg=dict(type="none"),
                        ),
                        camera1=dict(
                            datakeys=["camera1"],
                            encoder_cfg=dict(
                                type="crop_resnet18",
                                resize_shape=[32, 32],
                                crop_shape=None,
                                pretrained=False,
                            ),
                            posemb_cfg=dict(type="none"),
                        ),
                        camera2=dict(
                            datakeys=["camera2"],
                            encoder_cfg=dict(
                                type="crop_resnet18",
                                resize_shape=[32, 32],
                                crop_shape=None,
                                pretrained=False,
                            ),
                            posemb_cfg=dict(type="none"),
                        ),
                        camera1_depth=dict(
                            datakeys=["camera1_depth"],
                            encoder_cfg=dict(
                                type="crop_resnet18",
                                resize_shape=[32, 32],
                                crop_shape=None,
                                input_channel=1,
                                pretrained=False,
                            ),
                            posemb_cfg=dict(type="none"),
                        ),
                    ),
                    group_emb_cfg=dict(type="whole_seq_sine"),
                ),
            )

            for key in [
                "poseR",
                "hand_joints",
                "joint_states",
                "poseR_obs",
                "hand_joints_obs",
                "camera1",
                "camera2",
                "camera1_depth",
            ]:
                algo._normalizers[key] = LinearNormalizer(torch.tensor(0.0), torch.tensor(1.0))
            algo.set_eval()

            with torch.no_grad():
                outputs = algo.generate(batch, mask_batch)

            self.assertEqual(set(outputs.keys()), {"poseR", "hand_joints"})
            self.assertEqual(tuple(outputs["poseR"].shape), (1, 20, 7))
            self.assertEqual(tuple(outputs["hand_joints"].shape), (1, 20, 17))


if __name__ == "__main__":
    unittest.main()
