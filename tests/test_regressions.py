from pathlib import Path

import mujoco


ASSET_ROOT = Path(__file__).parents[1] / "humanoid_bench" / "assets"


def test_kitchen_scenes_compile_with_mujoco_33():
    for scene in ("g1_torque_kitchen.xml", "h1hand_pos_kitchen.xml"):
        model = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / "envs" / scene))
        assert model.nq > 0
