from pathlib import Path

import mujoco


ASSET_ROOT = Path(__file__).parents[1] / "humanoid_bench" / "assets"


def test_kitchen_scenes_compile_with_mujoco_33():
    for scene in ("g1_torque_kitchen.xml", "h1hand_pos_kitchen.xml"):
        model = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / "envs" / scene))
        assert model.nq > 0


def test_g1_uses_the_free_base_name_expected_by_tasks():
    for scene in ("g1_torque_truck.xml", "g1_torque_run.xml"):
        model = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / "envs" / scene))
        assert model.joint("free_base").id == 0
