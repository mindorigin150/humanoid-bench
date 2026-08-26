from pathlib import Path

import mujoco
import numpy as np
from types import SimpleNamespace

import humanoid_bench.env as env_module
from humanoid_bench.envs.highbar import HighBarSimple
from humanoid_bench.envs.package import Package
from humanoid_bench.envs.room import Room
from humanoid_bench.envs.spoon import Spoon
from humanoid_bench.envs.truck import Truck


ASSET_ROOT = Path(__file__).parents[1] / "humanoid_bench" / "assets"


def test_kitchen_scenes_compile_with_mujoco_33():
    for scene in ("g1_torque_kitchen.xml", "h1hand_pos_kitchen.xml"):
        model = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / "envs" / scene))
        assert model.nq > 0


def test_g1_uses_the_free_base_name_expected_by_tasks():
    for scene in ("g1_torque_truck.xml", "g1_torque_run.xml"):
        model = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / "envs" / scene))
        assert model.joint("free_base").id == 0


def test_room_reset_leaves_the_g1_hand_qpos_untouched():
    env = SimpleNamespace(
        data=SimpleNamespace(qpos=np.zeros(86), qvel=np.zeros(85))
    )
    env.set_state = lambda qpos, qvel: (
        setattr(env.data, "qpos", qpos), setattr(env.data, "qvel", qvel)
    )
    task = Room()
    task._env = env
    task.reset_model()
    assert (env.data.qpos[37:44] == 0).all()


def test_truck_reset_rebuilds_episode_bookkeeping():
    package_list = ["package_a", "package_b"]
    env = SimpleNamespace(
        data=SimpleNamespace(qpos=np.zeros(2), qvel=np.zeros(2)),
        named=SimpleNamespace(
            data=SimpleNamespace(
                xpos={package: np.array([0, 0, index + 1]) for index, package in enumerate(package_list)}
            )
        ),
    )
    task = Truck()
    task._env = env
    task.package_list = package_list
    task.packages_on_truck = []
    task.packages_picked_up = package_list[:1]
    task.packages_on_table = package_list[1:]
    task.initial_zs = {}
    task.reset_model()
    assert task.packages_on_truck == package_list
    assert task.packages_picked_up == []
    assert task.packages_on_table == []
    assert task.initial_zs == {"package_a": 1, "package_b": 2}


def test_spoon_reset_restarts_target_phase():
    task = Spoon()
    task._env = SimpleNamespace(
        data=SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1))
    )
    task.step_counter = 19
    obs = task.reset_model()
    assert task.step_counter == 0
    assert (obs[-3:] == np.array([0.81, -0.1, 0.95])).all()


def test_highbar_reset_applies_no_qpos_noise_before_grasp_initialization(monkeypatch):
    data = SimpleNamespace(qpos=np.zeros(3), qvel=np.zeros(2))
    uniform_calls = []

    def uniform(low, high, size):
        uniform_calls.append((low, high, size))
        return np.ones(size)

    env = SimpleNamespace(
        model=SimpleNamespace(nq=3),
        data=data,
        keyframe=0,
        randomness=0.01,
        np_random=SimpleNamespace(uniform=uniform),
    )
    task = HighBarSimple()
    task._env = env
    env.task = task
    env.set_state = lambda qpos, qvel: (
        setattr(data, "qpos", qpos), setattr(data, "qvel", qvel)
    )
    monkeypatch.setattr(env_module.mujoco, "mj_resetDataKeyframe", lambda *args: None)
    monkeypatch.setattr(env_module.mujoco, "mj_forward", lambda *args: None)
    env_module.HumanoidEnv.reset_model(env)
    assert uniform_calls[0][:2] == (0, 0)


def test_package_reset_observes_the_new_destination(monkeypatch):
    data = SimpleNamespace(qpos=np.zeros(8), qvel=np.zeros(7))
    model = SimpleNamespace(body_pos=np.zeros((1, 3)))
    named_data = SimpleNamespace(
        site_xpos={"destination_loc": np.zeros(3)},
    )
    env = SimpleNamespace(
        data=data,
        model=model,
        named=SimpleNamespace(data=named_data),
        render_mode=None,
    )
    env.set_state = lambda qpos, qvel: (
        setattr(data, "qpos", qpos), setattr(data, "qvel", qvel)
    )
    robot = SimpleNamespace(
        dof=1,
        left_hand_position=lambda: np.zeros(3),
        right_hand_position=lambda: np.zeros(3),
    )

    def forward(_model, _data):
        named_data.site_xpos["destination_loc"] = model.body_pos[-1].copy()

    monkeypatch.setattr(mujoco, "mj_forward", forward)
    np.random.seed(7)
    observation = Package(robot, env).reset_model()
    np.testing.assert_array_equal(observation[1:4], model.body_pos[-1])
