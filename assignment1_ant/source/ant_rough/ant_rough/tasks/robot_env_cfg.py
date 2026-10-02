"""Round 2, axis 2: robot-side changes (sensors, observation history, policy, foot material).

The TA loads our task config from GitHub and only swaps the terrain, so the robot, its sensors and the policy are
ours to design. Everything here avoids terrain-dependent sensors: Isaac Lab's RayCaster only supports one static
mesh, which would break on a terrain built from box primitives. Each variant has

- a TRAIN config: V2's lane curriculum and friction randomization + the robot-side change, and
- a DEPLOY config: Isaac-Ant-v0 (flat plane, original rewards/terminations, no randomization) + the same robot-side
  change. Evaluation (``eval_task.py --eval_terrain``) and the submission task swap only the terrain of a DEPLOY config.
"""

from __future__ import annotations

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import bind_physics_material, clone
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, EventCfg, ObservationsCfg

from .ant_rough_env_cfg import AntRoughCurriculumEnvCfg, AntRoughV2REnvCfg

FEET = ["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]


# ---------------------------------------------------------------- foot contact sensing (V12, V13)


def feet_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Per-foot contact force magnitude (x0.01, clipped at 5) and a binary contact flag (> 1 N): 8 values."""
    sensor = env.scene.sensors[sensor_cfg.name]
    f = torch.linalg.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    return torch.cat([torch.clamp(f * 0.01, max=5.0), (f > 1.0).float()], dim=-1)


@configclass
class ContactHistoryObservationsCfg:
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        feet_contact = ObsTerm(func=feet_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)})

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 3  # (60 + 8) x 3 = 204 inputs, oldest first
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ContactObservationsCfg:
    """Same terms without history, for the recurrent policy (V13) which keeps its own memory."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        feet_contact = ObsTerm(func=feet_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)})

    policy: PolicyCfg = PolicyCfg()


def add_contact_sensing(cfg, observations):
    cfg.scene.robot.spawn.activate_contact_sensors = True
    # contact reporters are not created on bodies cloned in Fabric: clone through USD for these variants
    cfg.scene.clone_in_fabric = False
    cfg.scene.contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_foot", history_length=3)
    cfg.observations = observations


# ---------------------------------------------------------------- foot / body material (V14)


@clone
def spawn_usd_with_material(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """``spawn_from_usd`` + one physics material bound to every collision shape of the asset (before cloning)."""
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    mat_path = f"{prim_path}/bodyMaterial"
    cfg.body_material.func(mat_path, cfg.body_material)
    stage = get_current_stage()
    # the Ant asset is instanceable: bindings on instance proxies are ignored, so de-instance the subtree first
    changed = True
    while changed:
        changed = False
        for p in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
            if p.IsInstance():
                p.SetInstanceable(False)
                changed = True
    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            bind_physics_material(p.GetPath().pathString, mat_path)
            n += 1
    print(f"[ant_rough] bound {cfg.body_material.friction_combine_mode} material (mu {cfg.body_material.static_friction})"
          f" to {n} collision prims under {prim_path}", flush=True)
    return prim


@configclass
class UsdFileWithMaterialCfg(sim_utils.UsdFileCfg):
    func = spawn_usd_with_material
    body_material: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.4, dynamic_friction=0.4, friction_combine_mode="min", restitution_combine_mode="min"
    )
    """Robot skin: friction 0.4 with combine mode "min". PhysX resolves combine modes by priority
    (average < min < multiply < max), so against a ground left at the default "average" the contact friction is
    min(ground, 0.4): the robot never sees more grip than its own material allows."""


def use_body_material(cfg, mu: float = 0.4):
    spawn = cfg.scene.robot.spawn
    fields = {k: getattr(spawn, k) for k in spawn.__dataclass_fields__ if k != "func"}
    fields["body_material"] = sim_utils.RigidBodyMaterialCfg(
        static_friction=mu, dynamic_friction=mu, friction_combine_mode="min", restitution_combine_mode="min"
    )
    cfg.scene.robot.spawn = UsdFileWithMaterialCfg(**fields)


# ---------------------------------------------------------------- configs


@configclass
class AntV12TrainEnvCfg(AntRoughCurriculumEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_contact_sensing(self, ContactHistoryObservationsCfg())


@configclass
class AntV12RTrainEnvCfg(AntRoughV2REnvCfg):
    """V12R: V12's robot side (contacts + history + normalization) on V2R's terrain (per-tile random types)."""

    def __post_init__(self):
        super().__post_init__()
        add_contact_sensing(self, ContactHistoryObservationsCfg())


@configclass
class AntV12DeployEnvCfg(AntEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_contact_sensing(self, ContactHistoryObservationsCfg())


@configclass
class AntV13TrainEnvCfg(AntRoughCurriculumEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_contact_sensing(self, ContactObservationsCfg())


@configclass
class AntV13DeployEnvCfg(AntEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_contact_sensing(self, ContactObservationsCfg())


@configclass
class AntV14TrainEnvCfg(AntRoughCurriculumEnvCfg):
    # no friction randomization: re-writing shape materials through the tensor API may drop the "min" combine mode,
    # so V14 trains with exactly the deployed skin (mu 0.4, "min") -- a single, clean design change on top of V2
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        use_body_material(self)
        # ground left at "average" (like an unmodified terrain) so the robot's "min" decides the contact friction
        self.scene.terrain.physics_material.friction_combine_mode = "average"
        self.scene.terrain.physics_material.restitution_combine_mode = "average"


@configclass
class AntV14DeployEnvCfg(AntEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_body_material(self)


@configclass
class AntV1214RTrainEnvCfg(AntRoughV2REnvCfg):
    """V1214R: V2R terrain + V12 sensing (contacts, history, normalization) + V14 skin (mu 0.4, "min"), no friction
    randomization (the skin sets the contact friction; see AntV14TrainEnvCfg)."""

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        add_contact_sensing(self, ContactHistoryObservationsCfg())
        use_body_material(self)
        self.scene.terrain.physics_material.friction_combine_mode = "average"
        self.scene.terrain.physics_material.restitution_combine_mode = "average"


@configclass
class AntV1214TrainEnvCfg(AntRoughCurriculumEnvCfg):
    """V1214: V2 terrain (block-heavy lanes) + V12 sensing + V14 skin."""

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        add_contact_sensing(self, ContactHistoryObservationsCfg())
        use_body_material(self)
        self.scene.terrain.physics_material.friction_combine_mode = "average"
        self.scene.terrain.physics_material.restitution_combine_mode = "average"


@configclass
class AntV1214DeployEnvCfg(AntEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_contact_sensing(self, ContactHistoryObservationsCfg())
        use_body_material(self)


# ---------------------------------------------------------------- skin friction sweep (mu 0.3 / 0.6) on V1214R


def _v1214r_mu(mu: float):
    @configclass
    class Train(AntV1214RTrainEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            use_body_material(self, mu)

    @configclass
    class Deploy(AntV1214DeployEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            use_body_material(self, mu)

    return Train, Deploy


AntV1214RMu03TrainEnvCfg, AntV1214Mu03DeployEnvCfg = _v1214r_mu(0.3)
AntV1214RMu06TrainEnvCfg, AntV1214Mu06DeployEnvCfg = _v1214r_mu(0.6)
AntV1214RMu02TrainEnvCfg, AntV1214Mu02DeployEnvCfg = _v1214r_mu(0.2)
