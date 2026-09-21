"""Excavator claw digging into MPM sand -- target-mode tracking under granular load.

This is a *standalone Newton* scene, not an Isaac Lab one. Isaac Lab 6.0.1 wraps only
four of Newton's solvers (mjwarp, xpbd, featherstone, kamino); ``SolverImplicitMPM``
is not among them, so sand cannot be spawned inside a ManagerBasedEnv. The excavator
USD is loaded straight into a ``newton.ModelBuilder`` instead.

What it reproduces from ``sim.py --integration target``: a commanded joint velocity is
integrated into a *position target*, and an articulation PD drives the joint toward
that target. The gains and joint names come from ``sim_common`` so the two stay in
sync. The measurement of interest is the lag between target and actual once the claw
is loaded by sand, written to a CSV.

Two ways the sand and the robot exchange force, selected with ``--coupling``:

  * ``manual`` (default) -- two models, two solvers, and the impulse exchange written
    out here by hand, after newton's ``example_mpm_twoway_coupling.py``. This is the
    only path that runs on the newton Isaac Lab pins (``newton[sim]==1.2.1``), so it
    works in ``env_isaaclab``.
  * ``proxy`` -- one model, one ``SolverCoupledProxy``. Newton grew a rigid<->MPM
    coupled solver after 1.2.1 which does the same exchange inside the framework:
    rigid bodies are exposed to MPM as proxy colliders in a shared model and the grid
    impulses are harvested back into body wrenches. It needs **newton >= 1.6**, which
    cannot be installed into ``env_isaaclab`` (it wants warp >= 1.17 and mujoco 3.11
    against Isaac Sim's warp 1.13 and mujoco 3.8), so run it from the newton
    checkout's own venv.

Everything else -- scene, gains, dig cycle, CSV -- is shared, so the two paths are
directly comparable and the comparison is the point of keeping both.

Differences from sim.py worth knowing:
  * Robot gravity is ON here. sim.py sets DISABLE_ROBOT_GRAVITY because the learned
    actuator already contains the machine's gravity response; there is no NN in this
    scene, so gravity is left in.
  * The rigid solver is Newton's SolverMuJoCo, not PhysX. Absolute gains will not map
    1:1 to sim.py -- compare no-sand vs sand runs, not this script vs PhysX.

Scene layout (metres, Z-up, excavator base spawned at z=0.09):
  * excavator origin at (0, 0)
  * open-topped container, near inner wall face at x=0.30, 0.20 m deep
  * sand emitted above the floor and dropped in

Usage:
    python scripts/masi/sand_excavator.py                    # bake to USD
    python scripts/masi/sand_excavator.py --viewer gl        # live window (needs pyglet)
    python scripts/masi/sand_excavator.py --no_sand          # baseline, same motion, no sand

    # the coupled-solver path, from the newton checkout's venv
    ../../newton/.venv/Scripts/python.exe scripts/masi/sand_excavator.py --coupling proxy
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM, SolverMuJoCo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_common import (  # noqa: E402
    ARM_DAMPING,
    ARM_JOINT_NAMES,
    ARM_STIFFNESS,
    CARRIAGE_DAMPING,
    CARRIAGE_STIFFNESS,
    SIM_HZ,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD = os.path.join(HERE, "assets", "excavator.usd")

# Same start pose as sim.py's ExcavatorSceneCfg.
DEFAULT_JOINT_POS = {
    "revolute_carriage": 0.0,
    "revolute_lift": -0.5498,
    "revolute_tilt": 1.2549,
    "revolute_tool": -0.7540,
}
ROBOT_BASE_Z = 0.09

# Midpoint of the bucket's cutting edge in the ``bucket`` body frame [m], measured off
# assets/excavator.usd: the edge is a straight line 0.110 m wide at y=+0.093, z=-0.087.
# The tool used to be a gripper whose tip was quoted as a fixed drop below a body
# origin; that only held at one pose. Transforming this point by the bucket's pose is
# pose-exact, so the logged tip is the real cutting edge at every phase of the dig.
BUCKET_TIP_LOCAL = np.array([0.0, 0.0926, -0.0869])

# Scripted dig cycle: (label, duration_s, {joint_name: velocity_cmd rad/s}).
#
# Calibration, measured from a --no_sand run of this scene:
#   * +revolute_lift lowers the claw at 0.637 m of claw height per rad
#   * the logged claw_z is the lowest gripper *body origin*; the claw geometry
#     extends 0.080 m below that origin, so the tip is at claw_z - 0.080
#   * start pose puts claw_z at 0.157, i.e. the tip at ~0.077
#   * the default 0.06 m sand bed settles to a surface at z ~0.067
# The cycle is tuned so the tip stops ~3 cm clear of the floor in free air. With
# sand present it stops higher still -- that shortfall is the measurement.
DIG_PHASES = [
    ("settle", 1.50, {}),
    ("approach", 0.40, {"revolute_lift": 0.16}),
    ("penetrate", 0.60, {"revolute_lift": 0.03, "revolute_tilt": 0.04}),
    ("curl", 0.80, {"revolute_tool": -0.35}),
    ("drag", 0.80, {"revolute_tilt": -0.15, "revolute_lift": -0.03}),
    ("lift_out", 1.20, {"revolute_lift": -0.30}),
    ("hold", 0.60, {}),
]


# --- two-way coupling kernels ------------------------------------------------
# Adapted from newton/examples/mpm/example_mpm_twoway_coupling.py (Apache-2.0).


@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array[int],
    collider_impulses: wp.array[wp.vec3],
    collider_impulse_pos: wp.array[wp.vec3],
    body_ids: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
):
    """Sum per-grid-node sand impulses into a force/torque at each body's COM."""
    i = wp.tid()
    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index == -1:
            return
        f_world = collider_impulses[i] / dt
        X_wb = body_q[body_index]
        X_com = body_com[body_index]
        r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)
        wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))


@wp.kernel
def subtract_body_force(
    dt: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_inv_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[float],
    body_q_res: wp.array[wp.transform],
    body_qd_res: wp.array[wp.spatial_vector],
):
    """Remove last step's sand force from body velocity before the MPM solve."""
    body_id = wp.tid()
    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q[body_id])
    delta_w = dt * wp.quat_rotate(r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))
    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)


class SandExcavator:
    def __init__(self, viewer, args):
        self.args = args
        self.viewer = viewer
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.use_sand = not args.no_sand
        # With no sand there is nothing to couple, so the cheaper path is the only one.
        self.coupling = args.coupling if self.use_sand else "manual"

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.6
        if self.coupling == "proxy":
            # The coupled solver slices ONE model into per-solver views, so the sand
            # particles have to be built into the same builder as the robot -- and the
            # MPM per-particle attributes have to be registered before anything is added.
            SolverImplicitMPM.register_custom_attributes(builder)

        rigid_body_start = builder.body_count
        self._add_robot(builder)
        self.rigid_body_ids = list(range(rigid_body_start, builder.body_count))
        self._add_container(builder)
        self._add_ground(builder)

        n_particles = 0
        if self.coupling == "proxy":
            n_particles = self._emit_sand(builder)
            self.model = builder.finalize()
            self.sand_model = self.model  # one model holds robot and sand
        else:
            self.model = builder.finalize()
            self.sand_model = None
            if self.use_sand:
                sand_builder = newton.ModelBuilder()
                SolverImplicitMPM.register_custom_attributes(sand_builder)
                n_particles = self._emit_sand(sand_builder)
                self.sand_model = sand_builder.finalize()

        if self.use_sand:
            self._apply_sand_material(self.sand_model)
            self._apply_collider_margin()
            print(
                f"[masi] sand particles: {n_particles}  density={args.density} kg/m^3  "
                f"mu={args.friction}  coupling={self.coupling}"
            )
        else:
            print("[masi] sand disabled (--no_sand baseline)")

        # --- solvers ---
        if self.coupling == "proxy":
            self._setup_proxy_solver()
        else:
            self._setup_manual_solvers()

        # --- target-mode controller state ---
        self.control_target = self._control_target_array()
        self.q_target = np.array(self.model.joint_q.numpy(), dtype=np.float64)
        self.limit_lo = np.array(self.model.joint_limit_lower.numpy(), dtype=np.float64)
        self.limit_hi = np.array(self.model.joint_limit_upper.numpy(), dtype=np.float64)
        self._target_host = wp.zeros(self.model.joint_dof_count, dtype=wp.float32, device="cpu")

        # bodies whose sand load we report: everything at or below the tool joint.
        # The bucket is the only one the sand actually touches -- tool_body is the small
        # link above it -- so dropping "bucket" here silently zeroes sand_force_N.
        self.tool_bodies = [
            i
            for i, label in enumerate(self.model.body_label)
            if any(k in label for k in ("tool_body", "bucket"))
        ]
        self.bucket_body = next(
            (i for i, label in enumerate(self.model.body_label) if "bucket" in label), None
        )
        self._warned_no_coupling_forces = False

        self.log_rows = []
        self.viewer.set_model(self.model)
        if self.use_sand:
            self.viewer.show_particles = True

    def tool_tip(self) -> np.ndarray:
        """World position of the bucket's cutting-edge midpoint [m].

        Falls back to the lowest tool body origin when the loaded asset has no bucket,
        which is what the older gripper USD needs.
        """
        body_q = self.state_0.body_q.numpy()
        if self.bucket_body is None:
            lowest = min(self.tool_bodies, key=lambda b: body_q[b][2])
            return np.array(body_q[lowest][:3], dtype=np.float64)
        pose = body_q[self.bucket_body]
        pos = np.array(pose[:3], dtype=np.float64)
        u, w = np.array(pose[3:6], dtype=np.float64), float(pose[6])
        v = BUCKET_TIP_LOCAL
        return pos + 2.0 * u.dot(v) * u + (w * w - u.dot(u)) * v + 2.0 * w * np.cross(u, v)

    def _control_target_array(self):
        """The Control array holding joint position targets.

        Newton 1.5 renamed ``Control.joint_target_pos`` to ``joint_target_q`` and left
        the old name as a descriptor that raises, so this has to be resolved by probing
        rather than by version number -- the script runs against both the 1.2.1 Isaac
        Lab pins and the >= 1.6 the coupled solver needs.
        """
        for name in ("joint_target_q", "joint_target_pos"):
            if hasattr(self.control, name):
                return getattr(self.control, name)
        raise AttributeError(
            f"newton {newton.__version__}: Control exposes neither joint_target_q nor joint_target_pos"
        )

    def _add_robot(self, builder) -> None:
        """Load the excavator USD, set its start pose and its PD gains."""
        args = self.args
        builder.add_usd(
            ROBOT_USD,
            xform=wp.transform(wp.vec3(0.0, 0.0, ROBOT_BASE_Z), wp.quat_identity()),
            floating=False,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            # By default 6 of the 55 collision shapes import as CONVEX_MESH, which the
            # MPM collider mesher rejects (NotImplementedError: Shape type 10). Skipping
            # the approximation keeps every shape a plain MESH, which it does support.
            skip_mesh_approximation=True,
        )
        self.joint_names = [label.split("/")[-1] for label in builder.joint_label]
        for name, value in DEFAULT_JOINT_POS.items():
            builder.joint_q[self.joint_names.index(name)] = value

        # PD gains, matching sim_common's ImplicitActuator settings.
        for i, name in enumerate(self.joint_names):
            if name == "revolute_carriage":
                builder.joint_target_ke[i] = args.carriage_ke
                builder.joint_target_kd[i] = args.carriage_kd
            else:
                builder.joint_target_ke[i] = args.arm_ke
                builder.joint_target_kd[i] = args.arm_kd

    def _apply_sand_material(self, model) -> None:
        """Push the CLI sand parameters onto whichever model owns the particles."""
        args = self.args
        for key in ("friction", "yield_pressure", "young_modulus", "poisson_ratio", "hardening", "viscosity"):
            if hasattr(model.mpm, key) and hasattr(args, key):
                getattr(model.mpm, key).fill_(getattr(args, key))

    def _apply_collider_margin(self) -> None:
        """Inflate every collision surface so the MPM grid can actually see it.

        MPM resolves contact at grid nodes, so a collider thinner than a voxel is
        invisible between them and particles stream straight through. The bucket is the
        worst case in this scene: its shell is 1.9 mm of wall against a 15 mm voxel,
        i.e. 0.13 of a cell. Refining the grid cannot close that gap -- the wall would
        need ``voxel_size <= 0.0019``, which is 8x finer in each axis and ~500x the
        cells -- so the margin is the knob that fixes it, and it costs nothing.

        ``SolverImplicitMPM`` takes the collider's signed-distance offset from the
        model's per-shape margin, and ``add_usd`` leaves that at exactly 0.0, which is
        why the leak is total rather than occasional. Half a voxel here puts the
        effective wall at ``1.9 mm + 2 * margin``, just over one cell.

        This inflates the container walls (10 mm, also sub-voxel) and the arm by the
        same amount, which is what you want -- the sand sees a slightly fatter machine
        and stays where it belongs.
        """
        margin = self.args.collider_margin
        if margin < 0.0:
            margin = 0.5 * self.args.voxel_size
        self.model.shape_margin.fill_(margin)
        print(f"[masi] collider margin: {margin * 1000:.1f} mm ({margin / self.args.voxel_size:.2f} voxel)")

    def _mpm_config(self):
        """MPM solver settings, shared by both coupling paths so sand behaves the same."""
        args = self.args
        cfg = SolverImplicitMPM.Config()
        cfg.voxel_size = args.voxel_size
        cfg.tolerance = args.tolerance
        cfg.max_iterations = args.max_iterations
        cfg.transfer_scheme = "pic"
        cfg.strain_basis = "P0"
        cfg.grid_type = "sparse"
        cfg.critical_fraction = 0.0
        cfg.air_drag = args.air_drag
        cfg.collider_velocity_mode = args.collider_velocity_mode
        return cfg

    # --- solver setup ---------------------------------------------------------

    def _setup_manual_solvers(self) -> None:
        """Two models, two solvers, and the impulse exchange written out by hand.

        This is the path that works on the newton Isaac Lab pins (1.2.1), which has no
        coupled-solver framework at all. Sand impulses are harvested off the MPM grid,
        summed into body wrenches, and subtracted back off the rigid velocities before
        the next MPM solve -- the recipe from newton's
        ``example_mpm_twoway_coupling.py``.
        """
        args = self.args
        # The claw meshes against the container walls and ground generate ~190 contacts;
        # MJWarp silently drops any beyond nconmax, so keep plenty of headroom.
        self.solver = SolverMuJoCo(
            self.model, use_mujoco_contacts=False, njmax=args.njmax, nconmax=args.nconmax
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        if not self.use_sand:
            return

        self.mpm_solver = SolverImplicitMPM(self.sand_model, self._mpm_config())
        # colliders come from the rigid model, not the sand model
        self.mpm_solver.setup_collider(model=self.model)

        self.sand_state = self.sand_model.state()
        self.sand_state.body_q = wp.empty_like(self.state_0.body_q)
        self.sand_state.body_qd = wp.empty_like(self.state_0.body_qd)
        self.sand_state.body_f = wp.empty_like(self.state_0.body_f)

        max_nodes = 1 << 20
        dev = self.model.device
        self.collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=dev)
        self.collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=dev)
        self.collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=dev)
        self._collect_collider_impulses()
        self.collider_body_id = self.mpm_solver.collider_body_index
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)
        self.particle_colors = wp.full(
            self.sand_model.particle_count, value=wp.vec3(0.76, 0.66, 0.45), dtype=wp.vec3, device=dev
        )

    def _setup_proxy_solver(self) -> None:
        """One model, one ``SolverCoupledProxy``, coupling done by the framework.

        Newton grew a rigid<->MPM coupled solver after 1.2.1: the rigid bodies are
        exposed to MPM as proxy colliders inside a shared model and the grid impulses
        are harvested back into body wrenches -- which is what the manual path does by
        hand. Same scene, same dig cycle, same CSV, so the two are directly comparable,
        and that comparison is the reason both are kept.
        """
        try:
            from newton.solvers.experimental.coupled import SolverCoupledProxy
        except ImportError as exc:
            raise RuntimeError(
                f"--coupling proxy needs newton >= 1.6 (found {newton.__version__}); the coupled "
                "solvers do not exist in the 1.2.1 that isaaclab_newton pins. Run this script with "
                "the newton checkout's own interpreter instead:\n"
                "    <newton>/.venv/Scripts/python.exe scripts/masi/sand_excavator.py --coupling proxy\n"
                f"(import failed with: {exc})"
            ) from exc

        args = self.args
        mujoco_kwargs = {"use_mujoco_contacts": False, "njmax": args.njmax, "nconmax": args.nconmax}
        mpm_config = self._mpm_config()

        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mjc",
                    solver=lambda view: SolverMuJoCo(model=view, **mujoco_kwargs),
                    bodies=self.rigid_body_ids,
                    joints=list(range(self.model.joint_count)),
                    # The rigid solver substeps inside one coupled step, so the robot
                    # still runs at frame_dt / substeps as it does on the manual path.
                    substeps=self.sim_substeps,
                ),
                SolverCoupledProxy.Entry(
                    name="mpm",
                    solver=lambda view: SolverImplicitMPM(model=view, config=mpm_config),
                    particles=list(range(self.model.particle_count)),
                    in_place=True,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mjc",
                        destination="mpm",
                        bodies=self.rigid_body_ids,
                        mass_scale=args.mass_scale,
                        mode="lagged",
                        # MPM does its own collider contact, so the proxy view must not
                        # run a second collision pipeline over the same shapes.
                        collision_pipeline=lambda _view: None,
                    )
                ],
                iterations=args.proxy_iterations,
            ),
        )
        self.mpm_solver = self.solver.solver("mpm")

        # One state: the coupled solver steps it in place.
        self.state_0 = self.model.state()
        self.state_1 = self.state_0
        self.control = self.model.control()
        self.rigid_collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=0)
        self.contacts = self.rigid_collision_pipeline.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.particle_colors = wp.full(
            self.model.particle_count,
            value=wp.vec3(0.76, 0.66, 0.45),
            dtype=wp.vec3,
            device=self.model.device,
        )

    # --- scene construction ---------------------------------------------------

    def _add_container(self, builder):
        a = self.args
        t = a.wall_thickness
        x0, x1 = a.near_x, a.near_x + a.length
        hy_out = a.width / 2.0 + t
        cfg = newton.ModelBuilder.ShapeConfig(mu=a.wall_friction, density=0.0)
        # Walls stand on the ground plane; the ground itself is the container floor.
        walls = [
            ((x0 - t / 2.0, 0.0, a.depth / 2.0), (t / 2.0, hy_out, a.depth / 2.0)),
            ((x1 + t / 2.0, 0.0, a.depth / 2.0), (t / 2.0, hy_out, a.depth / 2.0)),
            (((x0 + x1) / 2.0, a.width / 2.0 + t / 2.0, a.depth / 2.0), (a.length / 2.0, t / 2.0, a.depth / 2.0)),
            (((x0 + x1) / 2.0, -a.width / 2.0 - t / 2.0, a.depth / 2.0), (a.length / 2.0, t / 2.0, a.depth / 2.0)),
        ]
        for center, half in walls:
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                hx=half[0],
                hy=half[1],
                hz=half[2],
                cfg=cfg,
            )
        print(f"[masi] container x=[{x0:.3f},{x1:.3f}] y=+-{a.width / 2:.3f} depth={a.depth:.3f} m")

    def _add_ground(self, builder):
        """Add the ground plane the container and the sand stand on.

        A hook rather than an inline call so a subclass can change how the plane is
        added -- ``sand_teleop.py`` hides it -- without restating :meth:`build_scene`.
        """
        builder.add_ground_plane()

    def _emit_sand(self, builder) -> int:
        """Fill the container with a block of particles, dropped from ``--sand_drop``."""
        a = self.args
        lo = np.array([a.near_x, -a.width / 2.0, a.sand_drop])
        hi = np.array([a.near_x + a.length, a.width / 2.0, a.sand_drop + a.sand_fill])
        # shrink slightly so particles do not start inside the walls
        lo[:2] += a.voxel_size * 0.5
        hi[:2] -= a.voxel_size * 0.5

        res = np.array(np.ceil(a.particles_per_cell * (hi - lo) / a.voxel_size), dtype=int)
        cell = (hi - lo) / res
        radius = float(np.max(cell) * 0.5)
        mass = float(np.prod(cell) * a.density)

        builder.add_particle_grid(
            pos=wp.vec3(*lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(res[0]) + 1,
            dim_y=int(res[1]) + 1,
            dim_z=int(res[2]) + 1,
            cell_x=float(cell[0]),
            cell_y=float(cell[1]),
            cell_z=float(cell[2]),
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
        )
        return int(np.prod(res + 1))

    # --- control --------------------------------------------------------------

    def _phase_at(self, t: float):
        acc = 0.0
        for label, dur, cmd in DIG_PHASES:
            if t < acc + dur:
                return label, cmd
            acc += dur
        return "done", {}

    def apply_control(self):
        """Target mode: integrate the velocity command into a position target."""
        label, cmd = self._phase_at(self.sim_time)
        for name, vel in cmd.items():
            i = self.joint_names.index(name)
            self.q_target[i] = np.clip(
                self.q_target[i] + vel * self.frame_dt, self.limit_lo[i], self.limit_hi[i]
            )
        self._target_host.assign(self.q_target.astype(np.float32))
        wp.copy(self.control_target, self._target_host.to(self.model.device))
        return label

    # --- simulation -----------------------------------------------------------

    def _collect_collider_impulses(self):
        imp, pos, ids = self.mpm_solver.collect_collider_impulses(self.sand_state)
        self.collider_impulse_ids.fill_(-1)
        n = min(imp.shape[0], self.collider_impulses.shape[0])
        self.collider_impulses[:n].assign(imp[:n])
        self.collider_impulse_pos[:n].assign(pos[:n])
        self.collider_impulse_ids[:n].assign(ids[:n])

    def simulate_robot(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.use_sand:
                wp.launch(
                    compute_body_forces,
                    dim=self.collider_impulse_ids.shape[0],
                    inputs=[
                        self.frame_dt,
                        self.collider_impulse_ids,
                        self.collider_impulses,
                        self.collider_impulse_pos,
                        self.collider_body_id,
                        self.state_0.body_q,
                        self.model.body_com,
                        self.state_0.body_f,
                    ],
                )
                self.body_sand_forces.assign(self.state_0.body_f)
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate_sand(self):
        wp.launch(
            subtract_body_force,
            dim=self.sand_state.body_q.shape,
            inputs=[
                self.frame_dt,
                self.state_0.body_q,
                self.state_0.body_qd,
                self.body_sand_forces,
                self.model.body_inv_inertia,
                self.model.body_inv_mass,
                self.sand_state.body_q,
                self.sand_state.body_qd,
            ],
        )
        self.mpm_solver.step(self.sand_state, self.sand_state, contacts=None, control=None, dt=self.frame_dt)
        self._collect_collider_impulses()

    def simulate_coupled(self):
        """One coupled step; the solver runs both sub-solvers and exchanges the impulses.

        Note the single state in and out: the MPM entry is declared ``in_place``, and the
        rigid entry substeps internally, so what the manual path spells out over two
        models and four kernel launches is this one call.
        """
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.rigid_collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.frame_dt)

    def step(self):
        phase = self.apply_control()
        if self.coupling == "proxy":
            self.simulate_coupled()
        else:
            self.simulate_robot()
            if self.use_sand:
                self.simulate_sand()
        self._log(phase)
        self.sim_time += self.frame_dt

    def _log(self, phase: str):
        q = self.state_0.joint_q.numpy()
        row = {"t": round(self.sim_time, 4), "phase": phase}
        for name in ARM_JOINT_NAMES:
            i = self.joint_names.index(name)
            row[f"{name}_target"] = float(self.q_target[i])
            row[f"{name}_actual"] = float(q[i])
            row[f"{name}_err"] = float(self.q_target[i] - q[i])
        row["tip_z"] = float(self.tool_tip()[2])
        row["sand_force_N"] = self._sand_force_n()
        self.log_rows.append(row)

    def _sand_force_n(self) -> float:
        """Magnitude of the sand wrench on the tool bodies [N], for either path.

        Manual: the array this script fills itself. Proxy: the feedback the coupled
        solver harvested, which it keeps per *global* body id -- but on a private field,
        so a newton that renames it degrades to reporting 0 with one warning rather than
        taking the run down. Both are read as ``body_f`` layout, linear part first.
        """
        if not self.use_sand:
            return 0.0
        if self.coupling == "manual":
            forces = self.body_sand_forces.numpy()
        else:
            forces = self._proxy_coupling_forces()
            if forces is None:
                return 0.0
        return float(sum(np.linalg.norm(forces[b][:3]) for b in self.tool_bodies))

    def _proxy_coupling_forces(self):
        """Per-body coupling wrench out of ``SolverCoupledProxy``, or None if unavailable."""
        mappings = getattr(self.solver, "_proxy_mappings", None)
        if mappings:
            entries = mappings.values() if isinstance(mappings, dict) else mappings
            for mapping in entries:
                forces = getattr(mapping, "coupling_forces", None)
                if forces is not None:
                    return forces.numpy()
        if not self._warned_no_coupling_forces:
            print("[masi] [WARN] coupled solver exposes no coupling_forces; sand_force_N reads 0")
            self._warned_no_coupling_forces = True
        return None

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.use_sand:
            # Manual keeps the particles in their own state; coupled steps them in the
            # one shared state.
            particle_state = self.state_0 if self.coupling == "proxy" else self.sand_state
            self.viewer.log_points(
                "/sand",
                points=particle_state.particle_q,
                radii=self.sand_model.particle_radius,
                colors=self.particle_colors,
                hidden=not getattr(self.viewer, "show_particles", True),
            )
        self.viewer.end_frame()

    def write_log(self, path: str):
        if not self.log_rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.log_rows)
        print(f"[masi] tracking log: {path}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--viewer", default="usd", choices=["usd", "gl", "rerun", "null"])
    p.add_argument("--output_path", default=os.path.join(HERE, "sand_excavator.usd"))
    p.add_argument("--log_path", default=os.path.join(HERE, "sand_excavator_tracking.csv"))
    p.add_argument("--num_frames", type=int, default=0, help="0 = run the full dig cycle")
    p.add_argument("--fps", type=float, default=SIM_HZ)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--no_sand", action="store_true", help="baseline run with the same motion, no sand")
    p.add_argument(
        "--coupling",
        default="manual",
        choices=["manual", "proxy"],
        help="how sand and robot exchange forces. 'manual' is the hand-rolled impulse "
        "exchange and runs on newton 1.2.1; 'proxy' uses SolverCoupledProxy and needs "
        "newton >= 1.6, so run it from the newton checkout's own venv",
    )

    g = p.add_argument_group("container")
    g.add_argument("--near_x", type=float, default=0.30, help="inner face of the near wall [m]")
    g.add_argument("--length", type=float, default=0.40)
    g.add_argument("--width", type=float, default=0.40)
    g.add_argument("--depth", type=float, default=0.20)
    g.add_argument("--wall_thickness", type=float, default=0.010)
    g.add_argument("--wall_friction", type=float, default=0.4)

    g = p.add_argument_group("sand")
    g.add_argument("--density", type=float, default=1200.0, help="light/loose sand [kg/m^3]")
    g.add_argument("--friction", type=float, default=0.55)
    # The start pose puts the claw tip at z~0.081. A 0.06 m bed settles to a surface
    # just below that, so the claw begins in clear air and the dig actually enters sand.
    g.add_argument("--sand_fill", type=float, default=0.06, help="height of the emitted block [m]")
    g.add_argument("--sand_drop", type=float, default=0.015, help="height the block starts at [m]")
    g.add_argument("--voxel_size", type=float, default=0.015)
    g.add_argument(
        "--collider_margin", type=float, default=-1.0,
        help="Collider signed-distance offset [m]; <0 means half a voxel. Without this "
             "the 1.9 mm bucket shell is far thinner than a voxel and sand pours through it",
    )
    g.add_argument("--particles_per_cell", type=float, default=2.0)
    g.add_argument("--air_drag", type=float, default=1.0)
    g.add_argument("--max_iterations", type=int, default=50)
    g.add_argument("--tolerance", type=float, default=1.0e-4)
    # "backward" suits a kinematic (one-way) collider; the two-way path wants "forward".
    g.add_argument("--collider_velocity_mode", default="forward", choices=["forward", "backward"])
    g.add_argument("--young_modulus", type=float, default=1.0e15)
    g.add_argument("--poisson_ratio", type=float, default=0.3)
    g.add_argument("--yield_pressure", type=float, default=1.0e12)
    g.add_argument("--hardening", type=float, default=0.0)
    g.add_argument("--viscosity", type=float, default=0.0)

    g = p.add_argument_group("actuation")
    g.add_argument("--arm_ke", type=float, default=ARM_STIFFNESS)
    g.add_argument("--arm_kd", type=float, default=ARM_DAMPING)
    g.add_argument("--carriage_ke", type=float, default=CARRIAGE_STIFFNESS)
    g.add_argument("--carriage_kd", type=float, default=CARRIAGE_DAMPING)
    g.add_argument("--nconmax", type=int, default=512, help="MJWarp contact buffer size")
    g.add_argument("--njmax", type=int, default=512, help="MJWarp constraint buffer size")

    g = p.add_argument_group("proxy coupling (--coupling proxy only)")
    g.add_argument(
        "--proxy_iterations", type=int, default=1, help="proxy relaxation passes per coupled step"
    )
    g.add_argument(
        "--mass_scale",
        type=float,
        default=1.0,
        help="scale on the effective body mass MPM sees through the proxy. Raising it makes "
        "the arm harder for sand to push around, without touching the modelled mass",
    )
    return p


def main():
    args = build_parser().parse_args()
    total_frames = args.num_frames or int(sum(d for _l, d, _c in DIG_PHASES) * args.fps)

    if args.viewer == "usd":
        viewer = newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=total_frames)
    elif args.viewer == "gl":
        viewer = newton.viewer.ViewerGL()
    elif args.viewer == "rerun":
        viewer = newton.viewer.ViewerRerun()
    else:
        viewer = newton.viewer.ViewerNull(num_frames=total_frames)

    example = SandExcavator(viewer, args)
    print(f"[masi] running {total_frames} frames at {args.fps} Hz ({total_frames / args.fps:.1f} s)")

    frame = 0
    while viewer.is_running() and frame < total_frames:
        example.step()
        example.render()
        frame += 1

    viewer.close()
    example.write_log(args.log_path)


if __name__ == "__main__":
    main()
