"""Drive the excavator through Newton MPM soil, and tune the soil while it runs.

This is ``vla/record_episodes.py``'s demo with the fake sand swapped for real granular
material. Same machine, same start pose, same valve-command control path (stick -> u in
[-1,1]^4 -> joint velocity -> integrated position target), but the ground is
``SolverImplicitMPM`` particles instead of a Warp height field, so the bucket actually
pushes soil around and the soil pushes back.

The scene is deliberately two objects: the excavator, and a 15 cm cube of soil on the
ground in front of it. No walls, and the ground plane collides but is never drawn, so
everything not those two things is black. Nothing here is measured or recorded -- it is
a demo you drive and a material you tune while it runs.

It is a *standalone Newton* scene, like ``sand_excavator.py``, which it subclasses for
the scene, the coupling and the gains -- Isaac Lab 6.0.1 wraps four Newton solvers and
MPM is not one of them, so this cannot live in a ManagerBasedEnv. ``sand_excavator.py``
is the other half of the pair: walled box, scripted dig, CSV out. This one plays *you*.

Usage:
    isaaclab.bat -p scripts/masi/sand_teleop.py
    isaaclab.bat -p scripts/masi/sand_teleop.py --soil dry_sand
    isaaclab.bat -p scripts/masi/sand_teleop.py --surface              # one smooth skin
    isaaclab.bat -p scripts/masi/sand_teleop.py --voxel_size 0.008    # finer soil
    isaaclab.bat -p scripts/masi/sand_teleop.py --fps 50 --substeps 8  # ~1.4x throughput
    isaaclab.bat -p scripts/masi/sand_teleop.py --viewer null --autoplay --num_frames 200

Controls -- the gamepad is the intended path, the keyboard is the fallback:

    Xbox pad                       Keyboard
      left  stick X : carriage       J / L   carriage swing
      right stick Y : lift (boom)    I / K   lift
      left  stick Y : tilt (arm)     U / O   tilt
      right stick X : tool (bucket)  N / M   bucket curl / open
      B : reset scene                T       reset scene
      A : scripted dig on/off        P       scripted dig on/off

The viewer owns WASD/QE and the arrow keys (camera), SPACE (pause), ``.`` (single step),
F (frame camera), H (hide UI) and ESC (quit), which is why the machine is not on the
arrow keys that ``vla/teleop.py`` uses.

Tuning the soil
---------------
The point of the demo. Every material parameter is a live slider in the viewer's
"Example Options" panel -- drag it and the next step uses the new value, no restart.
"Print as CLI flags" writes the current settings to the console, so a tune you like can
be pasted back onto the command line.

What each knob does to a bucket cut:

    friction              internal friction. On its own it sets the angle of repose:
                          a pile stands at about atan(mu), a cut wall does not stand.
    yield_stress   [Pa]   cohesion. THIS is the "keeps its shape" knob. Zero is dry sand
                          and a trench that heals itself; a few hundred Pa stands a
                          bucket-deep wall. Scale it against the overburden at the
                          bottom of the cube, rho*g*h = 1200*9.81*0.15 ~ 1.8 kPa: below
                          ~10% of that it does nothing, above ~100% the soil stops
                          behaving like soil and starts behaving like a solid.
    tensile_yield_ratio   how much tension the material takes before it separates. Raise
                          it and the bucket lifts a lump out; leave it at 0 and the load
                          crumbles off the teeth.
    viscosity     [Pa.s]  rate-dependent resistance -- mud, not sand. Slows collapse.
    hardening / dilatancy plastic response to compaction. Snow and packed soil.
    damping         [s]   elastic relaxation time; mostly kills ringing.

``--soil`` presets are starting points, not answers -- pick the closest, then drag.

The fast loop is not the window at all. ``--shot`` runs the scripted dig to a given
step, ray-traces one frame through ``newton.sensors.SensorTiledCamera`` -- which traces
particles, not just meshes -- writes a PNG and exits, no window and no Kit:

    isaaclab.bat -p scripts/masi/sand_teleop.py --shot dry.png  --soil dry_sand  ^
      --voxel_size 0.008 --shot_step 630

At step 630 the bucket has just lifted clear, which is the moment that tells you whether
the cut held. ``--voxel_size 0.008`` (390k particles) costs nothing worth caring about
here because there is no window to draw them into -- the shot is the one place to spend
resolution freely. Change a number, shoot, look, repeat.

That the sensor exists matters beyond the shot: it is a policy camera. Anything the GL
window can be driven to do, this scene can also be recorded from, without Kit.

Speed
-----
It does not run in real time, and the status line prints the measured factor as ``rt=``.
Measured with the bucket down in the soil, on an RTX 2000 Ada laptop:

    rigid solver alone (--no_sand)                       17.2 ms/frame
    + MPM                                                38.2 ms/frame  (0.26x)
    of which grains + project_outside                     2.5 ms/frame

Two things about that are worth knowing before reaching for a knob.

**The rigid solver is half the bill.** MuJoCo over the excavator's 55 mesh colliders
costs as much as the soil does, and it is not negotiable: the arm needs ``sim_dt`` at
or below 2.5 ms. Raising ``--fps`` to shorten it is not needed (100 Hz already gives
2.5 ms with 4 substeps) but *lowering* it without raising ``--substeps`` to match is
what breaks the scene -- at ``sim_dt`` 16 ms the bucket sinks through the ground plane.
Keep ``fps * substeps == 400``. ``--fps 50 --substeps 8`` trades render smoothness for
about 1.4x the throughput at the same ``sim_dt``.

**Simulating soil is nearly free; drawing it is not.** The implicit MPM solve is
dominated by the grid, so 0.015 m voxels (25k particles) and 0.010 m voxels (81k) both
land at ~38 ms/frame -- soil resolution is the cheapest thing on offer. Drawing is where
it goes: the viewer instances a sphere per point, and past ~50k of them the window costs
as much as the solver. Everything below measured on screen, bucket in the soil:

    dx 0.012, grains 0  ->  48k spheres   0.23x     the default
    dx 0.015, grains 2  ->  51k spheres   0.22x     grainier, half the soil resolution
    dx 0.012, grains 3  -> 144k spheres   0.18x     renderer-bound

If the window goes choppy while ``rt=`` holds steady, that is the renderer, not the
physics: lower ``--grains`` before you lower ``--voxel_size``.

**Smooth surface mode** ("Smooth surface" in the side panel, or ``--surface``) trades
the sphere-per-particle draw for one isosurface pulled out of a density grid, so the
soil reads as a material rather than as beads. It costs about 14 ms/frame against 6 ms
for 70k spheres at this budget -- most of that is the viewer repacking a fixed vertex
buffer, so ``--surface_max_tris`` is the knob that moves it. The sliders beside the
checkbox are look-only; none of them reach the solver.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
import types
from dataclasses import replace

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverNotifyFlags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sand_excavator import SandExcavator, build_parser  # noqa: E402

# Valve-command order, shared with vla/teleop.py so muscle memory carries over.
COMMAND_NAMES = ["carriage", "lift", "tilt", "tool"]
COMMAND_JOINTS = ["revolute_carriage", "revolute_lift", "revolute_tilt", "revolute_tool"]

# Body labels that carry the only collision geometry the demo needs. "tool_body" is the
# small link the bucket hangs off; sand_excavator.tool_bodies uses the same two keys.
TOOL_BODY_KEYS = ("tool_body", "bucket")

# Per-particle material parameters that SolverImplicitMPM reads off model.mpm every
# step. sand_excavator._apply_sand_material pushes only a subset; the ones it leaves out
# (yield_stress above all) are exactly the ones that decide whether a cut wall stands,
# so this script pushes the lot.
SOIL_KEYS = (
    "young_modulus",
    "poisson_ratio",
    "damping",
    "friction",
    "yield_pressure",
    "tensile_yield_ratio",
    "yield_stress",
    "hardening",
    "dilatancy",
    "viscosity",
)

# (attribute, label, slider min, slider max). Only the parameters worth dragging live;
# young_modulus and yield_pressure sit at 1e15 / 1e12 (i.e. "off") and stay CLI-only.
SOIL_SLIDERS = (
    ("friction", "friction  mu", 0.0, 1.5),
    ("yield_stress", "cohesion [Pa]", 0.0, 2000.0),
    ("tensile_yield_ratio", "tensile ratio", 0.0, 1.0),
    ("viscosity", "viscosity [Pa.s]", 0.0, 200.0),
    ("hardening", "hardening", 0.0, 20.0),
    ("dilatancy", "dilatancy", 0.0, 2.0),
    ("damping", "damping [s]", 0.0, 0.1),
)

# Look-only, and live: these re-run the isosurface on the next frame and never reach
# the solver. iso is where the skin sits in the density field -- low wraps the outermost
# grains and reads loose, high shrinks onto the packed core and reads wet.
SURFACE_SLIDERS = (
    ("surface_iso", "surface level", 0.2, 6.0),
    ("surface_smooth", "surface smoothing", 0.6, 3.0),
    ("surface_roughness", "surface roughness", 0.0, 1.0),
)

# Starting points. dry_sand, mud and snow are newton's own example values
# (examples/mpm/example_mpm_twoway_coupling.py, example_mpm_multi_material.py);
# damp_sand and loam move towards cohesion on the overburden scale set out above.
SOIL_PRESETS = {
    "dry_sand": dict(
        friction=0.75, yield_stress=0.0, tensile_yield_ratio=0.0,
        viscosity=0.0, hardening=0.0, dilatancy=0.0, damping=0.0,
    ),
    "damp_sand": dict(
        friction=0.85, yield_stress=150.0, tensile_yield_ratio=0.1,
        viscosity=0.0, hardening=0.0, dilatancy=0.3, damping=0.0,
    ),
    "loam": dict(
        friction=0.90, yield_stress=600.0, tensile_yield_ratio=0.3,
        viscosity=5.0, hardening=2.0, dilatancy=0.6, damping=0.01,
    ),
    "mud": dict(
        friction=0.0, yield_stress=300.0, tensile_yield_ratio=1.0,
        viscosity=100.0, hardening=0.0, dilatancy=0.0, damping=0.0,
    ),
    "gravel": dict(
        friction=1.10, yield_stress=0.0, tensile_yield_ratio=0.0,
        viscosity=0.0, hardening=0.0, dilatancy=0.0, damping=0.0,
    ),
    "snow": dict(
        friction=0.10, yield_stress=0.0, tensile_yield_ratio=0.2,
        viscosity=0.0, hardening=10.0, dilatancy=1.0, damping=0.0,
    ),
}

# Scripted dig, in valve commands: (label, seconds, u = [carriage, lift, tilt, tool]).
# Mirrors vla/record_episodes.py's SELFTEST_CYCLE, copied rather than imported because
# importing that module launches Isaac Sim at module scope. Sign convention is
# sand_excavator.py's: +lift lowers the bucket.
DIG_CYCLE = [
    ("settle", 0.6, [0.0, 0.0, 0.0, 0.0]),
    ("approach", 1.0, [0.0, 0.35, 0.0, 0.0]),
    ("penetrate", 1.0, [0.0, 0.10, 0.10, 0.0]),
    ("curl", 1.2, [0.0, 0.0, 0.0, -0.70]),
    ("drag", 1.0, [0.0, -0.06, -0.30, 0.0]),
    ("lift_out", 1.5, [0.0, -0.60, 0.0, 0.0]),
    ("swing", 1.5, [0.60, 0.0, 0.0, 0.0]),
    ("dump", 1.2, [0.0, 0.0, 0.0, 0.70]),
    ("hold", 0.5, [0.0, 0.0, 0.0, 0.0]),
]

# MPM per-particle state, snapshotted at startup so a reset can put the soil back.
MPM_STATE_ATTRS = (
    "particle_qd_grad",
    "particle_elastic_strain",
    "particle_Jp",
    "particle_stress",
    "particle_transform",
)

DEAD_ZONE = 0.30


# --- smooth surface -----------------------------------------------------------


@wp.kernel
def _splat_density(
    particle_q: wp.array(dtype=wp.vec3),
    lo: wp.vec3,
    inv_h: float,
    reach: int,
    sigma_sq: float,
    field: wp.array3d(dtype=float),
):
    """Accumulate a Gaussian blob per particle into the density grid."""
    g = wp.cw_mul(particle_q[wp.tid()] - lo, wp.vec3(inv_h, inv_h, inv_h))
    base_i, base_j, base_k = int(wp.floor(g[0])), int(wp.floor(g[1])), int(wp.floor(g[2]))
    nx, ny, nz = field.shape[0], field.shape[1], field.shape[2]
    for di in range(-reach, reach + 1):
        for dj in range(-reach, reach + 1):
            for dk in range(-reach, reach + 1):
                i, j, k = base_i + di, base_j + dj, base_k + dk
                if i >= 0 and j >= 0 and k >= 0 and i < nx and j < ny and k < nz:
                    d = wp.vec3(float(i), float(j), float(k)) - g
                    wp.atomic_add(field, i, j, k, wp.exp(-wp.dot(d, d) / sigma_sq))


@wp.kernel
def _expand_triangles(
    verts: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    index_count: int,
    field: wp.array3d(dtype=float),
    lo: wp.vec3,
    inv_h: float,
    out_verts: wp.array(dtype=wp.vec3),
    out_normals: wp.array(dtype=wp.vec3),
):
    """Unweld the marching-cubes mesh into the fixed-size buffers the viewer wants.

    Every triangle gets its own three vertices so the index buffer can be a constant
    ``0, 1, 2, ...``; see :class:`SoilSurface` for why that matters. Slots past the end
    of this frame's mesh collapse to a single point, which draws as zero area.
    """
    t = wp.tid()
    if t >= index_count:
        out_verts[t] = wp.vec3(0.0, 0.0, 0.0)
        out_normals[t] = wp.vec3(0.0, 0.0, 1.0)
        return

    p = verts[indices[t]]
    out_verts[t] = p

    # Shade off the density gradient rather than the triangle's own plane: the field is
    # smooth where the facets are not, which is the whole point of the mode.
    g = wp.cw_mul(p - lo, wp.vec3(inv_h, inv_h, inv_h))
    i = wp.clamp(int(g[0]), 1, field.shape[0] - 2)
    j = wp.clamp(int(g[1]), 1, field.shape[1] - 2)
    k = wp.clamp(int(g[2]), 1, field.shape[2] - 2)
    grad = wp.vec3(
        field[i + 1, j, k] - field[i - 1, j, k],
        field[i, j + 1, k] - field[i, j - 1, k],
        field[i, j, k + 1] - field[i, j, k - 1],
    )
    norm = wp.length(grad)
    # Density climbs inwards, so the outward normal is the negated gradient.
    out_normals[t] = wp.vec3(0.0, 0.0, 1.0) if norm < 1.0e-12 else -grad / norm


class SoilSurface:
    """Draws the soil as one smooth skin instead of ~70k spheres.

    Individual particles read as beads, which is wrong for a material that is meant to
    look like earth. This splats them into a density grid, pulls an isosurface out of
    it with ``wp.MarchingCubes``, and hands the viewer a mesh. Smoothing is a property
    of the grid, so ``--surface_smooth`` and ``--surface_iso`` slide the look between
    "wet clay" and "loose heap" without touching the physics -- nothing here feeds back
    into the solver.

    The fixed-size buffers are not an optimisation, they are a requirement.
    ``MeshGL`` sizes its GL buffers from the first ``log_mesh`` call and refuses a later
    change (*"Number of points does not match"*), and it uploads indices exactly once
    (*"no topology changes"*). Marching cubes changes both counts every frame. So the
    mesh is stored unwelded -- three vertices per triangle against a constant index
    buffer -- with a fixed slot count, and unused slots are collapsed to zero area.
    That makes topology genuinely constant while the surface underneath moves freely.
    """

    def __init__(self, args, bounds_lo: np.ndarray, bounds_hi: np.ndarray):
        self.args = args
        h = float(args.surface_voxel)
        self.lo = wp.vec3(*bounds_lo)
        self.inv_h = 1.0 / h
        self.max_slots = 3 * int(args.surface_max_tris)
        self._overflowed = False

        dims = np.maximum(np.ceil((bounds_hi - bounds_lo) / h).astype(int), 4)
        self.field = wp.zeros(tuple(int(d) for d in dims), dtype=float)
        self.mc = wp.MarchingCubes(
            *(int(d) for d in dims),
            self.max_slots,
            int(args.surface_max_tris),
            domain_bounds_lower_corner=wp.vec3(*bounds_lo),
            domain_bounds_upper_corner=wp.vec3(*bounds_hi),
        )
        self.verts = wp.zeros(self.max_slots, dtype=wp.vec3)
        self.normals = wp.zeros(self.max_slots, dtype=wp.vec3)
        self.indices = wp.array(np.arange(self.max_slots, dtype=np.int32), dtype=wp.int32)
        print(
            f"[soil] surface grid {tuple(int(d) for d in dims)} at {h * 1000:.1f} mm "
            f"({int(np.prod(dims)):,} cells), up to {args.surface_max_tris:,} triangles"
        )

    def clear(self) -> None:
        """Collapse every triangle, so the mesh exists but draws nothing."""
        self.verts.zero_()
        self.field.zero_()

    def update(self, particle_q) -> None:
        """Rebuild the isosurface from this frame's particle positions."""
        a = self.args
        self.field.zero_()
        wp.launch(
            _splat_density,
            dim=len(particle_q),
            inputs=[particle_q, self.lo, self.inv_h, int(a.surface_reach),
                    float(a.surface_smooth) ** 2, self.field],
        )
        self.mc.surface(self.field, float(a.surface_iso))

        count = len(self.mc.indices)
        if count > self.max_slots and not self._overflowed:
            print(
                f"[soil] [WARN] surface needs {count // 3:,} triangles but only "
                f"{a.surface_max_tris:,} are allocated; raise --surface_max_tris"
            )
            self._overflowed = True
        wp.launch(
            _expand_triangles,
            dim=self.max_slots,
            inputs=[self.mc.verts, self.mc.indices, min(count, self.max_slots),
                    self.field, self.lo, self.inv_h, self.verts, self.normals],
        )


# --- input --------------------------------------------------------------------


class _XInputGamepad(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class _XInputState(ctypes.Structure):
    _fields_ = [("dwPacketNumber", ctypes.c_uint), ("Gamepad", _XInputGamepad)]


class Gamepad:
    """Xbox pad read straight off XInput.

    ``vla/teleop.py`` gets the pad from Kit's ``carb.input``, which does not exist in
    this process -- there is no Kit app here, only a pyglet window owned by newton's
    viewer. pyglet's own controller support delivers input as events pumped by
    ``pyglet.app.run()``, which the viewer never calls, so polling XInput directly is
    both simpler and immune to who owns the event loop. Windows only; everywhere else
    this degrades to keyboard.
    """

    A, B, X, Y = 0x1000, 0x2000, 0x4000, 0x8000

    def __init__(self) -> None:
        self._lib = None
        self._state = _XInputState()
        for name in ("XInput1_4", "XInput9_1_0", "XInput1_3"):
            try:
                self._lib = ctypes.WinDLL(name)
                break
            except OSError:
                continue
        self.connected = False
        self.axes = np.zeros(4, dtype=np.float32)  # LX, LY, RX, RY
        self.buttons = 0

    def poll(self) -> None:
        if self._lib is None:
            return
        if self._lib.XInputGetState(0, ctypes.byref(self._state)) != 0:
            if self.connected:
                print("[teleop] gamepad disconnected")
            self.connected = False
            self.axes[:] = 0.0
            self.buttons = 0
            return
        if not self.connected:
            print("[teleop] gamepad connected")
            self.connected = True
        pad = self._state.Gamepad
        self.axes[:] = (
            pad.sThumbLX / 32767.0,
            pad.sThumbLY / 32767.0,
            pad.sThumbRX / 32767.0,
            pad.sThumbRY / 32767.0,
        )
        self.buttons = pad.wButtons

    def held(self, mask: int) -> bool:
        return bool(self.buttons & mask)


def _deadzone(v: float) -> float:
    """vla/teleop.py's dead band, so the sticks feel the same in both scenes."""
    if abs(v) < DEAD_ZONE:
        return 0.0
    sign = 1.0 if v > 0.0 else -1.0
    return float(np.clip(sign * (abs(v) - DEAD_ZONE) / (1.0 - DEAD_ZONE), -1.0, 1.0))


class _Latch:
    """Rising-edge detector over a polled boolean."""

    def __init__(self) -> None:
        self._prev = False

    def rising(self, now: bool) -> bool:
        edge = now and not self._prev
        self._prev = now
        return edge


class Teleop:
    """Valve commands from an XInput pad and the viewer's keyboard, summed."""

    # Keyboard axis pairs: name -> (key for +1, key for -1), keeping vla/teleop.py's
    # sign convention (carriage+ swings the way LEFT did, lift+ lowers the boom).
    KEYS = {
        "carriage": ("j", "l"),
        "lift": ("i", "k"),
        "tilt": ("u", "o"),
        "tool": ("n", "m"),
    }

    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.pad = Gamepad()
        self._reset = _Latch()
        self._autoplay = _Latch()
        self.reset_pressed = False
        self.autoplay_pressed = False
        self.pad.poll()
        if self.pad._lib is None:
            print("[teleop] XInput unavailable -- keyboard only")
        elif not self.pad.connected:
            print("[teleop] no gamepad detected -- keyboard only")

    def poll(self) -> np.ndarray:
        """Poll every source once; returns u in [-1, 1] ordered ``COMMAND_NAMES``."""
        self.pad.poll()
        lx, ly, rx, ry = self.pad.axes

        # Same mapping as vla/teleop.py: left stick is carriage/tilt, right is
        # lift/tool, and three of the four are inverted against raw stick sign.
        stick = np.array(
            [
                -_deadzone(lx),  # carriage
                _deadzone(ry),  # lift
                -_deadzone(ly),  # tilt
                -_deadzone(rx),  # tool
            ],
            dtype=np.float32,
        )

        keys = np.array(
            [
                float(self.viewer.is_key_down(pos)) - float(self.viewer.is_key_down(neg))
                for pos, neg in (self.KEYS[n] for n in COMMAND_NAMES)
            ],
            dtype=np.float32,
        )

        self.reset_pressed = self._reset.rising(
            self.pad.held(Gamepad.B) or self.viewer.is_key_down("t")
        )
        self.autoplay_pressed = self._autoplay.rising(
            self.pad.held(Gamepad.A) or self.viewer.is_key_down("p")
        )
        return np.clip(stick + keys, -1.0, 1.0)


# --- scene --------------------------------------------------------------------


class SoilTeleop(SandExcavator):
    """sand_excavator.py's scene, driven by an operator and tunable while it runs."""

    def __init__(self, viewer, args):
        super().__init__(viewer, args)

        self.teleop = Teleop(viewer)
        self.u = np.zeros(4, dtype=np.float32)
        self.phase_label = "teleop"
        self.autoplay = bool(args.autoplay)
        self.autoplay_time = 0.0
        self.log_path = args.log
        self.log_decim = max(1, int(args.log_every))
        self._step_index = 0
        self._rtf = 0.0
        self._wall_mark = time.perf_counter()
        self._sim_mark = 0.0

        self.command_dof = [self.joint_names.index(n) for n in COMMAND_JOINTS]
        self.vel_max = np.array(
            [args.carriage_vel_max] + [args.arm_vel_max] * 3, dtype=np.float32
        )

        self.surface = None
        self.surface_mode = bool(args.surface)
        self._surface_shown = False
        self.grains = None
        self.grain_radii = None
        self.grain_colors = None
        self._prev_particle_q = None
        if self.use_sand:
            self._setup_soil_look()

        self._snapshot_initial()

        # set_model() clears side-panel callbacks AND builds a fresh Camera, and the
        # base class calls it in __init__, so both of these have to happen after
        # super().__init__().
        self._apply_camera()
        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self._render_ui, position="side")
        if hasattr(viewer, "set_reset_callback"):
            viewer.set_reset_callback(self.reset)

        self._print_controls()

    def _apply_camera(self) -> None:
        """Point the window at the scene, after ``set_model`` has had its say.

        ``ViewerGL.set_model`` does not adjust the camera -- it replaces it, with a
        default that sits 10 m out on +x. Anything set before that call is silently
        thrown away, which is why this runs from ``__init__`` rather than from
        ``make_viewer``.

        The two guards are not the same test: every viewer inherits a ``set_camera``
        (a no-op on the headless ones), but only the GL viewer owns a ``camera`` object
        to read a field of view off.
        """
        viewer, a = self.viewer, self.args
        if hasattr(viewer, "set_camera"):
            viewer.set_camera(wp.vec3(*a.cam_pos), a.cam_pitch, a.cam_yaw)
        camera = getattr(viewer, "camera", None)
        if camera is not None:
            camera.fov = a.cam_fov

    # --- scene ----------------------------------------------------------------

    def _add_robot(self, builder) -> None:
        """Load the machine, then leave collision on the bucket alone.

        The excavator imports about thirty collision meshes and every one of them is
        paid for twice: MuJoCo broad/narrow-phases it against the ground, and the MPM
        collider meshes it, refits its BVH and samples it onto the grid every step.
        Only the bucket ever touches anything -- the machine is fixed-base,
        self-collision is already off at import, and the cab and boom sit well clear
        of a 15 cm cube -- so the rest is pure overhead.

        Two independent flags come off, and both are needed: ``COLLIDE_SHAPES`` is what
        MuJoCo contacts read, ``COLLIDE_PARTICLES`` is what
        :meth:`SolverImplicitMPM.setup_collider` selects on. ``VISIBLE`` is untouched,
        so the machine still draws in full.

        The cost is that a stripped link can be driven through the ground or through
        the soil without resistance. That is the right trade for a demo you drive by
        hand -- and ``--full_collision`` puts every shape back for anyone who needs the
        arm to collide.
        """
        super()._add_robot(builder)
        if self.args.full_collision:
            return

        keep = newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES
        kept = 0
        for i, body in enumerate(builder.shape_body):
            label = builder.body_label[body] if body >= 0 else ""
            if any(k in label for k in TOOL_BODY_KEYS):
                kept += 1
                continue
            builder.shape_flags[i] &= ~keep
        print(f"[soil] collision shapes: {kept} kept on the bucket, {builder.shape_count - kept} stripped")

    def _add_container(self, builder):
        """No walls. The demo is a free-standing pile on open ground.

        ``sand_excavator.py`` boxes the sand in because a scripted dig into a free bed
        is hard to keep repeatable. Nothing here is being measured, so the block stands
        on its own and slumps at its own angle of repose -- which is also the quickest
        way to see what a material change did. Run ``sand_excavator.py`` for the walled
        scene.
        """

    def _add_ground(self, builder):
        """Ground that collides but is never drawn.

        ``is_visible`` clears only ``ShapeFlags.VISIBLE``; the collision flags are
        separate, so the plane still holds the machine and the soil up while the
        viewer draws nothing for it. That leaves the frame black behind the excavator
        and the pile, which is the whole look of the demo.
        """
        cfg = replace(builder.default_shape_cfg, is_visible=False)
        builder.add_ground_plane(cfg=cfg)

    def _emit_sand(self, builder) -> int:
        """Emit the block of soil the machine digs into.

        A cube by default: ``--length`` x ``--width`` on the ground in front of the
        machine, ``--sand_fill`` deep, with its near face at ``--near_x``. Dropped from
        ``--sand_drop`` and left to settle, so the first half-second of any run is the
        block finding its own shape under the current material.
        """
        a = self.args
        x0, x1 = a.near_x, a.near_x + a.length
        y0, y1 = -a.width / 2.0, a.width / 2.0
        lo = np.array([x0, y0, a.sand_drop])
        hi = np.array([x1, y1, a.sand_drop + a.sand_fill])

        res = np.array(np.ceil(a.particles_per_cell * (hi - lo) / a.voxel_size), dtype=int)
        cell = (hi - lo) / res
        radius = float(np.max(cell) * 0.5)

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
            mass=float(np.prod(cell) * a.density),
            jitter=2.0 * radius,
            radius_mean=radius,
        )
        print(
            f"[soil] block x=[{x0:.3f},{x1:.3f}] y=[{y0:.3f},{y1:.3f}] "
            f"depth={a.sand_fill:.3f} m on hidden ground, no container"
        )
        return int(np.prod(res + 1))

    # --- material -------------------------------------------------------------

    def _apply_sand_material(self, model) -> None:
        """Push every material parameter, not just the subset the base class pushes."""
        for key in SOIL_KEYS:
            value = getattr(self.args, key, None)
            if value is not None and hasattr(model.mpm, key):
                getattr(model.mpm, key).fill_(float(value))

    def set_material(self, key: str, value: float) -> None:
        """Change one material parameter on the running solver."""
        setattr(self.args, key, float(value))
        if not self.use_sand:
            return
        getattr(self.sand_model.mpm, key).fill_(float(value))
        # The solver caches feature flags (has_viscosity, has_dilatancy) and extrema
        # derived from these arrays, so an in-place fill on its own is not enough.
        self.mpm_solver.notify_model_changed(SolverNotifyFlags.MODEL_PROPERTIES)

    def apply_preset(self, name: str) -> None:
        for key, value in SOIL_PRESETS[name].items():
            self.set_material(key, value)
        self.args.soil = name
        print(f"[soil] preset {name}: {self.material_flags()}")

    def material_flags(self) -> str:
        """Current material as a CLI fragment, so a good tune survives the session."""
        return " ".join(f"--{k} {getattr(self.args, k):g}" for k in SOIL_KEYS)

    # --- grains ---------------------------------------------------------------

    def _tinted(self, count: int) -> wp.array:
        """Per-point colour with a little scatter.

        A flat colour flattens the relief -- every sphere shades identically and the pile
        reads as one surface. A few percent of scatter is what makes it read as grain.
        """
        rng = np.random.default_rng(0)
        base = np.array(self.args.soil_color, dtype=np.float32)
        tint = np.clip(base + rng.normal(0.0, self.args.soil_color_jitter, (count, 3)), 0.0, 1.0)
        return wp.array(tint, dtype=wp.vec3, device=self.model.device)

    def _setup_soil_look(self) -> None:
        """Decide what actually gets drawn, and how many spheres that is.

        Drawing is not free: the GL viewer instances one low-poly sphere per point, and
        past roughly 50k of them the window starts costing as much as the solver (0.26x
        real time headless against 0.18x on screen at 144k). So the count is the budget,
        and there are two ways to spend it.

        ``--grains 0`` (the default) draws the MPM particles themselves. At the default
        0.012 m voxels that is ~48k spheres of ~3 mm, a packed surface, and every sphere
        is a simulated particle -- the most physics per drawn pixel.

        ``--grains N`` instead scatters N smaller points inside each particle and advects
        them with the particle's deformation gradient. Finer and more obviously granular,
        but the budget then has to come out of the soil resolution:
        ``--voxel_size 0.015 --grains 2`` costs the same as the default and simulates
        half as much soil. Purely cosmetic either way -- no grain touches the solve.
        """
        args = self.args
        ps = self._particle_state
        if args.grains <= 0:
            self.particle_colors = self._tinted(self.sand_model.particle_count)
            print(f"[soil] drawing {self.sand_model.particle_count} particles directly")
            return

        self.grains = self.mpm_solver.sample_render_grains(ps, args.grains)
        radius = args.grain_radius or args.voxel_size / (3.0 * args.grains)
        self.grain_radii = wp.full(
            self.grains.size, value=float(radius), dtype=float, device=self.model.device
        )
        self.grain_colors = self._tinted(self.grains.size)
        self._prev_particle_q = wp.clone(ps.particle_q)
        self.viewer.show_particles = False
        print(f"[soil] drawing {self.grains.size} grains, radius {radius * 1000:.2f} mm")

    def _soil_surface(self) -> SoilSurface | None:
        """The isosurface builder, made on first use.

        Built lazily because it is not free to hold: a grid, a marching-cubes workspace
        and the fixed vertex buffers, none of which a run that never turns the mode on
        should pay for.

        The grid is a fixed box, so soil carried outside it stops being drawn. It
        covers the block plus ``--surface_margin`` on all sides and
        ``--surface_lift`` above, which spans the dig envelope; dump soil further out
        than that and widen the margin.
        """
        if self.surface is not None or not self.use_sand:
            return self.surface
        a = self.args
        x0, x1 = a.near_x, a.near_x + a.length
        lo = np.array([x0, -a.width / 2.0, 0.0]) - a.surface_margin
        hi = np.array([x1, a.width / 2.0, a.sand_drop + a.sand_fill]) + a.surface_margin
        hi[2] += a.surface_lift
        self.surface = SoilSurface(a, lo, hi)
        return self.surface

    def _resample_grains(self) -> None:
        if self.grains is None:
            return
        ps = self._particle_state
        self.grains.assign(self.mpm_solver.sample_render_grains(ps, self.args.grains))
        self._prev_particle_q.assign(ps.particle_q)

    def render_models(self):
        """(model, state) pairs a renderer has to trace to see the whole scene.

        One pair under ``--coupling proxy``, which shares a model; two under
        ``manual``, which does not.
        """
        pairs = [(self.model, self.state_0)]
        if self.use_sand and self.sand_model is not self.model:
            pairs.append((self.sand_model, self.sand_state))
        return pairs

    @property
    def _particle_state(self):
        """Manual coupling keeps particles in their own state; proxy shares one."""
        return self.state_0 if self.coupling == "proxy" else self.sand_state

    # --- reset ----------------------------------------------------------------

    def _snapshot_initial(self) -> None:
        self._init_q_target = self.q_target.copy()
        self._init_joint_q = self.state_0.joint_q.numpy().copy()
        self._init_particles = {}
        self._init_mpm = {}
        if self.use_sand:
            ps = self._particle_state
            self._init_particles = {
                "particle_q": wp.clone(ps.particle_q),
                "particle_qd": wp.clone(ps.particle_qd),
            }
            for name in MPM_STATE_ATTRS:
                array = getattr(ps.mpm, name, None)
                if isinstance(array, wp.array):
                    self._init_mpm[name] = wp.clone(array)

    def _robot_states(self):
        states = [self.state_0]
        if self.state_1 is not self.state_0:
            states.append(self.state_1)
        return states

    def reset(self) -> None:
        """Put machine and soil back to the start pose."""
        self.q_target[:] = self._init_q_target
        for state in self._robot_states():
            state.joint_q.assign(self._init_joint_q)
            state.joint_qd.zero_()
            state.body_f.zero_()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

        if self.use_sand:
            ps = self._particle_state
            for name, array in self._init_particles.items():
                getattr(ps, name).assign(array)
            for name, array in self._init_mpm.items():
                getattr(ps.mpm, name).assign(array)
            if self.coupling == "manual":
                self.body_sand_forces.zero_()
                self.collider_impulses.zero_()
                self.collider_impulse_pos.zero_()
                self.collider_impulse_ids.fill_(-1)
                self.sand_state.body_q.assign(self.state_0.body_q)
                self.sand_state.body_qd.assign(self.state_0.body_qd)
                self.sand_state.body_f.zero_()
            self._resample_grains()

        self.u[:] = 0.0
        self.sim_time = 0.0
        self.autoplay_time = 0.0
        self.log_rows.clear()
        print("[soil] scene reset")

    # --- control --------------------------------------------------------------

    def _autoplay_command(self):
        t = self.autoplay_time
        for label, seconds, u in DIG_CYCLE:
            if t < seconds:
                return label, np.array(u, dtype=np.float32)
            t -= seconds
        return "done", np.zeros(4, dtype=np.float32)

    def poll_input(self) -> None:
        """Read the operator. Runs every frame, including while the viewer is paused."""
        command = self.teleop.poll()
        if self.teleop.reset_pressed:
            self.reset()
            return
        if self.teleop.autoplay_pressed:
            self.autoplay = not self.autoplay
            self.autoplay_time = 0.0
            print(f"[soil] scripted dig {'on' if self.autoplay else 'off'}")
        if not self.autoplay:
            self.u = command
            self.phase_label = "teleop"
        elif np.any(command != 0.0):
            # Touching a stick takes the machine back off autopilot.
            self.autoplay = False
            self.u = command
            self.phase_label = "teleop"
            print("[soil] scripted dig off (operator input)")

    def apply_control(self):
        """Target mode: integrate the valve command into a joint position target.

        Identical to the base class except that the velocities come from ``u`` rather
        than from a scripted phase table -- and identical to what
        ``vla/record_episodes.py`` does through ``VelocityIntegratedActuator``.
        """
        if self.autoplay:
            self.phase_label, self.u = self._autoplay_command()
            self.autoplay_time += self.frame_dt

        qdot = self.u * self.vel_max
        for k, dof in enumerate(self.command_dof):
            self.q_target[dof] = np.clip(
                self.q_target[dof] + qdot[k] * self.frame_dt,
                self.limit_lo[dof],
                self.limit_hi[dof],
            )
        self._target_host.assign(self.q_target.astype(np.float32))
        wp.copy(self.control_target, self._target_host.to(self.model.device))
        return self.phase_label

    # --- simulation -----------------------------------------------------------

    def _log(self, phase: str):
        """Off unless ``--log`` names a file.

        ``sand_excavator.py`` logs because it is a measurement run with a fixed script
        and a fixed length. This one is neither: an interactive session has no end, and
        a row per frame is a memory leak on a demo nobody is measuring. With ``--log``
        it keeps every ``--log_every``-th frame.
        """
        if self.log_path and self._step_index % self.log_decim == 0:
            super()._log(phase)

    def step(self):
        if self.grains is not None:
            # update_render_grains needs where the particles were, and the MPM step
            # below runs in place, so those positions have to be kept by hand.
            self._prev_particle_q.assign(self._particle_state.particle_q)

        super().step()

        if self.use_sand:
            ps = self._particle_state
            if self.args.project_outside:
                # Push particles back out of the tool. Cosmetic, but without it soil
                # visibly sinks into the bucket at the moment of contact.
                self.mpm_solver.project_outside(ps, ps, self.frame_dt)
            if self.grains is not None:
                self.mpm_solver.update_particle_frames(ps, ps, self.frame_dt)
                self.mpm_solver.update_render_grains(
                    types.SimpleNamespace(particle_q=self._prev_particle_q),
                    ps,
                    self.grains,
                    self.frame_dt,
                )

        self._step_index += 1
        self._update_rtf()

    def _update_rtf(self) -> None:
        """Measured real-time factor, averaged over the last second of wall clock."""
        now = time.perf_counter()
        elapsed = now - self._wall_mark
        if elapsed >= 1.0:
            self._rtf = (self.sim_time - self._sim_mark) / elapsed
            self._wall_mark = now
            self._sim_mark = self.sim_time

    # --- output ---------------------------------------------------------------

    def print_status(self) -> None:
        tip = self.tool_tip()
        print(
            f"[{self._step_index:7d}]  {self.phase_label:<9s}"
            f"  u={np.array2string(self.u, precision=2, suppress_small=True)}"
            f"  tip=({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f})"
            f"  soil={self._sand_force_n():6.1f} N  rt={self._rtf:.2f}x"
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.use_sand:
            # Exactly one of the three representations is visible, but whichever is
            # hidden still has to be logged every frame -- dropping the call leaves the
            # viewer drawing whatever it last saw.
            surface = self.surface_mode and self._soil_surface() is not None
            if self.surface is not None:
                if surface:
                    self.surface.update(self._particle_state.particle_q)
                elif self._surface_shown:
                    # Collapse the mesh on the frame the box is unticked. `hidden` alone
                    # would do for the GL window, but the buffers would still hold the
                    # last skin -- which is what a viewer that ignores the flag draws,
                    # and what would flash for one frame on the way back in.
                    self.surface.clear()
                self.viewer.log_mesh(
                    "/soil",
                    points=self.surface.verts,
                    indices=self.surface.indices,
                    normals=self.surface.normals,
                    hidden=not surface,
                    backface_culling=False,
                    color=tuple(self.args.soil_color),
                    roughness=self.args.surface_roughness,
                    metallic=0.0,
                )
            self._surface_shown = surface
            if self.grains is not None:
                self.viewer.log_points(
                    "/grains",
                    points=self.grains.flatten(),
                    radii=self.grain_radii,
                    colors=self.grain_colors,
                    hidden=surface,
                )
            else:
                self.viewer.log_points(
                    "/sand",
                    points=self._particle_state.particle_q,
                    radii=self.sand_model.particle_radius,
                    colors=self.particle_colors,
                    hidden=surface,
                )
        self.viewer.log_scalar("real-time factor", self._rtf)
        self.viewer.end_frame()

    def _print_controls(self) -> None:
        print("\n[soil] pad : L-stick carriage/tilt  R-stick lift/tool  A scripted dig  B reset")
        print("[soil] keys: J/L carriage  I/K lift  U/O tilt  N/M bucket  P scripted dig  T reset")
        print("[soil] view: WASD/QE camera  SPACE pause  . step  F frame  H ui  ESC quit")
        print(f"[soil] material: {self.material_flags()}\n")

    def _render_ui(self, imgui) -> None:
        """Live material sliders. Guarded: an imgui API change must not kill the demo."""
        try:
            imgui.text(f"soil preset: {self.args.soil}")
            for i, name in enumerate(SOIL_PRESETS):
                if i % 3:
                    imgui.same_line()
                if imgui.button(name):
                    self.apply_preset(name)
            imgui.separator()
            for key, label, lo, hi in SOIL_SLIDERS:
                changed, value = imgui.slider_float(label, float(getattr(self.args, key)), lo, hi)
                if changed:
                    self.set_material(key, value)
            imgui.separator()
            changed, self.surface_mode = imgui.checkbox("Smooth surface", self.surface_mode)
            if changed and self.surface_mode:
                self._soil_surface()  # build it on the click, not mid-frame
            if self.surface_mode:
                for key, label, lo, hi in SURFACE_SLIDERS:
                    moved, value = imgui.slider_float(label, float(getattr(self.args, key)), lo, hi)
                    if moved:
                        setattr(self.args, key, value)
            imgui.separator()
            if imgui.button("Print as CLI flags"):
                print(f"[soil] {self.material_flags()}")
            if imgui.button("Reset scene"):
                self.reset()
            changed, self.autoplay = imgui.checkbox("Scripted dig", self.autoplay)
            if changed:
                self.autoplay_time = 0.0
            imgui.text(f"real-time factor  {self._rtf:.2f}x")
        except Exception as exc:  # noqa: BLE001 - the UI is not worth a crash
            imgui.text(f"UI error: {exc}")


# --- offscreen shot -----------------------------------------------------------


class SoilShot:
    """One ray-traced frame of the scene, robot and soil composited on depth.

    ``newton.sensors.SensorTiledCamera`` ray-traces meshes *and* particles, on Warp,
    with no Kit and no window -- which is what makes headless soil tuning possible:
    change a number, render, look, repeat, in the time one Isaac Sim startup takes.

    The one wrinkle is that a sensor is bound to a single model and decides at
    construction whether it has particles at all (from ``model.particle_q``), while
    ``--coupling manual`` deliberately keeps the robot and the soil in two models. So
    this runs one sensor per model over identical rays and merges them on the depth
    buffer. Same reason the two models exist: it is the only path that works on the
    newton 1.2.1 that Isaac Lab pins.
    """

    # The raytracer has no per-particle colour: a particle hit reports this reserved
    # shape id and shades against a white albedo. Which is fine, because white albedo
    # times the lighting is exactly what a tint wants to multiply.
    PARTICLES_SHAPE_ID = 0xFFFFFFFE

    def __init__(self, demo, width: int, height: int, fov_deg: float):
        from newton.sensors import SensorTiledCamera

        self.width, self.height = width, height
        self.demo = demo
        self.tint = np.array(demo.args.soil_color, dtype=np.float32)
        self.passes = []

        for model, state in demo.render_models():
            sensor = SensorTiledCamera(model=model)
            sensor.utils.create_default_light(enable_shadows=True)
            if model.shape_count:
                newton.geometry.build_bvh_shape(model, state)
            if model.particle_count:
                newton.geometry.build_bvh_particle(model, state)
            self.passes.append((
                sensor,
                model,
                state,
                sensor.utils.create_color_image_output(width, height, 1),
                sensor.utils.create_depth_image_output(width, height, 1),
                sensor.utils.create_shape_index_image_output(width, height, 1),
            ))

        self.rays = self.passes[0][0].utils.compute_pinhole_camera_rays(
            width, height, float(np.radians(fov_deg))
        )

    @staticmethod
    def look_at(eye, target) -> wp.array:
        """Camera-to-world transform in the layout ``update()`` wants, ``(1, 1)``.

        The rotation is the *view* rotation, which is what the shipped example feeds
        it, so this borrows pyglet's look-at rather than reinventing the convention.
        """
        from pyglet.math import Mat4, Vec3

        view = np.array(Mat4.look_at(Vec3(*eye), Vec3(*target), Vec3(0.0, 0.0, 1.0)), dtype=np.float32)
        return wp.array(
            [[wp.transformf(wp.vec3f(*eye), wp.quat_from_matrix(wp.mat33f(view.reshape(4, 4)[:3, :3])))]],
            dtype=wp.transformf,
        )

    def render(self, eye, target) -> np.ndarray:
        """RGB uint8, shape (height, width, 3)."""
        transforms = self.look_at(eye, target)
        rgb = None
        best_depth = None
        for sensor, model, state, color, depth, shape_index in self.passes:
            if model.shape_count:
                newton.geometry.refit_bvh_shape(model, state)
            if model.particle_count:
                newton.geometry.refit_bvh_particle(model, state)
            sensor.update(
                state,
                transforms,
                self.rays,
                color_image=color,
                depth_image=depth,
                shape_index_image=shape_index,
                clear_data=_black_clear(),
            )
            hw = (self.height, self.width)
            this_rgb = sensor.utils.to_rgba_from_color(color).numpy().reshape(*hw, 4)[..., :3].astype(np.float32)
            is_soil = shape_index.numpy().reshape(hw) == self.PARTICLES_SHAPE_ID
            this_rgb = np.where(is_soil[..., None], this_rgb * self.tint, this_rgb)

            this_depth = depth.numpy().reshape(hw)
            # A ray that hit nothing comes back at or below zero; make it lose every
            # comparison rather than win them all.
            this_depth = np.where(this_depth > 1.0e-4, this_depth, np.inf)
            if rgb is None:
                rgb, best_depth = this_rgb, this_depth
            else:
                nearer = this_depth < best_depth
                rgb = np.where(nearer[..., None], this_rgb, rgb)
                best_depth = np.minimum(this_depth, best_depth)
        return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def _black_clear():
    """Clear the ray-traced frame to black, matching the live window.

    The shipped presets clear to mid grey or to the default blue; with the ground plane
    hidden that colour *is* the whole background, so it has to be black here too.
    """
    from newton.sensors import SensorTiledCamera

    return replace(SensorTiledCamera.GRAY_CLEAR_DATA, clear_color=0xFF000000)


def take_shot(demo, args) -> None:
    """Run the scripted dig to ``--shot_step`` and write one frame, then return."""
    import imageio.v2 as imageio

    demo.autoplay = True
    for _ in range(args.shot_step):
        demo.step()
    shot = SoilShot(demo, args.shot_size[0], args.shot_size[1], args.shot_fov)
    imageio.imwrite(args.shot, shot.render(args.shot_pos, args.shot_target))
    tip = demo.tool_tip()
    print(f"[soil] shot at step {args.shot_step} ({demo.sim_time:.2f} s), phase {demo.phase_label}")
    print(f"[soil] tip ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f})  soil {demo._sand_force_n():.1f} N")
    print(f"[soil] written: {args.shot}")


# --- entry point --------------------------------------------------------------


def _look_direction(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Unit view direction for a Z-up pitch/yaw pair, as the GL viewer defines them."""
    pitch, yaw = np.radians(pitch_deg), np.radians(yaw_deg)
    return np.array(
        [np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)], dtype=float
    )


def _rehelp(parser, **help_by_flag: str) -> None:
    """Rewrite inherited options' ``--help`` text without redefining them.

    ``build_parser`` describes ``sand_excavator.py``'s walled, scripted run. The
    options this demo reuses mean something slightly different here, and the ones it
    cannot use should not be in ``--help`` at all -- pass ``argparse.SUPPRESS`` for
    those. Either way the flag still parses, so nothing anyone has typed before breaks.
    """
    wanted = {flag.lstrip("-"): text for flag, text in help_by_flag.items()}
    for action in parser._actions:
        for option in action.option_strings:
            if option.lstrip("-") in wanted:
                action.help = wanted[option.lstrip("-")]
                break


def parse_args():
    parser = build_parser()
    parser.description = __doc__
    # A live window instead of a baked USD, and sand tuned for teleop rather than for
    # the scripted measurement run: half the voxel and a 15 cm cube sitting in front of
    # the machine, instead of the wide flat bed sand_excavator.py uses. A cube deep
    # enough to dig a full bucket into is what makes a spoil pile rather than a smear,
    # and --near_x 0.52 puts its near face at the start of the tool-tip envelope
    # (x 0.52..0.72), so the bucket enters the soil rather than reaching over it.
    #
    # MPM cost goes as volume / voxel^3, so the 7.5 mm voxel is 8x per unit volume, but
    # the cube holds far less of it than a wide flat bed does: ~71k particles. Raise
    # --voxel_size back toward 0.015 if that is too slow (~10k) -- the collider margin
    # follows it automatically and the bucket's 1.9 mm shell stays sealed either way.
    parser.set_defaults(
        viewer="gl",
        voxel_size=0.0075,
        near_x=0.52,
        length=0.15,
        width=0.15,
        sand_fill=0.15,
    )
    # None means "take it from --soil"; an explicit flag wins.
    parser.set_defaults(friction=None, viscosity=None, hardening=None)

    g = parser.add_argument_group("soil")
    g.add_argument("--soil", default="damp_sand", choices=sorted(SOIL_PRESETS), help="material preset")
    g.add_argument("--yield_stress", type=float, default=None, help="cohesion [Pa]; 0 = free-flowing")
    g.add_argument("--tensile_yield_ratio", type=float, default=None, help="tension taken before separating, 0..1")
    g.add_argument("--dilatancy", type=float, default=None)
    g.add_argument("--damping", type=float, default=None, help="elastic relaxation time [s]")
    g.add_argument(
        "--full_collision",
        action="store_true",
        help="keep collision on every machine shape; default is bucket-only, which is "
        "faster and is all this scene needs",
    )
    g.add_argument(
        "--project_outside",
        type=int,
        default=1,
        help="1 = push particles back out of the tool after each step (cosmetic)",
    )

    g = parser.add_argument_group("looks")
    g.add_argument(
        "--grains",
        type=int,
        default=0,
        help="render points scattered inside each particle; 0 = draw the particles themselves",
    )
    g.add_argument("--grain_radius", type=float, default=0.0, help="[m]; 0 = voxel_size / (3 * grains)")
    g.add_argument("--soil_color", type=float, nargs=3, default=[0.72, 0.62, 0.44])
    g.add_argument(
        "--surface",
        action="store_true",
        help="start with the soil drawn as one smooth surface instead of particles; "
        "also a checkbox in the side panel",
    )
    g.add_argument("--surface_voxel", type=float, default=0.006, help="isosurface grid spacing [m]")
    g.add_argument("--surface_iso", type=float, default=1.0, help="density level the skin sits at")
    g.add_argument("--surface_smooth", type=float, default=1.0, help="splat width, in grid cells")
    g.add_argument("--surface_reach", type=int, default=2, help="splat radius, in grid cells")
    g.add_argument("--surface_roughness", type=float, default=0.85, help="0 = wet and glossy, 1 = matte")
    g.add_argument("--surface_margin", type=float, default=0.22, help="grid padding around the block [m]")
    g.add_argument("--surface_lift", type=float, default=0.25, help="extra grid headroom above the block [m]")
    # Sized off a measured dig, which peaks near 37k triangles. This is not a cap that
    # only costs memory: the viewer repacks and re-uploads the whole buffer every frame
    # whether or not the triangles are used, so 200k measured 18.8 ms/frame against
    # 10.1 ms at 50k. Keep it just above the peak.
    g.add_argument("--surface_max_tris", type=int, default=80000, help="fixed triangle budget")
    g.add_argument("--soil_color_jitter", type=float, default=0.035, help="per-grain colour scatter")
    g.add_argument("--window", type=int, nargs=2, default=[1600, 900])
    # Eye plus pitch/yaw/FOV rather than eye plus aim point: these are exactly what the
    # viewer stores, so what the window's camera readout shows after you have dragged
    # the view into a shot you like can be pasted straight back in as defaults.
    g.add_argument("--cam_pos", type=float, nargs=3, default=[-0.39, 0.98, 0.76], help="camera eye [m]")
    g.add_argument("--cam_pitch", type=float, default=-21.9, help="camera pitch [deg]")
    g.add_argument("--cam_yaw", type=float, default=-55.2, help="camera yaw [deg]")
    g.add_argument("--cam_fov", type=float, default=45.0, help="vertical field of view [deg]")

    g = parser.add_argument_group("shot")
    g.add_argument("--shot", default="", help="ray-trace one frame to this PNG and exit; no window")
    g.add_argument(
        "--shot_step",
        type=int,
        default=640,
        help="steps of the scripted dig before the shot; 640 is just after the bucket "
        "lifts clear, which is when you can see whether the cut held",
    )
    g.add_argument("--shot_size", type=int, nargs=2, default=[768, 512])
    # None everywhere: the shot uses the window's camera unless it is told otherwise, so
    # there is one default view to keep good rather than two to keep in sync. The
    # raytracer wants an aim point where the window wants angles, hence --shot_target.
    g.add_argument("--shot_pos", type=float, nargs=3, default=None, help="camera eye [m]; default --cam_pos")
    g.add_argument(
        "--shot_target", type=float, nargs=3, default=None,
        help="aim point [m]; default is one metre along --cam_pitch / --cam_yaw",
    )
    g.add_argument("--shot_fov", type=float, default=None, help="vertical FOV [deg]; default --cam_fov")

    g = parser.add_argument_group("teleop")
    g.add_argument("--arm_vel_max", type=float, default=0.5, help="rad/s at full stick")
    g.add_argument("--carriage_vel_max", type=float, default=0.8, help="rad/s at full stick")
    g.add_argument("--autoplay", action="store_true", help="start with the scripted dig running")
    g.add_argument("--status_every", type=int, default=30, help="frames between status lines")
    g.add_argument("--log", default="", help="write a tracking CSV here; empty = no logging")
    g.add_argument("--log_every", type=int, default=6, help="frames between CSV rows, with --log")

    # sand_excavator's "container" group sizes a walled box. There is no box here, so
    # the three wall options are dead and the three that survive describe the soil cube
    # instead. --log_path goes too; this script has its own opt-in --log.
    for group in parser._action_groups:
        if group.title == "container":
            group.title = "soil block"
    _rehelp(
        parser,
        depth=argparse.SUPPRESS,
        wall_thickness=argparse.SUPPRESS,
        wall_friction=argparse.SUPPRESS,
        log_path=argparse.SUPPRESS,
        near_x="near face of the soil block [m], measured from the machine",
        length="block size along x [m]",
        width="block size along y [m], centred on y = 0",
        sand_fill="block size along z [m]",
        num_frames="0 = run until the window is closed",
    )

    args = parser.parse_args()
    if args.status_every < 1:
        parser.error("--status_every must be at least 1")

    # The preset fills in whatever was not given explicitly.
    for key, value in SOIL_PRESETS[args.soil].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    # ... and the window's camera fills in the shot's.
    if args.shot_pos is None:
        args.shot_pos = list(args.cam_pos)
    if args.shot_fov is None:
        args.shot_fov = args.cam_fov
    if args.shot_target is None:
        args.shot_target = list(np.array(args.shot_pos) + _look_direction(args.cam_pitch, args.cam_yaw))
    return args


def _blacken(viewer) -> None:
    """Turn the GL window's sky and fog off so the background is flat black.

    Hiding the ground plane is only half of it: the renderer still paints a sky
    gradient behind everything and fogs distant geometry towards ``sky_lower``. With
    both off and both colours black, the only lit things in frame are the excavator and
    the soil, which is the look this demo is after. Guarded by ``hasattr`` because the
    null, USD and rerun viewers have no renderer to change.
    """
    renderer = getattr(viewer, "renderer", None)
    if renderer is None:
        return
    renderer.draw_sky = False
    renderer.background_color = (0.0, 0.0, 0.0)
    renderer.sky_upper = (0.0, 0.0, 0.0)  # also the glClearColor
    renderer.sky_lower = (0.0, 0.0, 0.0)  # also the fog colour


def make_viewer(args):
    if args.viewer == "gl":
        viewer = newton.viewer.ViewerGL(width=args.window[0], height=args.window[1])
        _blacken(viewer)
        # The camera is deliberately NOT set here: see SoilTeleop._apply_camera.
        return viewer
    if args.viewer == "usd":
        return newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames or 600)
    if args.viewer == "rerun":
        return newton.viewer.ViewerRerun()
    return newton.viewer.ViewerNull(num_frames=args.num_frames or 600)


def main():
    args = parse_args()
    if args.shot:
        # Nothing to drive and nothing to draw: the sensor does its own rendering.
        args.viewer = "null"
    viewer = make_viewer(args)
    demo = SoilTeleop(viewer, args)

    if args.shot:
        take_shot(demo, args)
        viewer.close()
        return

    frame = 0
    while viewer.is_running():
        if args.num_frames and frame >= args.num_frames:
            break
        demo.poll_input()
        if viewer.should_step():
            demo.step()
            frame += 1
            if frame % args.status_every == 0:
                demo.print_status()
        demo.render()

    viewer.close()
    if args.log:
        demo.write_log(args.log)
    print(f"[soil] final material: {demo.material_flags()}")


if __name__ == "__main__":
    main()
