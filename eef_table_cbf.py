"""
eef_table_cbf.py

A single-constraint CBF: keep the Kinova end-effector (pinch_site) above the
tabletop plane, whether the commanded motion comes from keyboard teleop or
an autonomous policy.

=========================== THE MATH (recap) ===============================

  h(q)   = z_eef(q) - z_table_surface - margin        (>0 means safe)
  hdot   = J_z(q) . qdot                               (only z-Jacobian matters)
  CBF condition:      hdot >= -alpha * h
  Rearranged for a QP ("G qdot <= h_bound" form):
                       -J_z(q) . qdot  <=  alpha * h

Given a nominal joint velocity qdot_nom (from teleop or a policy), we solve:

    min_{qdot}  || qdot - qdot_nom ||^2
    s.t.        -J_z(q) . qdot  <=  alpha * h

This is a 7-variable QP with a single scalar constraint -- about as simple
as a CBF-QP gets, which is exactly why it's a good first one to implement.

See cbf_table_safety.py's docstring for the control-affine / Lie-derivative
formulation (x=q, u=qdot, xdot=f(x)+g(x)u with f=0, g=I) -- the same
structure applies here with a single h(q) instead of h_i(q) per geom pair.

=========================== TELEOPERATION ===================================

We don't command joint velocities directly with the keyboard -- that would be
awkward (which key moves which of 7 joints?). Instead we command a Cartesian
end-effector velocity (e.g. "move eef in +z") and convert it to joint
velocities using the Jacobian pseudo-inverse:

    qdot_teleop = J_full(q)^+ @ cartesian_velocity_command

where J_full is the eef's full 3x7 translational Jacobian and `^+` denotes
the Moore-Penrose pseudo-inverse (numpy: np.linalg.pinv). This is standard
"differential inverse kinematics" -- it finds the joint velocities that best
reproduce the desired Cartesian motion (exactly, when possible; least-squares
approximation otherwise, since 7 joints can produce a given 3D velocity in
infinitely many ways -- pinv picks the minimum-norm one).

That qdot_teleop is our "nominal" command. It then gets passed through the
same CBF-QP filter described above before being applied.
"""

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer
from qpsolvers import solve_qp

MODEL_PATH = 'models/stanford_tidybot/scene_table.xml'
ARM_JOINT_NAMES = [f'joint_{i}' for i in range(1, 8)]

# From your edited scene_table.xml: body pos.z=0.15, table_top local pos.z=0.15,
# table_top half-thickness=0.02 -> surface = 0.15 + 0.15 + 0.02
Z_TABLE_SURFACE = 0.32
MARGIN = 0.08       # stay at least 2cm above the table
ALPHA = 5.0         # CBF class-K gain
CARTESIAN_SPEED = 0.15  # m/s commanded per key hold
KEY_TIMEOUT = 0.15  # seconds before a key's velocity contribution decays to 0


class EefTableCBF:
    """
    Control-affine system:  xdot = f(x) + g(x) u
        x = q    in R^7   (arm joint angles)
        u = qdot in R^7   (arm joint velocities -- what we command)

    Single integrator: f(x) = 0 (no drift), g(x) = I_7 (identity).

    Constrains MULTIPLE points on the gripper, not just pinch_site --
    pinch_site is a fixed virtual point on the gripper's mount body and
    does NOT track the actual finger pads, which hang on their own
    separate linkage (driver -> coupler/spring_link -> follower -> pad).
    Measured gap: the real pads can sit several mm lower than pinch_site
    depending on approach angle, enough to slip through pinch_site's
    margin undetected. So we give every one of these points its own
    barrier h_i and its own row in the QP -- the safe command has to
    satisfy ALL of them simultaneously.
    """

    # (kind, name) for every point we protect. 'site' uses data.site_xpos /
    # model.site_bodyid; 'geom' uses data.geom_xpos / model.geom_bodyid.
    CONSTRAINED_POINTS = [
        ('site', 'pinch_site'),
        ('geom', 'right_pad1'),
        ('geom', 'right_pad2'),
        ('geom', 'left_pad1'),
        ('geom', 'left_pad2'),
    ]

    def __init__(self, model, data):
        self.model = model
        self.data = data

        arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES]
        self.arm_dof_ids = [model.jnt_dofadr[j] for j in arm_joint_ids]
        self.arm_qpos_adrs = [model.jnt_qposadr[j] for j in arm_joint_ids]
        self.n = len(self.arm_dof_ids)  # = 7

        # Resolve each constrained point's id + which array/body table to use.
        self._points = []
        for kind, name in self.CONSTRAINED_POINTS:
            if kind == 'site':
                pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                body_id = model.site_bodyid[pid]
            else:
                pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                body_id = model.geom_bodyid[pid]
            if pid < 0:
                raise RuntimeError(f'{kind} "{name}" not found in model')
            self._points.append((kind, pid, body_id))

    # kept for backwards compatibility / teleop's differential-IK target (still pinch_site)
    def eef_pos(self):
        return self.data.site_xpos[self._points[0][1]].copy()

    def full_jacobian(self):
        """3x7 translational Jacobian of pinch_site (used for teleop's diff-IK target,
        NOT for the CBF constraints themselves -- those use point_jacobian below)."""
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, None, self.eef_pos(), self._points[0][2])
        return jacp[:, self.arm_dof_ids]

    def _point_world_pos(self, kind, pid):
        return (self.data.site_xpos if kind == 'site' else self.data.geom_xpos)[pid].copy()

    def _point_jacobian(self, kind, pid, body_id):
        """3x7 translational Jacobian of one constrained point, arm columns only."""
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, None, self._point_world_pos(kind, pid), body_id)
        return jacp[:, self.arm_dof_ids]

    # ------------------------- dynamics: xdot = f(x) + g(x) u -------------------------

    def f(self):
        """Drift term. Single integrator -> no drift: f(x) = 0 in R^7."""
        return np.zeros(self.n)

    def g(self):
        """Input matrix. Single integrator -> commanded velocity passes through
        unchanged: g(x) = I_7 (identity, so g(x) @ u = u)."""
        return np.eye(self.n)

    # ------------------------------- barriers + Lie derivatives (per point) -------------

    def compute_h_all(self):
        """h_i(x) for every constrained point -- length-5 array."""
        return np.array([
            self._point_world_pos(kind, pid)[2] - Z_TABLE_SURFACE - MARGIN
            for kind, pid, _ in self._points
        ])

    def grad_h_all(self):
        """grad(h_i)(x) = J_z_i(q) for every point -- shape (5, 7)."""
        return np.array([
            self._point_jacobian(kind, pid, body_id)[2, :]
            for kind, pid, body_id in self._points
        ])

    def L_f_h_all(self):
        """L_f h_i(x) = grad(h_i)(x) . f(x). Zero drift -> always 0 for every i."""
        return self.grad_h_all() @ self.f()

    def L_g_h_all(self):
        """L_g h_i(x) = grad(h_i)(x) . g(x). g(x)=I -> equals grad(h_i)(x) itself."""
        return self.grad_h_all() @ self.g()

    def compute_h(self):
        """Backwards-compatible scalar h: the WORST (smallest, most dangerous) of
        the 5 point barriers -- this is what actually determines safety."""
        return float(np.min(self.compute_h_all()))

    def filter(self, qdot_nom):
        """CBF-QP safety filter: closest control u=qdot to qdot_nom satisfying,
        for EVERY constrained point i simultaneously:
              L_f h_i(x) + L_g h_i(x) . u  >=  -alpha * h_i(x)
        i.e. (QP "G u <= b" form, one row per point):
              -L_g h_i(x) . u  <=  alpha * h_i(x) + L_f h_i(x)
        """
        h_all = self.compute_h_all()          # (5,)
        Lf_all = self.L_f_h_all()              # (5,) -- all zero here
        Lg_all = self.L_g_h_all()              # (5, 7)

        P = np.eye(self.n)
        q = -qdot_nom
        G = -Lg_all                            # (5, 7)
        h_bound = ALPHA * h_all + Lf_all       # (5,)

        qdot_safe = solve_qp(P, q, G, h_bound, solver='quadprog')
        if qdot_safe is None:
            return np.zeros(self.n), True
        active = bool(np.any(G @ qdot_safe > h_bound - 1e-6))
        return qdot_safe, active


class KeyboardTeleop:
    """Tracks which directional keys are currently 'held' (via repeat-decay)."""

    # WASD + Q/E for a simple 3D Cartesian jog. GLFW key codes match ASCII
    # for letters, so we can just compare against ord('W') etc.
    KEY_MAP = {
        ord('W'): np.array([0.0, 0.0, 1.0]),   # +z (up)
        ord('S'): np.array([0.0, 0.0, -1.0]),  # -z (down) -- this is the one the CBF should fight
        ord('A'): np.array([0.0, 1.0, 0.0]),   # +y
        ord('D'): np.array([0.0, -1.0, 0.0]),  # -y
        ord('Q'): np.array([1.0, 0.0, 0.0]),   # +x
        ord('E'): np.array([-1.0, 0.0, 0.0]),  # -x
    }

    def __init__(self):
        self.last_press_time = {k: 0.0 for k in self.KEY_MAP}

    def key_callback(self, keycode):
        if keycode in self.last_press_time:
            self.last_press_time[keycode] = time.time()

    def cartesian_velocity(self):
        v = np.zeros(3)
        now = time.time()
        for keycode, direction in self.KEY_MAP.items():
            if now - self.last_press_time[keycode] < KEY_TIMEOUT:
                v += direction
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            v = v / norm  # normalize so diagonal jogs aren't faster
        return v * CARTESIAN_SPEED


def run(use_viewer=True):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    cbf = EefTableCBF(model, data)
    teleop = KeyboardTeleop()

    home_qpos = np.array([0.0, -0.349, 3.1416, -2.548, 0.0, -0.873, 1.571])
    for adr, qval in zip(cbf.arm_qpos_adrs, home_qpos):
        data.qpos[adr] = qval
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    viewer_ctx = None
    if use_viewer:
        viewer_ctx = mujoco.viewer.launch_passive(model, data, key_callback=teleop.key_callback)
        print('Controls: I/K = up/down, J/L = left/right, U/O = forward/back.')
        print('Try holding K to drive the eef down through the table -- the CBF should stop it.')

    print(f"{'t':>6} {'h (m)':>10} {'cbf active':>10}")
    t = 0.0
    last_print = 0.0
    running = True
    while running:
        step_start = time.time()
        mujoco.mj_forward(model, data)

        cart_vel = teleop.cartesian_velocity()
        J = cbf.full_jacobian()
        qdot_nom = np.linalg.pinv(J) @ cart_vel  # differential IK

        qdot_safe, active = cbf.filter(qdot_nom)

        for adr, dq in zip(cbf.arm_qpos_adrs, qdot_safe):
            data.qpos[adr] += dq * dt

        if viewer_ctx is not None:
            viewer_ctx.sync()
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            running = viewer_ctx.is_running()
        else:
            running = t < 5.0  # headless smoke-test cap

        t += dt
        if t - last_print > 0.5:
            last_print = t
            print(f'{t:>6.1f} {cbf.compute_h():>10.4f} {str(active):>10}')

    if viewer_ctx is not None:
        viewer_ctx.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-viewer', action='store_true',
                         help='Run headless (for quick smoke-testing without a display).')
    args = parser.parse_args()
    run(use_viewer=not args.no_viewer)