"""
Simple Control Barrier Function (CBF) safety filter: keep the Kinova arm
off the table.

=================== CONTROL-AFFINE FORMULATION ==============================

State and control:
    x = q    in R^7      (arm joint configuration)
    u = qdot in R^7      (arm joint velocity -- what we actually command)

This is a *kinematic* / velocity-level CBF: we command velocity directly,
so the dynamics are a single integrator, control-affine with zero drift:
    xdot = f(x) + g(x) u  =  0  +  I_7 u

  f(x) = 0     (the arm does not move on its own if u = 0 -- no drift)
  g(x) = I_7   (commanded velocity IS the state's rate of change)

For every arm geom that could hit the table, define a barrier function
using MuJoCo's mj_geomDistance:
    h_i(x) = dist(arm_geom_i(x), table_geom) - margin      (h_i > 0: safe)

Its time derivative, via the chain rule under control-affine dynamics,
splits into two Lie derivatives:
    hdot_i(x, u) = grad(h_i)(x) . xdot
                 = grad(h_i)(x).f(x)  +  grad(h_i)(x).g(x) . u
                 =     L_f h_i(x)     +      L_g h_i(x)     . u

  L_f h_i(x) = grad(h_i)(x) . f(x) = 0      (zero drift -> always 0 here)
  L_g h_i(x) = grad(h_i)(x) . g(x) = grad(h_i)(x) = -n_i^T J_i(q)

    n_i = unit vector from the arm's nearest point toward the table
          (from mj_geomDistance's fromto output)
    J_i(q) = translational Jacobian of that nearest point on the arm
             geom's body (from mj_jac)

So for this system: hdot_i = L_f h_i(x) + L_g h_i(x).u = -n_i^T J_i(q).qdot.
The Lie-derivative split is trivial here only because f(x)=0 (a kinematic
model with no drift). It becomes essential once you move to a dynamic /
torque-level CBF: there x=(q,qdot), f(x) is the robot's actual (nonzero)
drift dynamics, and h (a function of q alone) has *relative degree 2* with
respect to torque -- L_g h(x) = 0 at first order, so torque doesn't appear
in hdot at all, only in hddot. That's exactly the motivation for
Higher-Order CBFs (HOCBFs): differentiate h twice before u shows up.

CBF condition (general Lie-derivative form), for a safety margin alpha > 0:
    L_f h_i(x) + L_g h_i(x) . u  >=  -alpha * h_i(x)

which for our zero-drift system reduces to:
    -n_i^T J_i(q) . qdot  >=  -alpha * h_i(q)
     (n_i^T J_i(q)) . qdot <=  alpha * h_i(q)      <- QP-ready "G u <= b" form

This is the standard CBF condition: "h is allowed to decrease, but not
faster than a rate proportional to how close it already is to zero" --
guaranteeing h_i never crosses zero if this holds for all time and
h_i(q0) >= 0.

Given a *nominal* command qdot_nom (from teleop or an autonomous policy),
we solve a small QP to find the *closest* safe command:

    min_{qdot}  || qdot - qdot_nom ||^2
    s.t.        (n_i^T J_i(q)) . qdot <= alpha * h_i(q)   for every risky pair i

This is the classic "CBF-QP safety filter": it barely touches the nominal
command when things are safe, and firmly pushes back only when the arm
gets close to the table.

This script demonstrates the filter running inside a MuJoCo sim with a
table added in front of the robot (models/stanford_tidybot/scene_table.xml).
It works the same way whether qdot_nom comes from a human teleoperator
or an autonomous controller -- the filter doesn't care about the source,
it just guards the final command before it's sent to the low-level
position/velocity controller.
"""

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer
from qpsolvers import solve_qp

MODEL_PATH = 'models/stanford_tidybot/scene_table.xml'

# Arm joints we control at the velocity level for this demo (7-DoF Kinova).
ARM_JOINT_NAMES = [f'joint_{i}' for i in range(1, 8)]

# Which geoms count as "the arm" (anything that could hit the table).
# We match by body name substring so we don't have to hardcode geom ids
# that vary between MJCF versions.
ARM_BODY_KEYWORDS = [
    'gen3/base_link', 'shoulder_link', 'half_arm_1_link', 'half_arm_2_link',
    'forearm_link', 'spherical_wrist_1_link', 'spherical_wrist_2_link',
    'bracelet_link', 'base', 'right_', 'left_',
]

TABLE_GEOM_NAMES = ['table_top', 'table_leg1', 'table_leg2', 'table_leg3', 'table_leg4']

ALPHA = 5.0          # CBF class-K gain: higher = reacts closer to the limit, more aggressive
SAFETY_DISTMAX = 0.30  # only bother computing distance for geoms within this range (m)
MARGIN = 0.02        # extra buffer so h=0 means "2cm before actual contact"


class TableCBF:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.arm_joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES
        ]
        self.arm_dof_ids = [model.jnt_dofadr[j] for j in self.arm_joint_ids]
        self.nv_arm = len(self.arm_dof_ids)

        self.arm_geom_ids = self._collect_geoms(ARM_BODY_KEYWORDS)
        self.table_geom_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in TABLE_GEOM_NAMES
        ]
        if any(g < 0 for g in self.table_geom_ids):
            raise RuntimeError('Table geoms not found -- did you load scene_table.xml?')

    def _collect_geoms(self, body_keywords):
        geom_ids = []
        for g in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[g]
            body_name = self.model.body(body_id).name
            if any(k in body_name for k in body_keywords):
                geom_ids.append(g)
        return geom_ids

    def _jacobian_at_point(self, body_id, point_world):
        """Translational Jacobian (3 x nv) of a world-frame point rigidly
        attached to body_id, restricted to the arm's velocity columns."""
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, None, point_world, body_id)
        return jacp[:, self.arm_dof_ids]

    def compute_constraints(self):
        """Returns (G, h) such that G @ qdot_arm <= h enforces the CBF
        condition for every arm-table geom pair within range."""
        G_rows = []
        h_rows = []
        fromto = np.zeros(6)

        for ag in self.arm_geom_ids:
            for tg in self.table_geom_ids:
                dist = mujoco.mj_geomDistance(self.model, self.data, ag, tg, SAFETY_DISTMAX, fromto)
                if dist >= SAFETY_DISTMAX:
                    continue  # far away, no constraint needed

                p_arm = fromto[0:3]
                p_table = fromto[3:6]
                vec = p_table - p_arm
                norm = np.linalg.norm(vec)
                if norm < 1e-9:
                    continue
                n = vec / norm  # points from arm surface toward table

                h = dist - MARGIN
                body_id = self.model.geom_bodyid[ag]
                J = self._jacobian_at_point(body_id, p_arm)

                # CBF condition:  -n^T J qdot  >=  -alpha * h
                # Rewritten as a "<=" row for the QP solver:
                #   (n^T J) qdot  <=  alpha * h
                G_rows.append(n @ J)
                h_rows.append(ALPHA * h)

        if not G_rows:
            return None, None
        return np.array(G_rows), np.array(h_rows)

    def filter(self, qdot_nom_arm):
        """Given a nominal 7-vector of desired arm joint velocities, return
        the closest command that keeps every arm-table distance safe."""
        G, h = self.compute_constraints()
        n = self.nv_arm
        P = np.eye(n)
        q = -qdot_nom_arm

        if G is None:
            # nothing nearby, command passes through unchanged
            return qdot_nom_arm, False

        qdot_safe = solve_qp(P, q, G, h, solver='quadprog')
        if qdot_safe is None:
            # infeasible (shouldn't happen with a single obstacle and margin
            # room) -- fail safe by stopping the arm rather than passing
            # through something unsafe
            return np.zeros(n), True
        active = bool(np.any(G @ qdot_safe > h - 1e-6))
        return qdot_safe, active


def demo(use_viewer=False):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cbf = TableCBF(model, data)

    # Start the arm in a folded pose, then command it to reach forward
    # and down -- straight toward the tabletop -- to show the CBF engage.
    home_qpos = np.array([0.0, -0.349, 3.1416, -2.548, 0.0, -0.873, 1.571])
    for jid, qval in zip(cbf.arm_joint_ids, home_qpos):
        data.qpos[model.jnt_qposadr[jid]] = qval
    mujoco.mj_forward(model, data)

    qdot_nominal = np.array([0.0, 0.15, 0.0, 0.15, 0.0, 0.1, 0.0])  # slower drive toward table
    dt = model.opt.timestep
    n_steps = 3000  # ~6 seconds of sim time at dt=0.002, enough to actually watch

    viewer_ctx = mujoco.viewer.launch_passive(model, data) if use_viewer else None

    print(f"{'step':>4} {'min table dist (m)':>20} {'cbf active':>10}")
    for step in range(n_steps):
        step_start = time.time()
        mujoco.mj_forward(model, data)
        qdot_safe, active = cbf.filter(qdot_nominal)

        # integrate arm joints only, for this kinematic demo
        for jid, dq in zip(cbf.arm_joint_ids, qdot_safe):
            adr = model.jnt_qposadr[jid]
            data.qpos[adr] += dq * dt

        if viewer_ctx is not None:
            viewer_ctx.sync()
            # keep it roughly real-time so it's watchable
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            if not viewer_ctx.is_running():
                break

        if step % 300 == 0:
            fromto = np.zeros(6)
            dists = [
                mujoco.mj_geomDistance(model, data, ag, tg, SAFETY_DISTMAX, fromto)
                for ag in cbf.arm_geom_ids for tg in cbf.table_geom_ids
            ]
            min_dist = min(dists) if dists else float('nan')
            print(f'{step:>4} {min_dist:>20.4f} {str(active):>10}')

    print('\nMotion finished: the arm should be resting just above the table')
    print('instead of having driven qdot_nominal straight through it.')

    if viewer_ctx is not None:
        print('Viewer window is still open -- close it (or Ctrl+C here) when done looking.')
        while viewer_ctx.is_running():
            time.sleep(0.1)
        viewer_ctx.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewer', action='store_true',
                         help='Open an interactive MuJoCo viewer window (requires a display, e.g. via xrdp). '
                              'Set MUJOCO_GL=glfw before running with this flag.')
    args = parser.parse_args()
    demo(use_viewer=args.viewer)