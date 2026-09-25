"""The quad's Newton Articulation: the stock Isaac Lab 3 Newton articulation plus a substep foot-contact
tracker registered inside Newton's graphed decimation loop (see contact_tracker.py).

The tracker has to exist before the env captures the CUDA graph (``set_decimation`` runs before the
managers are built), so it is created here, on PHYSICS_READY, rather than by the reward term.
"""

from __future__ import annotations

from isaaclab_newton.assets.articulation import Articulation as NewtonArticulation
from isaaclab_newton.physics import NewtonManager

from . import spec
from .contact_tracker import FootContactTracker


class QuadArticulation(NewtonArticulation):
    contact_tracker: FootContactTracker | None = None

    def _initialize_impl(self):
        super()._initialize_impl()
        if self.contact_tracker is None:
            self.contact_tracker = FootContactTracker(
                self.num_instances, self.device, spec.LOWER_LEGS, spec.FOOT_POINTS[0], spec.TORSO, spec.PHYSICS_DT,
                spec.UPPER_LEGS,
            )
            NewtonManager.register_post_actuator_callback(self.contact_tracker.graph_step)
            if spec.GROUND_PRIORITY:
                self._set_ground_priority(spec.GROUND_PRIORITY)

    @staticmethod
    def _set_ground_priority(priority: int) -> None:
        """Round 9 soft feet: give the ground MuJoCo priority so its (pair) contact parameters win (see spec.py).
        Written into the Newton model's ``mujoco.geom_priority`` on PHYSICS_READY, i.e. before SolverMuJoCo builds
        the MuJoCo model from it, so the value also survives Newton's later shape-property updates."""
        model = NewtonManager.get_model()
        body = model.shape_body.numpy()
        ground = [i for i in range(model.shape_count) if body[i] == -1]
        if len(ground) != 1:
            raise RuntimeError(f"expected one static (ground) shape, found {ground}")
        prio = model.mujoco.geom_priority
        vals = prio.numpy()
        vals[ground[0]] = priority
        prio.assign(vals)
