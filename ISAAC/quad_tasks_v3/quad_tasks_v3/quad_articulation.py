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
