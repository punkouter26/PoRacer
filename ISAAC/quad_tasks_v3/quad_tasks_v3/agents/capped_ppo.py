"""RSL-RL PPO with its adaptive-KL learning rate capped at QUAD_SPEC's 3e-4 (round 4).

RSL-RL 5.0.1's adaptive schedule, per mini-batch: KL > 2 x desired -> lr = max(1e-5, lr / 1.5);
KL < desired / 2 -> lr = min(1e-2, lr * 1.5). QUAD_SPEC wants the same rule with 3e-4 as the ceiling, so it
can only lower the lr below the spec value and climb back to it. The cap is applied by making
``learning_rate`` a clamping property; every line of PPO.update() is RSL-RL's own.
"""

from rsl_rl.algorithms import PPO

LR_CAP = 3.0e-4


class CappedPPO(PPO):
    @property
    def learning_rate(self) -> float:
        return self._capped_lr

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self._capped_lr = min(LR_CAP, float(value))
