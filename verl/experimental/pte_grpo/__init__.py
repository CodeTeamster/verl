"""Opt-in PTE-regularized GRPO for unchanged multi-turn agent rollouts."""

# Keep package import lightweight: the rollout config imports ``agent_loop``
# explicitly, which registers the estimator and manager on the trainer.
__all__: list[str] = []
