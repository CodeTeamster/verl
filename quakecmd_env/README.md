# Quakecmd ALFWorld GRPO entrypoint

`launch_alfworld.sh` is the Quake entry file. Quake starts the Ray cluster when
`LAUNCH_RAY=1`; only local rank 0 starts the Verl driver and Ray schedules its
actors on all allocated pods.

Use `requirements.txt`. The selected image must already provide vLLM, Torch,
and the accelerator-specific attention kernels. The requirements file contains
only shared Python dependencies and does not install an accelerator runtime.

The entrypoint follows `lab/qwen2_5_alfworld_grpo.sh`. TensorBoard writes to
Quake's injected `QUAKE_TENSORBOARD_DIR` when it is available. All training-sensitive
values are environment variables and are passed by the scripts under
`quakecmd_submission/verl`, including batch sizes, rollout count, sequence
lengths, learning rate, KL coefficient, precision, rollout TP, and epochs.

For a resume job, point `RUN_DIR` at the original run directory and pass
`RESUME_MODE=auto`. Checkpoints remain in `${RUN_DIR}/ckpts`; no output path is
recreated under a new timestamp.
