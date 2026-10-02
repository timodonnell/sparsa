# Dedicated A100 run status

The [plan](PLAN.md) defines two larger, sequence-only triangle diffusion arms:
G4-U4 (315,297,028 parameters) and G4-L4 (309,359,620 parameters). Each uses an
independent dedicated node with eight A100-SXM4-80GB GPUs connected by NVLink.

Single-GPU profiling is complete; see `profiles/`. Both fit crop 384 with
microbatch 8 at about 52 GiB allocated, with finite losses and gradients.
The local regression suite passes **82 tests**. Distributed launch checks and
production launch evidence will be recorded here after they complete.

Neither arm has a scientific evaluation result yet. Smoke evaluations are
infrastructure checks and must not enter learning curves or selection decisions.
