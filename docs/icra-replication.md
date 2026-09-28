# ICRA 2024 example replication

Goal: ship runnable versions of the principal examples from our MPCGPU paper,
including the iiwa five-goal pick-and-place task, with the updated solver stack.
The prepared figure-eight timing batch is the first performance checkpoint.
Paper-task replication is the next milestone, with its own configurations and
validation. Timing continues to wait for an explicitly assigned quiet window.

## 1. Establish the paper configurations

- [ ] Inventory the original example entry points, trajectories and figure
  scripts. Map each principal ICRA experiment to a maintained runnable example.
- [ ] Record model/EE frame, goals and switching rule, initial conditions,
  timestep/integrator, costs/constraints, regularization, warm start, SQP/PCG
  limits and tolerances, and simulated control-update policy.
- [ ] Recover the published trial-generation procedure and seeds where
  available. Document missing inputs instead of silently substituting fig8 data.
- [ ] Keep corrected dynamics and safety fixes. Describe differences from the
  original implementation explicitly; retain an isolated historical baseline
  if needed for attribution rather than restoring known bugs.

## 2. Restore the task examples and check correctness

- [ ] Implement or repair the five-goal pick-and-place example for PCG and QDLDL
  using shared reference loading and solver configuration.
- [ ] Validate finite states/controls, goal completion and switching, tracking
  error, solver exits and applicable limits. Run deterministic single-trial
  checks before the complete trial set; correctness runs disable timers.
- [ ] Add a documented one-command example and machine-readable task-quality
  outputs. Test from a fresh clone using only declared inputs and dependencies.
- [ ] Produce a trajectory visualization/animation from the run. The author's
  original high-resolution figures or animation can also illustrate the site.

## 3. Collect comparable results in a separate quiet-window batch

- [ ] First review the prepared modernized-runtime batch for regressions.
- [ ] Prepare immutable paper-task binaries and a manifest before measurement.
  The paper's protocol uses 100 ten-second trials; verify the precise settings
  against the original experiment before collecting the replication set.
- [ ] Collect the relevant horizon, linear-system and closed-loop comparisons
  with both backends, preserving raw samples, quality metrics and provenance.
- [ ] Report linear-system time, controller time and task quality separately.
  Compare PCG and QDLDL on the same machine/configuration.
- [ ] Explain the current RTX 5090 / Core Ultra 9 285K / CUDA 13.2 setup versus
  the paper's RTX 4090 / i9-12900K / CUDA 12.1. Exact old latency values are not
  the replication target on different hardware.

## 4. Update examples, docs and website

- [ ] Summarize which paper examples replicate, which differ, and why. Resolve
  unexplained task-quality regressions before marking an example complete.
- [ ] Add quickstart commands and protocol manifests for the working examples.
- [ ] Add current results alongside the original ICRA results with clear
  hardware/methodology labels. Refresh plots only from matching experiment data.
- [ ] Keep the paper citation visible and apply the project MIT license to our
  code, documentation, website and figures.

Exit criterion: the principal paper tasks are runnable and checked on the current
stack; qualitative behavior and measured comparisons are explained and
reproducible. The current figure-eight suite alone does not close this milestone.
Budget the replication batch after its workloads are prepared; it is not hidden
inside the existing 30–45-minute runtime reservation.
