# ICRA 2024 pick-and-place circuit

The five-goal circuit from Figure 3 of our ICRA 2024 paper. A simulated iiwa leaves its start
pose, visits four goals and returns to the start in 10.4 seconds (666 rows at 1/64 s).

| File | Contents |
| --- | --- |
| `pick_place_traj.csv` | Joint reference `[q(7) qd(7) u(7)]`, verbatim from the paper code's `examples/trajfiles/0_0_traj.csv` (commits 077252e and c556b19) |
| `pick_place_eepos.traj` | End-effector targets `[x y z 0 0 0]`, recomputed from the joint reference with the current model |
| `pick_place_eepos_2024.traj` | The paper code's original targets, kept for provenance only |

The original targets used the link-7 origin as the end effector. The current model uses the
named flange frame, which moves every target by 4.0 cm. Regenerate the tracked targets with:

```bash
python tools/build.py ee-from-joints --output build/correctness/ee_from_joints.exe
build/correctness/ee_from_joints.exe examples/icra/pick_place_traj.csv examples/icra/pick_place_eepos.traj
```

Run the task and check it from the repository root:

```bash
python tools/build.py icra-pcg --knots 64      # or icra-qdldl; horizons 32 to 512
LD_LIBRARY_PATH=qdldl/build/out build/correctness/icra-pcg-14x64-float.exe --out tmp/icra/pcg-N64
python tools/icra_report.py tmp/icra/pcg-N64
```

The run writes `summary.json`, `samples.csv` and `updates.csv`. The report adds `report.json`
and `trajectory.svg`. The protocol, its sources and every difference from the 2024 code are in
[docs/icra-replication.md](../../docs/icra-replication.md).
