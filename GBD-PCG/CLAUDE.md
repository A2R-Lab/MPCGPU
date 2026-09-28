# GBD-PCG orientation

Follow ../CLAUDE.md and README.md. GBD-PCG is maintained in this tree, sharing
MPCGPU's top-level GLASS pin and signed correctness suite. The standalone remote
has not yet been archived. Do not add a GATO dependency or nested GLASS copy.

Preserve cooperative grid-wide iteration; defer block algebra to glass::block::.
GLASS's single-block PCG is not a substitute for grid-wide decomposition.
Use test_api.cu and test_pcg_spd.cu, not indefinite legacy examples or ambient
/tmp dumps. Small scratch layouts and N=1 halos have explicit regression gates.
The recurrence eta norm is not the true residual; SPD is a caller precondition.
