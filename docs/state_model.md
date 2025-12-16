# Poisson → GMC-LSS: minimal state-count sketch

If you want a coarse-grained **state machine** that can move an initially Poisson-like point process into a GMC-mediated LSS filament with coherent rotation + spin alignment, you need *at least* these distinct macrostates:

1. **P0: Poisson / Isotropic**
   - No preferred axis; nearest-neighbor stats ~ Poisson.

2. **F1: Filament density anisotropy**
   - A preferred axis emerges (spine); 1D compression + 2D depletion.

3. **V2: Coherent vorticity band**
   - Velocity field breaks mirror symmetry across the spine; approaching/receding sides become statistically separable.

4. **S3: Spin–filament coupling**
   - Galaxy spins show measurable non-random alignment with filament direction.

Those 4 are the *core* states. In practice you usually also need:

5. **T4: Transition / mixed state**
   - Partial filament + intermittent rotation patches; the system has memory but is not locked.

6. **D5: Disrupted / decohered**
   - Mergers, sampling selection, or external tides erase vorticity/alignment while leaving some density anisotropy.

So: **4 states is the theoretical minimum**, but **5–6 states** is the realistic minimum if you want transitions + failure modes.

This repo exposes tests that correspond to crossing those state boundaries:
- P0→F1: spine fit quality (anisotropy)
- F1→V2: rotation curve fit + Td
- V2→S3: |cosψ| peak > random
