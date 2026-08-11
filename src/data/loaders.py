"""Loading and preprocessing. EMPTY ON PURPOSE — this project fills it in.

Two things the template does ask of whatever goes here:

1. **Never build a path.** Ask `src.utils.paths`, so one call works on a laptop and on both
   cluster tiers.
2. **`data/raw/` is read-only**, and a cache is valid only once its marker file exists — write
   that marker LAST, so a run killed halfway leaves a cache correctly treated as absent. A
   half-written cache that looks complete is a wrong result nobody investigates.
"""

from __future__ import annotations
