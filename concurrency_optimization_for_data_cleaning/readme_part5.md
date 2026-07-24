# v1 Retained Component: Adaptive Concurrency

This directory is retained from G-Frame v1 as the local serving and task-limit
component. In G-Frame v2, it corresponds at a coarse level to the manuscript
flow's runtime observation and bounded concurrency-control area.

The v2-owned workflow integration is documented in the repository root
[`readme.md`](../readme.md). This retained component remains useful when an
operator chooses to connect a local serving process and task queue; endpoint
settings, limits, inputs, and output locations are operator-supplied.

This note records component continuity only. The v2 source and configuration
files define the active execution contract.
