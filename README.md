# MooncakeGPExperiments

This is a series of numerical experiments that investigates good practices of using Mooncake.jl via the DifferentiationInterface.jl and in conjuction to Optim.jl when implementing Gaussian processes.

We present a series of codes that incrementally address issues regarding memory allocations. The incremental addressing aims to reveal pitfalls that novices (and perhaps more experienced programmers) are often unaware of.