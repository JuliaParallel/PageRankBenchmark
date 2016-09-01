# PageRankBenchmark

This repository is a fork which contains a variety of parallelized Julia implementations
of the [PageRank Pipeline Benchmark](http://arxiv.org/abs/1603.01876).

The `other` folder contains the implementations in other languages from the original
repository (including the reference Julia implementation). An updated version of the
reference Julia implementation can be found in the `reference` folder.

### Dependencies

To run the driver program `run.jl` you will need to install a few packages:
```julia
Pkg.add("ArgParse")
Pkg.add("BenchmarkTools")
Pkg.add("BufferedStreams")
Pkg.add("DistributedArrays") # for DArray benchmark
Pkg.add("https://github.com/JuliaParallel/Dagger.jl.git")            # for Dagger benchmark
```
