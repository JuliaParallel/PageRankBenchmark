module PageRank
module DArray
using DistributedArrays

include("kronGraph500NoPerm.jl")
include("dsparse.jl") # Provides create_adj_matrix
include("../common/common.jl")
using .Common


#
# Global state
#

typealias BenchmarkState Void

setup() = return BenchmarkState()

teardown(state) = return nothing


#
# Pipeline
#

function kernel0(state, dir, scl, EdgesPerVertex)
   files = collect(joinpath(dir, "$i.tsv") for i in 1:nworkers())

   n = 2^scl # Total number of vertices
   m = EdgesPerVertex * n # Total number of edges

   # Make sure that we distribute the workload over the workers.
   EdgesPerWorker = m ÷ nworkers()
   surplus = m % nworkers()

   @assert length(files) == nworkers()

   lastWorker = maximum(workers())
   @sync begin
      for (id, filename) in zip(workers(), files)
         nEdges = EdgesPerWorker
         nEdges += ifelse(id == lastWorker, surplus, 0)
         @async remotecall_wait(kronGraph500, id, filename, scl, nEdges)
      end
   end
   return dir, files, n
end

function kernel1(state, dir, files, n)
   # Shuffle the files so that we minimize cache effect
   # TODO ideally we would like to make sure that no processor reads in
   # its own file.
   shuffle!(files)

   info("Read data")
   @time edges = DistributedArrays.DArray(dread(files)) # DArray construction will wait on the futures

   info("Sort edges")
   @time sorted_edges = sort(edges, by = first)
   close(edges)

   info("Write edges")
   files = collect(joinpath(dir, "chunk_$i.tsv") for i in 1:nworkers())
   @time dwrite(files, sorted_edges)

   return files, n
end

function kernel2(state, files, n)
   info("Read data and turn it into a sparse matrix")
   @time begin
      rrefs = dread(files)
      adj_matrix = create_adj_matrix(rrefs, n)
      rrefs = nothing
   end

   @assert size(adj_matrix) == (n, n)
   info("Pruning and scaling")
   @time begin
      din = sum(adj_matrix, 1)                  # Compute in degree
      adj_matrix[find(din == maximum(din))]=0   # Eliminate the super-node.
      adj_matrix[find(din == 1)]=0              # Eliminate the leaf-node.
      # dout = sum(adj_matrix, 2)                 # Compute out degree
      # is = find(dout)                           # Find vertices with outgoing edges (dout > 0).
      # DoutInvD = zeros(size(adj_matrix, 1))     # Create diagonal weight matrix.
      # DoutInvD[is] = 1./dout[is]
      # scale!(DoutInvD, adj_matrix)              # Apply weight matrix.
   end

   return adj_matrix
end


#
# Auxiliary
#

function dread(files)
   map(zip(files, workers())) do iter
      filename, id = iter
      remotecall(read_edges, id, filename)
   end
end

function dwrite(files, edges)
   @sync for (id, filename) in zip(workers(), files)
      @async remotecall_wait(id, filename) do filename
         write_edges(filename, localpart(edges))
      end
   end
end

# Two helper functions to make sort work on tuples
Base.typemin{T}(::Type{Tuple{T,T}}) = (typemin(T),typemin(T))
Base.typemax{T}(::Type{Tuple{T,T}}) = (typemax(T),typemax(T))

end
end
