module PageRankDArray

using DistributedArrays
import DistributedArrays: map_localparts!, map_localparts

include("kronGraph500NoPerm.jl")
include("dsparse.jl") # Provides create_adj_matrix
include("../common/common.jl")
using .Common


#
# Global state
#

setup() = return nothing
teardown(state) = return nothing


#
# Pipeline
#

function kernel0(dir, scl, EdgesPerVertex, niter, state=nothing)
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
   return dir, files, n, niter
end

function kernel1(dir, files, n, niter, state=nothing)
   # Shuffle the files so that we minimize cache effect
   # TODO ideally we would like to make sure that no processor reads in
   # its own file.
   shuffle!(files)

   info("Read data")
   @time edges = DArray(dread(files)) # DArray construction will wait on the futures

   info("Sort edges")
   @time sorted_edges = sort(edges, by = first)
   close(edges)

   info("Write edges")
   files = collect(joinpath(dir, "chunk_$i.tsv") for i in 1:nworkers())
   @time dwrite(files, sorted_edges)

   return files, n, niter
end

function kernel2(files, n, niter, state=nothing)
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
      max_din = maximum(din)
      map_localparts!(adj_matrix) do ladjm
         ldin = Array(fetch(din))
         SparseArrays.fkeep!(ladjm, (i, j, v) -> begin
            return !(ldin[j] == 1 || ldin[j] == max_din) # Drop the supernode or any leafnode
         end)
      end

      dout = sum(adj_matrix, 2)                 # Compute out degree

      # Construct weight diagonal
      InvD = map(t -> ifelse(t==0, 0.0, inv(t)), dout)
      # This is stupid but I don't think we have an easy way to convert to Vector
      InvD = DArray(DistributedArrays.next_did(), I -> vec(localpart(InvD)), (size(InvD,1),),
         vec(procs(InvD)), vec(map(t -> (t[1],), InvD.indexes)), InvD.cuts[1:1])

      adj_matrix_float = DistributedArrays.map_localparts(SparseMatrixCSC{Float64,Int}, adj_matrix)
      scale!(InvD, adj_matrix_float)              # Apply weight matrix.
   end

   return adj_matrix_float, niter
end

function kernel3(Adj, niter, state = nothing)
   c = 0.85 # Should this be an argument
   info("Run PageRank")
   @time begin
      n = size(Adj, 1)
      x = drand(n)
      scale!(x, inv(norm(x, 1)))
      a = (1 - c)/n

      # Run first iteration outside loop to get the right chunks size for free
      scale!(x, c)
      y = Adj'x + a*norm(x, 1)

      for i in 2:niter
         copy!(x, y)
         fill!(y, 1)
         Ac_mul_B!(c, Adj, x, a*norm(x, 1), y)
      end

      scale!(y, inv(norm(y, 1)))
   end
   println("Sum of PageRank $(norm(y, 1))")
   return y
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
