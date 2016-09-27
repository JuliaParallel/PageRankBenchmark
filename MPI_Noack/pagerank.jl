module PageRankMPI_Noack

include("../DArray/kronGraph500NoPerm.jl") # for the Kronecker generator. FixMe! Use the fancy version in DArray
include("../common/common.jl")
include("samplesort.jl")
using .Common
using MPI

#
# Global state
#

function setup()
    MPI.Init()
    return MPI.COMM_WORLD
end
function teardown(state)
    MPI.Finalize()
end


#
# Pipeline
#

function kernel0(dir, scl, EdgesPerVertex, niter, state=nothing)

    # MPI information
    nworkers = MPI.Comm_size(state)
    id       = MPI.Comm_rank(state)

    files = [joinpath(dir, "$i.tsv") for i in 1:nworkers]

    n = 2^scl # Total number of vertices
    m = EdgesPerVertex * n # Total number of edges

    # Make sure that we distribute the workload over the workers.
    EdgesPerWorker = m ÷ nworkers
    surplus = m % nworkers

    @assert length(files) == nworkers

    nEdges = EdgesPerWorker
    nEdges += ifelse(id + 1 == nworkers, surplus, 0)
    # FixMe! Decouple number of files from number of workers
    kronGraph500(files[id + 1], scl, nEdges)

    return dir, files, n, niter
end

function kernel1(dir, files, n, niter, state=nothing)

    # MPI information
    nworkers = MPI.Comm_size(state)
    id       = MPI.Comm_rank(state)

    # Shuffle the files so that we minimize cache effect
    # TODO ideally we would like to make sure that no processor reads in
    # its own file.
    perm = randperm(nworkers)
    MPI.Bcast!(perm, 0, state)
    files = files[perm]

    info("Read data")
    # FixMe! Decouple number of files from number of procs
    @time edges = read_edges(files[id + 1])

    info("Sort edges")
    @time sorted_edges = sortSample(edges, state, by = first)

    info("Write edges")
    # FixMe! Decouple number of files from number of procs
    files = [joinpath(dir, "chunk_$i.tsv") for i in 1:nworkers]
    @time write_edges(files[id + 1], sorted_edges)

    return files, n, niter
end

function kernel2(files, n, niter, state=nothing)

    # MPI information
    nworkers = MPI.Comm_size(state)
    id       = MPI.Comm_rank(state)

    info("Read data and turn it into a sparse matrix")

    # FixMe! Decouple number of files from number of procs
    edges = read_edges(files[id + 1])
    is, js, vs = Vector{Int}(length(edges)), Vector{Int}(length(edges)), ones(Float64, length(edges))

    # get the beginning first index of all chunks
    is1, js1 = edges[1]
    cuts = MPI.Allgather(is1, state)

    # Compute the size of chunks
    if id == nworkers - 1 # last worker
        n_local = n - is1 + 1
    else
        n_local = cuts[id + 2] - is1
    end

    # Make all local arrays have first observation at row 1
    for i in 1:length(edges)
        isi, jsi = edges[i]
        is[i], js[i] = isi - is1 + 1, jsi
    end
    # We store each local stripe of the distributed sparse matrix in an n x x sparse matrix
    # i.e. most of each local matrix will be structurally zero but it doesn't matter
    adj_matrix_local = sparse(is, js, vs, n_local, n)

    info("Pruning and scaling")
    @time begin

        din = sum(adj_matrix_local, 1)                  # Compute in degree
        din = MPI.Allreduce!(din, similar(din), +, state)
        max_din = maximum(din)

        SparseArrays.fkeep!(adj_matrix_local, (i,j,v) -> begin
            return !(din[j] == 1 || din[j] == max_din)
        end)

        dout = sum(adj_matrix_local, 2)                 # Compute out degree

        # Construct weight diagonal
        InvD = vec(map(t -> ifelse(t == 0, 0.0, inv(t)), dout))

        # Scale the matrix
        scale!(InvD, adj_matrix_local)              # Apply weight matrix.
    end

    return adj_matrix_local, niter
end

function kernel3(adj, niter, state = nothing)

    # MPI information
    nworkers = MPI.Comm_size(state)
    id       = MPI.Comm_rank(state)

    c = 0.85 # Should this be an argument
    info("Run PageRank")
    @time begin
        n_local, n = size(adj)
        n_locals = MPI.Allgather(Int32(n_local), state)

        x = rand(n_local)

        # comp
        nrm_x_local = sumabs(x)
        nrm_x = MPI.Allreduce(nrm_x_local, +, state)
        scale!(x, inv(nrm_x))
        a = (1 - c)/n

        # Run first iteration outside loop to get the right chunks size for free
        scale!(x, c)
        y_local = adj'x + a
        y = MPI.Reduce(y_local, +, nworkers - 1, state) # Should implement Reduce! in MPI.jl

        for i in 2:niter
            x = MPI.Scatterv(y, n_locals, nworkers - 1, state) # Should implement Scatterv! in MPI.jl
            fill!(y_local, 1)
            nrm_x_local = sumabs(x)
            nrm_x       = MPI.Allreduce(nrm_x_local, +, state)
            Ac_mul_B!(c, adj, x, a*nrm_x, y_local)
            y = MPI.Reduce(y_local, +, nworkers - 1, state) # Should implement Reduce! in MPI.jl
        end

        if id == nworkers - 1
            scale!(y, inv(norm(y, 1)))
        end
    end
    if id == nworkers - 1
        println("Sum of PageRank $(norm(y, 1))")
    end
    return y
end

end