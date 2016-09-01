module PageRankDagger

using Dagger
import Dagger: parts, DomainSplit, BlockedDomains, ComputedArray, Cat, parttype

include("../common/common.jl")

function setup()
    Dict(:nparts=>nworkers(), :ctx=>Context())
end

function teardown(state)
    nothing
end

include("kernel0.jl")

function kernel0(path, scl, avg_connections, state)
    X = generate_par(scl, avg_connections, state[:nparts])
    (2^scl, write_files(X, state[:nparts], path, state[:ctx])...)
end

function kernel1(n, dir, files, dmn, state=setup())
    info("Read data")
    @time edges = read_array(files, dmn, state)
    info("Sort")
    @time edges = compute(state[:ctx], sort(edges))

   info("Write edges")
   @time (n, write_files(edges, parts(edges.result) |> length,
               dir, state[:ctx])...)
end

function read_array(files, dmn, state)
    p = BlockPartition(1)
    ord = shuffle!([i for i=1:length(files)])
    
    # shuffle files
    clen = cumsum(map(length, parts(dmn)[ord]))
    dmn1 = DomainSplit(Dagger.head(dmn), BlockedDomains((1,), (clen,)))
    files = files[ord]

    fs = compute(state[:ctx], Distribute(p, files))
    tmp = compute(state[:ctx], mappart(fs) do f
        Common.read_edges(f[1])
    end)
    edges = ComputedArray(
       Cat(Vector{Tuple{Int,Int}}, dmn1, parts(tmp.result))
    )
end

function create_adj_matrix(state, n, ij)
    tmp = compute(state[:ctx], mappart(ij) do p
        p[1][1], p[end][1]
    end)

    minmaxes = map(gather, parts(tmp.result))
    mins = map(first, minmaxes)
    mins[1] = 1
    maxs = map(last, minmaxes)
    maxs[end] = n
    nrows = maxs .- mins .+ 1

    p = BlockPartition(1)
    tmp2=mappart(Distribute(p, mins), Distribute(p, nrows), ij) do min_i_, m_, ij
        # modified version of Valentin's code from DArray implementation
        min_i = min_i_[1]::Int64
        m = m_[1]::Int64
        I = Vector{Int64}(length(ij))
        J = Vector{Int64}(length(ij))
        V = Vector{Float64}(length(ij))


        for idx in 1:length(ij)
           i, j = ij[idx]
           i = i - (min_i - 1)
           @inbounds I[idx] = i
           @inbounds J[idx] = j
           @inbounds V[idx] = 1.0
        end
        SparseMatrixCSC(m,n,I, J, V)
    end
    ps = reshape(parts(compute(state[:ctx], tmp2).result), (length(nrows), 1))
    dmn = DomainSplit(DenseDomain(1:n, 1:n), BlockedDomains((1,1), (cumsum(nrows), [n])))
    ComputedArray(Cat(parttype(ps[1]), dmn, ps))
end

function kernel2(n, dir, files, dmn, state=setup())
   info("Read data and turn it into a sparse matrix")
   @time begin
      ij = read_array(files, dmn, state)
      adj_matrix = create_adj_matrix(state, n, ij)
   end

   @assert size(adj_matrix) == (n, n)
   info("Pruning and scaling")
   @time begin
      din = gather(sum(adj_matrix, 1))          # Compute in degree
      dout = gather(sum(adj_matrix, 2))
      zero_nodes = find((din .== maximum(din)) | (din .== 1.0))
      adj_matrix = compute(state[:ctx], setindex(adj_matrix, 0.0, :, zero_nodes))
      dout = gather(sum(adj_matrix, 2))
      nz = find(dout)
      diagonal_wt = zeros(size(adj_matrix, 1))
      diagonal_wt[nz] = 1./dout[nz]
      compute(state[:ctx], Diagonal(diagonal_wt) * adj_matrix)
   end
end


end
