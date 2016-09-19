module Kron

# A square matrix with a single unit element
immutable OneNZ <: DenseMatrix{Int}
    sz::Int
    i::Int
    j::Int
end

Base.size(A::OneNZ) = (A.sz, A.sz)
Base.size(A::OneNZ, i) = i < 1 ? error("") :
                    i < 3 ? A.sz : 1

Base.getindex(A::OneNZ, i::Integer, j::Integer) = Int(i == A.i && j == A.j)

Base.kron(A::OneNZ, B::OneNZ) = OneNZ(A.sz * B.sz, (A.i - 1) * B.sz + B.i, (A.j - 1) * B.sz + B.j)


const ⊗ = kron

# RNG for random 2x2 OneNZ matrices
immutable OneNZRNG
    p01::Float64
    p01p10::Float64
    p01p10p11::Float64
    OneNZRNG(p01,p10,p11) = new(p01, p01 + p10, p01 + p10 + p11)
end

@inline function Base.rand(X::OneNZRNG)
    u = rand(Float64)
    ifelse(u < X.p01, OneNZ(2, 1, 2),
    ifelse(u < X.p01p10, OneNZ(2, 2, 1),
    ifelse(u < X.p01p10p11, OneNZ(2, 2, 2),
        OneNZ(2, 1, 1))))
end

# Define a sparse matrix type to store the accumulation of Kronecker terms
immutable COOSquare <: AbstractSparseMatrix{Int}
    keys::Vector{Tuple{Int,Int}}
    sz::Int
end

Base.size(A::COOSquare) = (A.sz, A.sz)
Base.size(A::COOSquare, i) = i < 1 ? error("") :
                    i < 3 ? A.sz : 1

Base.getindex(A::COOSquare, i::Integer, j::Integer) = mapreduce(a -> i == a[1] && j == a[2], +, A.keys)

# Addition of OneNZs create a COOSquare with two key pairs
import Base: +
function (+)(A::OneNZ, B::OneNZ)
    if A.sz != B.sz
        error("")
    end
    return COOSquare(Tuple{Int,Int}[(A.i, A.j),(B.i, B.j)], A.sz)
end
# In place addition of a COOSquare and OneNZ pushes to the COOSquare
function add!(A::COOSquare, B::OneNZ)
#     if A.sz != B.sz
#         error("")
#     end
    push!(A.keys, (B.i, B.j))
    return A
end
function add!(A::COOSquare, B::COOSquare)
#     if A.sz != B.sz
#         error("")
#     end
    append!(A.keys, B.keys)
    return A
end

# Make sure that the reduction initialized correctly for our new type
Base.r_promote(::typeof(add!), x::OneNZ) = COOSquare([(x.i, x.j)], x.sz)

# Create RNG for Kronecker graph
immutable Kronecker
    scl::Int
    epv::Int
    onenz::OneNZRNG
end
# Add default constructor with the probabilities from Graph500
Kronecker(scl, epv) = Kronecker(scl, epv, OneNZRNG(0.57, 0.19, 0.19))

function Base.rand(x::Kronecker)
    n = 2^x.scl
    m = x.epv*n

    mapreduce(_ -> mapreduce(_ -> rand(x.onenz), ⊗, 1:x.scl), add!, 1:m)
end

# To get a distributed version just distribute the reduction range
using DistributedArrays
function DistributedArrays.drand(x::Kronecker)
    n = 2^x.scl
    m = x.epv*n

    mapreduce(_ -> mapreduce(_ -> rand(rand(x.onenz)), ⊗, 1:x.scl), add!, distribute(1:m))
end

end
