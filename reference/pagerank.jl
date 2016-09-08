module PageRankReference

########################################################
# PageRank Pipeline Benchmark
# Architect: Dr. Jeremy Kepner (kepner@ll.mit.edu)
# Julia Translation: Dr. Chansup Byun (cbyun@ll.mit.edu)
# MIT
########################################################
# (c) <2015> Massachusetts Institute of Technology
########################################################

using Compat

include("KronGraph500NoPerm.jl")
include("StrFileWrite.jl")
include("StrFileRead.jl")


#
# Global state
#

const Nfile = 4
const Niter = 20                                      # Number of PageRank iterations.
const damping = 0.15                                        # PageRank damping factor.

setup() = return nothing
teardown(state) = return nothing


#
# Pipeline
#

########################################################
# Kernel 0: Generate a Graph500 Kronecker graph and save to data files.
########################################################
function kernel0(dir, SCALE, EdgesPerVertex, state=nothing)
  Nmax = 2.^SCALE;                           # Max vertex ID.
  M = EdgesPerVertex .* Nmax;                # Total number of edges.
  myFiles = collect(1:Nfile).';              # Set list of files.
  #
  # Julia parallel version
  # Figure it out later: how to distribute the load
  # myFiles = global_ind(zeros(Nfile,1,map([Np 1],{},0:Np-1)));   # PARALLEL.


  for i in myFiles
    fname = joinpath(dir, "kernel0_$i.tsv")
    srand(i);                                                # Set random seed to be unique for this file.
    ut, vt = KronGraph500NoPerm(SCALE,EdgesPerVertex./Nfile);  # Generate data.

    writeuv(fname, ut, vt)
  end

  return myFiles, dir, Nmax
end

########################################################
# Kernel 1: Read data, sort data, and save to files.
########################################################
function kernel1(myFiles, dir, Nmax, state=nothing)
  # Read in all the files into one array.
  for i in myFiles
    fname = joinpath(dir, "kernel0_$i.tsv")
    ut,vt = StrFileRead(fname);
    # Concatenate to u,v
    if i == 1
       u = ut; v = vt;
    else
       append!(u, ut)
       append!(v, vt)
    end
  end

  sortIndex = sortperm(u)                      # Sort starting vertices.
  u = u[sortIndex]                                  # Get starting vertices.
  v = v[sortIndex]                                  # Get ending vertices.

  # Write all the data to files.
  j = 1;                                                         # Initialize file counter.
  c = size(u,1)/length(myFiles)        # Compute first edge of file.
  for i in myFiles
    jEdgeStart = round(Int, (j-1)*c+1)# Compute first edge of file.
    jEdgeEnd = round(Int, j*c)          # Compute last edge of file.
    uu = view(u,jEdgeStart:jEdgeEnd)                                 # Select start vertices.
    vv = view(v,jEdgeStart:jEdgeEnd)                                 # Select end vertices.
    fname = joinpath(dir, "kernel1_$i.tsv")

    writeuv(fname, uu, vv)

    j = j + 1                                                   # Increment file counter.
  end

  return myFiles, dir, u, v, Nmax
end

########################################################
# Kernel 2: Read data, filter data.
########################################################
function kernel2(myFiles, dir, u, v, Nmax, state=nothing)
  # Read in all the files into one array.
  for i in myFiles
    fname = joinpath(dir, "kernel1_$i.tsv")
    ut,vt = StrFileRead(fname);
    if i == 1
       u = ut; v = vt;                             # Initialize starting and ending vertices.
    else
       append!(u, ut)
       append!(v, vt)
       # Get the rest of starting and ending vertices.
    end
  end

  # Construct adjacency matrix.
  A = sparse(vec(u),vec(v),1.0,Nmax,Nmax)      # Create adjacency matrix.

  # Filter and weight the adjacency matrix.
  din = sum(A,1)                               # Compute in degree.
  max_din = maximum(din)
  A[find(x->x==max_din || x ==1, din)]=0               # Eliminate the super-node.
  dout = sum(A,2)                              # Compute the out degree.
  is = find(dout)                               # Find vertices with outgoing edges (dout > 0).
  DoutInvD = zeros(Nmax)        # Create diagonal weight matrix.
  DoutInvD[is] = 1./dout[is]
  scale!(DoutInvD, A)           # Apply weight matrix.

  return Nmax, A
end

########################################################
# Kernel 3: Compute PageRank.
########################################################
function kernel3(Nmax, A, state=nothing)
  r = rand(1,Nmax);                     # Generate a random starting rank.
  r = r ./ norm(r,1);                   # Normalize
  a = (1-damping) ./ Nmax;                    # Create damping vector

  for i=1:Niter
      s = r * A
      scale!(s, damping)
      r = s .+ (a * sum(r,2));                # Compute PageRank.
  end

  r = r./norm(r,1);

  return nothing
end

end
