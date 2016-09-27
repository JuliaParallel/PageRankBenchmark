using MPI

function sortSample(a, ct = MPI.COMM_WORLD; by = identity)

    id = MPI.Comm_rank(ct)
    p  = MPI.Comm_size(ct)

    as = sort(a; by = by)

    n = length(a)

    splitters_local = [as[round(Int, (n + 1)/p*i)] for i = 1:(p - 1)]

    splitters_all = MPI.Gather(splitters_local, 0, ct)

    splitters_final = similar(a, p - 1)

    if id == 0
        sort!(splitters_all; by = by)
        for i = 1:p - 1
            splitters_final[i] = splitters_all[round(Int, (length(splitters_all) + 1)/p*i)]
        end
    end

    MPI.Bcast!(splitters_final, 0, ct)

    offsets_local = Cint[0]
    i = j = 1
    while j <= length(splitters_final)
        if by(as[i]) >= by(splitters_final[j])
            push!(offsets_local, i - 1)
            j += 1
        end
        i += 1
    end
    push!(offsets_local, n)
    scounts = diff(offsets_local)

    scounts_global = MPI.Allgather(scounts, ct)

    rcounts = zeros(Cint, length(scounts))
    for i = 0:p - 1
        rcounts[i + 1] += scounts_global[i*length(scounts) + id + 1]
    end

    b = MPI.Alltoallv(as, scounts, rcounts, ct)
    sort!(b; by = by)

    return b
end
