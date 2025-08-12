# MLPBasis

[![Build Status](https://github.com/ngiann/MLPBasis.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ngiann/MLPBasis.jl/actions/workflows/CI.yml?query=branch%3Amain)


```
Z, srest, erest, lrest, s, e, l = readspectra();

L = Matrix(reduce(hcat, [l/(1+z) for z in Z])')

idx = findall(isinf.(srest));

erest[idx].=100;

srest[idx].=0.0;

erest = Matrix(map(Float64,erest));

srest = Matrix(map(Float64,srest));

idx = findall(iszero.(Z) .== 0)
srest = srest[:,idx];
erest = erest[:,idx];
```