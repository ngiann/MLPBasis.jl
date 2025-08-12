############################################################################
function fitbasis(λ0, y, σ; K = 3, H = 10, η = 0.1, iterations = 1)
############################################################################


    λ = (λ0 .- minimum(λ0))/(maximum(λ0) - minimum(λ0)) 

    λ = (2λ .- 1)*3

    N = size(λ, 1); @assert(size(λ) == size(σ) == size(y))


    net = ForwardNeuralNetworks.ThreeLayerNetwork(in = 1, out = 1, H1 = H, H2=H)

    numw = numweights(net)


    param = [0.01*randn(numw*K);invsoftplus.(ones(K*N))]

    fitbasis(λ0, y, σ, param; K = K, H = H, η = η, iterations = iterations)

end

############################################################################
function fitbasis(λ0, y, σ, param; K = 3, H = 10, η = 0.1, iterations = 1)
############################################################################

    λ = (λ0 .- minimum(λ0))/(maximum(λ0) - minimum(λ0)) 

    λ = (2λ .- 1)

    N = size(λ, 1); @assert(size(λ) == size(σ) == size(y))

    D = size(y, 2)

    h(x) = softplus(x)

    net = ForwardNeuralNetworks.ThreeLayerNetwork(in = 1, out = 1, H1 = H, H2=H)

    numw = numweights(net)


    @printf("There are %d number of spectra of dimension %d\n", N, D)
    @printf("Learn %d basis functions\n\n", K)
    @printf("Optimising %d number of free parameters\n\n", numw*K + K*N)


    #--------------------------------------------
    function unpack(p)
    #--------------------------------------------

        local MARK = 0

        local w = reshape(p[MARK+1:MARK+numw*K], K, numw); MARK += numw*K

        local α = h.(reshape(p[MARK+1:MARK+K*N], N, K)); MARK += K*N

        @assert(MARK == length(p))

        return w, α

    end


    #--------------------------------------------
    function logl(w, α)
    #--------------------------------------------

        local aux = zero(eltype(w))#- 0.5*η*sum(abs2.(w)) - 1e-4*sum(abs2.(α))


        for n in 1:N
            
            pred_n = α[n,1] * h.(net(vec(w[1,:]), reshape(λ[n,:], 1, D)))

            for k in 2:K

                pred_n += α[n,k] * h.(net(vec(w[k,:]), reshape(λ[n,:], 1, D)))

            end

            aux += -0.5*sum( (vec(pred_n) - y[n,:]).^2 ) #./ (σ[n,:]).^2 )
            
        end       

        return aux
        
        

    end


    helper(p) = -logl(unpack(p)...)

    @show helper(param)


    #--------------------------------------------
    # Setup optimisation
    #--------------------------------------------

    # param1 = best_candidate(bboptimize(helper; param,MaxFuncEvals = 100000, Method=:generating_set_search, NumDimensions = length(param),  SearchRange = (-100.0, 100.0)))


    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    result = optimize(helper, param, ConjugateGradient(), opt, autodiff = AutoMooncake())


    #--------------------------------------------
    # Instantiate basis
    #--------------------------------------------

    Basis = let
        
        wopt, αopt = unpack(result.minimizer)

        λtest = reshape(collect(LinRange(minimum(λ), maximum(λ), 1000)), 1, 1000)

        [h.(net(vec(wopt[k,:]), reshape(λtest,1,length(λtest)))) for k in 1:K]

    end

    _, αopt = unpack(result.minimizer)


    return Basis, result.minimizer, αopt
    
end