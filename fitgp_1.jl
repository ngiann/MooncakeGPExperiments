function fitgp_1(t, x; iterations = 10, JITTER = 1e-6)

    # t are the target outputs (here one-dimensional)
    # x are the inputs (here one-dimensional)

    # Initialise all hyperparameters to zero
    logℓ, logα, logσ = 0.0, 0.0, 0.0
   
    rbf(x, y, logℓ, logα) = exp(-abs2(x - y)*exp(logℓ))*exp(logα)

    function calculatecovariance!(K, logℓ, logα)
        # Can be done better, but the point is to understand where
        # problems may arise with allocations and performance
        # when writing own code

        local N = length(x)

        for i in 1:N
            for j in 1:N
                K[i, j] = rbf(x[i], x[j], logℓ, logα)
            end
        end

    end

    function unpack_parameters(params)
        local logℓ, logα, logσ = params
        return logℓ, logα, logσ
    end

    # calculate the covariance matrix for the passed hyperparameters, add noise variance and jitter
    function calculateK(x, logℓ, logα, logσ)
        local K = zeros(length(x), length(x))
        calculatecovariance!(K, logℓ, logα)
        K += exp(logσ) * I # add noise variance to the diagonal
        K += JITTER * I * exp(logα) # add jitter for numerical stability
        return K
    end

    function marginal_loglikelihood_MvNormal(logℓ, logα, logσ)

        # calculate the covariance matrix for the passed hyperparameters
        local K = calculateK(x, logℓ, logα, logσ)

        # return log marginal likelihood of a Gaussian process for Gaussian likelihood
        return logpdf(MvNormal(zeros(length(x)), K), t)

    end


    function marginal_loglikelihood_explicit(logℓ, logα, logσ)

        # calculate the covariance matrix for the passed hyperparameters
        local K = calculateK(x, logℓ, logα, logσ)

        # return log marginal likelihood of a Gaussian process for Gaussian likelihood
        local L = cholesky(Symmetric(K)).L
        return -0.5 * sum(abs2, L \ t) - sum(log, diag(L)) - 0.5*length(x)*log(2π)

    end

    # define negative log-likelihood for optimization
    nll_MvNormal(params) = -marginal_loglikelihood_MvNormal(unpack_parameters(params)...)
    nll_explicit(params) = -marginal_loglikelihood_explicit(unpack_parameters(params)...)

    # Sanity check:
    # show that the two implementations of the marginal log-likelihood 
    # differ only very slightly for the same hyperparameters.
    let
        initparam = randn(3)
        @info "evaluate nll_MvNormal with random parameters" nll_MvNormal(initparam)
        @info "evaluate nll_explicit with random parameters" nll_explicit(initparam)
    end

    # Optimise hyperparameters using LBFGS
    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    # optimise using the MvNormal implementation
    @info "optimising with nll_MvNormal"
    res1 = optimize(nll_MvNormal, [logℓ, logα, logσ], LBFGS(), opt, autodiff=Mooncake.AutoMooncake())
 
    # display condition number of the covariance matrix
    @info "cond(K) after optimising with nll_MvNormal"
    display(cond(calculateK(x, unpack_parameters(res1.minimizer)...)))
    
    # optimise using the explicit implementation
    @info "optimising with nll_explicit"
    res2 = optimize(nll_explicit, [logℓ, logα, logσ], LBFGS(), opt, autodiff=Mooncake.AutoMooncake())

    # display condition number of the covariance matrix
    @info "cond(K) after optimising with nll_explicit"
    display(cond(calculateK(x, unpack_parameters(res2.minimizer)...)))
end





