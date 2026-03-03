function toydata()

    rng = MersenneTwister(1234)
    x = rand(rng, 1000)*10
    y = sin.(x) + 0.1*randn(rng, length(x))

    return y, x

end