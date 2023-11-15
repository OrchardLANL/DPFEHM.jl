include("weeds.jl")
x_test = randn(num_eigenvectors) #the input to the function should follow an N(0, I) distribution -- the eigenvectors and eigenvalues are embedded inside f
@show f(x_test) #compute the pressure at the critical location
@show extrema(grad_f(x_test)) #compute the gradient of the pressure at the critical location with respect to the eigenvector coefficients (then only show the largest/smallest values so it doesn't print a huge array)

#Quick demo of how you could do Monte Carlo to get the distribution of pressures at the critical location -- hopefully the method Georg described can help us understand the tail distribution better
numruns = 10 ^ 3
fs = Float64[]
for i = 1:numruns
    x_test = randn(num_eigenvectors)
    push!(fs, f(x_test))
end
@show sum(fs) / length(fs)
@show extrema(fs)
fig, ax = PyPlot.subplots()
ax.hist(fs, bins=30)
ax.set(xlabel="Pressure [MPa] at critical location", ylabel="Count (out of $(numruns))")
fig.show()
fig.savefig("distribution_of_pressure.pdf")
PyPlot.close(fig)
