
#-------------------------------------
# Inputs
#-------------------------------------

# Initial capital
start.capital = 55523

# Investment
annual.mean.return = 10.2 / 100
annual.ret.std.dev = 10 / 100

# Inflation
annual.inflation = 1.5 / 100
annual.inf.std.dev = 3.5 / 100

# Withdrawals
monthly.withdrawals =0

# Number of observations (in Years)
n.obs = 5

# Number of simulations
n.sim = 1000

#-------------------------------------
# Simulation
#-------------------------------------

# number of months to simulate
n.obs = 12 * n.obs


# monthly Investment and Inflation assumptions
monthly.mean.return = annual.mean.return / 12
monthly.ret.std.dev = annual.ret.std.dev / sqrt(12)

monthly.inflation = annual.inflation / 12
monthly.inf.std.dev = annual.inf.std.dev / sqrt(12)


# simulate Returns
monthly.invest.returns = matrix(0, n.obs, n.sim)
monthly.inflation.returns = matrix(0, n.obs, n.sim)

monthly.invest.returns[] = rnorm(n.obs * n.sim, mean = monthly.mean.return, sd = monthly.ret.std.dev)
monthly.inflation.returns[] = rnorm(n.obs * n.sim, mean = monthly.inflation, sd = monthly.inf.std.dev)

# simulate Withdrawals
nav = matrix(start.capital, n.obs + 1, n.sim)
for (j in 1:n.obs) {
  nav[j + 1, ] = nav[j, ] * (1 + monthly.invest.returns[j, ] - monthly.inflation.returns[j, ]) - monthly.withdrawals
}

# once nav is below 0 => run out of money
nav[ nav < 0 ] = NA

# convert to millions
nav = nav / 1000

#-------------------------------------
# Plots
#-------------------------------------
layout(matrix(c(1,2,1,3),2,2))

# plot all scenarios    
matplot(nav, type = 'l', las = 1, xlab = 'Miesi¹ce', ylab = 'Tysi¹ce', 
        main = 'Prognozowana Wartoœæ zainwestowanego kapita³u')

# plot % of scenarios that are still paying
p.alive = 1 - rowSums(is.na(nav)) / n.sim

plot(100 * p.alive, las = 1, xlab = 'Miesi¹ce', ylab = 'Procent Wyp³acajacych', 
     main = 'Procent Wyp³acaj¹cych Scenariuszy', ylim=c(0,100))
grid()  

# plot distribution of final wealth
final.nav = nav[n.obs + 1, ]
final.nav = final.nav[!is.na(final.nav)]

plot(density(final.nav, from=0, to=max(final.nav)), las = 1, xlab = 'Ostateczny Kapita³', 
     main = paste('Rozk³ad Ostatecznego Kapita³u,', 100 * p.alive[n.obs + 1], '% nadal wyp³aca'))
grid()  