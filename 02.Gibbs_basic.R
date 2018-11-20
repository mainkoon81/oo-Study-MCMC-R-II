### Gibbs Sampling Basic ###

# To implement the Gibbs sampler, we return to our running example where the data are the percent change in total personnel from 
#last year to this year for `n=10` companies. We'll still use a **Normal likelihood**, but now we'll relax the assumption that we 
#know the variance of growth between companies `??^2`, and estimate that variance. Instead of the **t-prior** from earlier, we will
#use the two **conditionally conjugate priors**: Normal for `??` and inverse-gamma for `??2`.

# Simulate from the **full-conditional** we derived in the previous segment:

update_mu = function(n, ybar, sig2, mu_0, sig2_0) {
  sig2_1 = 1.0 / (n / sig2 + 1.0 / sig2_0)
  mu_1 = sig2_1 * (n * ybar / sig2 + mu_0 / sig2_0)
  rnorm(n=1, mean=mu_1, sd=sqrt(sig2_1))
  }

update_sig2 = function(n, y, mu, nu_0, beta_0) {
  nu_1 = nu_0 + n / 2.0
  sumsq = sum( (y - mu)^2 ) # vectorized
  beta_1 = beta_0 + sumsq / 2.0
  out_gamma = rgamma(n=1, shape=nu_1, rate=beta_1) # rate for gamma is shape for inv-gamma
  1.0 / out_gamma # reciprocal of a gamma random variable is distributed inv-gamma
  }


# write a function to perform Gibbs sampling
gibbs = function(y, n_iter, init, prior) {
  ybar = mean(y)
  n = length(y)
  
  ## initialize
  mu_out = numeric(n_iter)
  sig2_out = numeric(n_iter)
  
  mu_now = init$mu
  
  ## Gibbs sampler
  for (i in 1:n_iter) {
    sig2_now = update_sig2(n, y, mu_now, 
                           nu_0 = prior$nu_0, beta_0 = prior$beta_0)
    mu_now = update_mu(n, ybar, sig2_now, 
                       mu_0 = prior$mu_0, sig2_0 = prior$sig2_0)
    
    sig2_out[i] = sig2_now
    mu_out[i] = mu_now
  }
  
  cbind(mu=mu_out, sig2=sig2_out)
}# we're going to have a matrix of two columns with 'n_iter' rows. 


#### [set up the problem] ####--------------------------------------------------
y = c(1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9)
ybar = mean(y)
n = length(y)



## prior ##
prior = list()

prior$mu_0 = 0.0
prior$sig2_0 = 1.0
prior$n_0 = 2.0 # prior effective sample size for sig2
prior$s2_0 = 1.0 # prior point estimate for sig2
prior$nu_0 = prior$n_0 / 2.0 # prior parameter for inverse-gamma
prior$beta_0 = prior$n_0 * prior$s2_0 / 2.0 # prior parameter for inverse-gamma



# Let's just check the histogram of the data
hist(y, freq=FALSE, xlim=c(-1.0, 3.0))
points(y, rep(0,n), pch=1) # individual data points
points(ybar, 0, pch=19) # sample mean


curve(dnorm(x, mean = prior$mu_0, sd = sqrt(prior$sig2_0)), lty=2, add=TRUE) # prior for mu..looks like t(0,1,1)..but it's normal

#-------------------------------------------------------------------------------
# Initialize and RUN the sampler!!!!
set.seed(53)
init = list()
init$mu = 0.0

post = gibbs(y=y, n_iter=1e3, init=init, prior=prior)

post
library("coda")
plot(as.mcmc(post))
summary(as.mcmc(post))
#-------------------------------------------------------------------------------

###############################
### Convergence diagnostics ###
###############################

####trace-plot####
# Has our simulated Markov chain converged to its stationary distribution yet? Unfortunately, 
#this is a difficult question to answer, but we can do several things to investigate.

# Our first visual tool for assessing chains is the trace plot. A trace plot shows the history of a parameter value across
#iterations of the chain. It shows you precisely where the chain has been exploring.

set.seed(61)
post0 = MH(n=n, ybar=ybar, n_iter=10e3, mu_init=0.0, cand_sd=0.9)
coda::traceplot(as.mcmc(post0$mu[-c(1:500)]))

post1 = MH(n=n, ybar=ybar, n_iter=1e3, mu_init=0.0, cand_sd=0.04)
coda::traceplot(as.mcmc(post1$mu[-c(1:500)]))


post2 = MH(n=n, ybar=ybar, n_iter=100e3, mu_init=0.0, cand_sd=0.04)
coda::traceplot(as.mcmc(post2$mu))

#One major difference between the two chains we've looked at is the level of autocorrelation in each. Autocorrelation is a number
#between ???1???1 and +1+1 which measures how linearly dependent the current value of the chain is on past values (called lags). 
#We can see this with an autocorrelation plot:

coda::autocorr.plot(as.mcmc(post0$mu))
coda::autocorr.diag(as.mcmc(post0$mu))

coda::autocorr.plot(as.mcmc(post1$mu))
coda::autocorr.diag(as.mcmc(post1$mu))

str(post0)
str(post1)
str(post2) # contains 100,000 iterations

coda::effectiveSize(as.mcmc(post0$mu)) # effective sample size of ~170
coda::effectiveSize(as.mcmc(post1$mu)) # effective sample size of ~3
coda::effectiveSize(as.mcmc(post2$mu)) # effective sample size of ~350

## thin out the samples until autocorrelation is essentially 0. This will leave you with approximately independent samples. 
#The number of samples remaining is similar to the effective sample size.
coda::autocorr.plot(as.mcmc(post2$mu), lag.max=500)
coda::autocorr.diag(as.mcmc(post2$mu))






thin_interval = 400 # how far apart the iterations are for autocorrelation to be essentially 0.
thin_indx = seq(from=thin_interval, to=length(post2$mu), by=thin_interval)
head(thin_indx)


post2mu_thin = post2$mu[thin_indx]
traceplot(as.mcmc(post2$mu))

traceplot(as.mcmc(post2mu_thin))

coda::autocorr.plot(as.mcmc(post2mu_thin), lag.max=10)

effectiveSize(as.mcmc(post2mu_thin))

length(post2mu_thin)

str(post0) # contains 10,000 iterations

coda::effectiveSize(as.mcmc(post0$mu)) # effective sample size of ~2,500 ??????

?effectiveSize

# The chain from post0 has 10,000 iterations, but an effective sample size of about 2,500. That is, this chain essentially 
#provides the equivalent of 2,500 independent Monte Carlo samples.
#Notice that the chain from post0 has 10 times fewer iterations than for post2, but its Monte Carlo effective sample size is 
#about seven times greater than the longer (more correlated) chain. We would have to run the correlated chain for 700,000+ 
#iterations to get the same amount of information from both chains.
# It is usually a good idea to check the Monte Carlo effective sample size of your chain. If all you seek is a posterior mean 
#estimate, then an effective sample size of a few hundred to a few thousand should be enough. However, if you want to create 
#something like a 95% posterior interval, you may need many thousands of effective samples to produce a reliable estimate of 
#the outer edges of the distribution. The number you need can be quickly calculated using the Raftery and Lewis diagnostic.

raftery.diag(as.mcmc(post0$mu))

raftery.diag(as.mcmc(post0$mu), q=0.005, r=0.001, s=0.95)
?raftery.diag


coda::autocorr.plot(as.mcmc(post0$mu))
coda::autocorr.diag(as.mcmc(post0$mu))

thin_interval = 8 # how far apart the iterations are for autocorrelation to be essentially 0.
thin_indx = seq(from=thin_interval, to=length(post2$mu), by=thin_interval)
head(thin_indx)


post0mu_thin = post0$mu[thin_indx]
traceplot(as.mcmc(post0$mu))
traceplot(as.mcmc(post0mu_thin))
coda::autocorr.plot(as.mcmc(post0mu_thin), lag.max=10)


#-------------------------------------------------------------------------------
####Burn-in####
#We have also seen how the initial value of the chain can affect how quickly the chain converges. If our initial value is far
#from the bulk of the posterior distribution, then it may take a while for the chain to travel there. We saw this in an earlier
#example.

set.seed(62)
post3 = MH(n=n, ybar=ybar, n_iter=500, mu_init=10.0, cand_sd=0.3)
coda::traceplot(as.mcmc(post3$mu))
#Clearly, the first 100 or so iterations do not reflect draws from the stationary distribution, so they should be discarded 
#before we use this chain for Monte Carlo estimates. This is called the "burn-in" period. You should always discard early 
#iterations that do not appear to be coming from the stationary distribution. Even if the chain appears to have converged early
#on, it is safer practice to discard an initial burn-in.



####Multiple chains, Gelman-Rubin####
# If we want to be more confident that we have converged to the true stationary distribution, we can simulate multiple chains, 
#each with a different starting value.
set.seed(61)
nsim = 500

post1 = MH(n=n, ybar=ybar, n_iter=nsim, mu_init=15.0, cand_sd=0.4)
post1$accpt

post2 = MH(n=n, ybar=ybar, n_iter=nsim, mu_init=-5.0, cand_sd=0.4)
post2$accpt

post3 = MH(n=n, ybar=ybar, n_iter=nsim, mu_init=7.0, cand_sd=0.1)
post3$accpt

post4 = MH(n=n, ybar=ybar, n_iter=nsim, mu_init=23.0, cand_sd=0.5)
post4$accpt

post5 = MH(n=n, ybar=ybar, n_iter=nsim, mu_init=-17.0, cand_sd=0.4)
post5$accpt


pmc = mcmc.list(as.mcmc(post1$mu), as.mcmc(post2$mu), 
                as.mcmc(post3$mu), as.mcmc(post4$mu), as.mcmc(post5$mu))
str(pmc)
coda::traceplot(pmc)
# It appears that after about iteration 200, all chains are exploring the stationary (posterior) distribution. 
#We can back up our visual results with the Gelman and Rubin diagnostic. This diagnostic statistic calculates the variability
#within chains, comparing that to the variability between chains. If all chains have converged to the stationary distribution, 
#the variability between chains should be relatively small, and the potential scale reduction factor, reported by the the 
#diagnostic, should be close to one. If the values are much higher than one, then we would conclude that the chains have not
#yet converged.
?gelman.diag

coda::gelman.diag(pmc)

coda::gelman.plot(pmc)

#From the plot, we can see that if we only used the first 50 iterations, the potential scale reduction factor or "shrink factor" 
#would be close to 10, indicating that the chains have not converged. But after about iteration 300, the "shrink factor" is 
#essentially one, indicating that by then, we have probably reached convergence. Of course, we shouldn't stop sampling as soon as
#we reach convergence. Instead, this is where we should begin saving our samples for Monte Carlo estimation.


### Monte Carlo estimation ###
#If we are reasonably confident that our Markov chain has converged, then we can go ahead and treat it as a Monte Carlo sample
#from the posterior distribution. Thus, we can use the techniques from Lesson 3 to calculate posterior quantities like the 
#posterior mean and posterior intervals from the samples directly.

nburn = 1000 # remember to discard early iterations
post0$mu_keep = post0$mu[-c(1:1000)]
summary(as.mcmc(post0$mu_keep))

mean(post$mu_keep > 1.0) # posterior probability that mu  > 1.0
















