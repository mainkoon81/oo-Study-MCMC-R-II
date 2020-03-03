# Study-V003-MCMC-Python-R-II

### (B) Example for Metropolis Hastings II.(continuous MarkovChain - multiple parameters)
 - I have a model that is not conjugate. What should I do? Do MCMC or VI..dude..but you still need prior for sure
 
## 3. Gibbs Sampling(basic)
Sample the two parameters one at a time? `P(θ,ϕ|y) ∝ g(θ,ϕ)`
 - We would draw a **candidate for ϕ** using some proposal distribution `q( )` and use this `g(θ,ϕ)` where we plug in the value of **θ** to compute our `Metropolis-Hastings ratio`: **α**. We pretend we know the value of **θ** by substituting in `its current value` or `current iteration` from the Markov chain. Once we've drawn for both **θ** and **ϕ**, that completes one iteration and we begin the next iteration by drawing a new **θ** (we are just going back and forth, updating the parameters one at a time, plugging in the current value of the other parameter into the `g(θ,ϕ)`function. This idea of **one at a time updates** is used in what we call Gibbs sampling. 
 - Again, using the **chain rule** of probability, the joint posterior distribution can be factored: 
 ## --- `P(θ,ϕ|y) = P(θ|ϕ,y)*P(ϕ|y)` --- Full Joint Posterior
   - Notice that the only difference between this **full joint** posterior `P(θ,ϕ|y)` and this full **conditional** `P(θ|ϕ,y)` here, is multiplication by a factor `P(ϕ|y)` that does not involve **θ** at all.
 - When `g(θ,ϕ)` is viewed as a function of **θ**, it is proportional to both: 
   - full posterior **joint**: `P(θ,ϕ|y)`
   - **full-conditional** for θ: `P(θ|ϕ,y)`(meaning that `θ given everyting else`)
 - *Therefore, we can replace `g(θ,ϕ)` with the **full-conditional** `P(θ|ϕ,y)` when we performe the update for **θ**. *Why do we use this instead of `g(θ,ϕ)`? 
   - In some cases, the **full-conditional** is a **standard distribution** that we know how to sample!!! If that happens, we no longer need to draw a candidate and decide whether to accept it or not. In fact, if we treat the **full-conditional** a candidate proposal distribution `q( )`, the resulting Metropolis-Hastings acceptance probability **α** becomes exactly `1`. 
 - Gibbs Samplers require to find the **full-conditional** for each parameter beforehand. The good news is, that all **full-conditional** have the same starting point: the `full joint posterior distribution`! 
   - For θ: `g(θ,ϕ)` **∝** `P(θ|ϕ,y)` **∝** `P(θ,ϕ|y)`
   - For ϕ: `g(θ,ϕ)` **∝** `P(ϕ|θ,y)` **∝** `P(θ,ϕ|y)`
 - We always start with the `full joint posterior distribution`, thus the process of finding **full-conditional** is the same as finding the **posterior distribution of each parameter** and pretend that all of the other parameters are known constants. The idea of Gibbs sampling is that we can update multiple parameters by sampling just one parameter at a time and cycling through all parameters and then repeating.
   <img src="https://user-images.githubusercontent.com/31917400/48234880-651f4780-e3b3-11e8-9482-5f75b1fa19a5.jpg" />

 - Let's say we have a normal likelihood, with unknown mean `μ` and unknown variance `σ^2`. 
   <img src="https://user-images.githubusercontent.com/31917400/48290317-59428c80-e46a-11e8-9abe-f6f05c70e80a.jpg" />
   - In this case, we chose a **normal prior** for `μ` because when **σ^2** is a known constant, the normal distribution is the conjugate prior for `μ`. Likewise, in the case where **μ** is known, the **inverse gamma** is the conjugate prior for `σ^2`. This will give us the **full-conditional** in a Gibbs sampler. 
   - **FIRST**, If we work out the form of the `full joint posterior`(JAGS software will do this step for us),...   
     <img src="https://user-images.githubusercontent.com/31917400/48291210-8ba1b900-e46d-11e8-8122-cf2fc3e0be8c.jpg" />
   
   - **SECOND**, If we continue on to find the two **full-conditional** distributions,...
     <img src="https://user-images.githubusercontent.com/31917400/48292668-2781f380-e473-11e8-8d45-5101fd972d27.jpg" />
     - These two distributions provide the `basis of a Gibbs sampler` to simulate from a Markov chain whose stationary distribution is the full posterior of both μ and σ^2. We simply alternate draws between these two parameters, using the most recent draw of one parameter to update the other.
     - so in Gibbs sampler, we can complete an update for `μ` by simply drawing from the **Normal full-conditional** for `μ` since the `full conditional distribution` is a Normal distribution, this update is easy to simulate, and we can do it without drawing a candidate and deciding whether to accept it.
       - Of course, we can draw a candidate `μ∗` from a proposal distribution `q( )` and use the normal **full-conditional** for `μ` (which is `g(μ)`) or use the `full joint posterior` for `μ` and `σ^2` (which is `g(μ,σ^2)`) to evaluate the acceptance ratio, but it is not efficient.  

### Example:
 - To implement the Gibbs sampler, we return to our running example where the data are the percent change in total personnel from last year to this year for `n=10` companies. We’ll still use a **Normal likelihood**, but now we’ll relax the assumption that we know the variance of growth between companies `σ^2`, and estimate that variance. Instead of the **t-prior** from earlier, we will use the two **conditionally conjugate priors**: Normal for `μ` and inverse-gamma for `σ2`.
 - Simulate from the **full-conditional** we derived in the previous segment:
   <img src="https://user-images.githubusercontent.com/31917400/48294095-41730480-e47a-11e8-81e0-887ae85a983b.jpg" />
   
   ```
   update_mu = function(n, ybar, sig2, mu_0, sig2_0) {
     sig2_1 = 1.0 / (n / sig2 + 1.0 / sig2_0)
     mu_1 = sig2_1 * (n * ybar / sig2 + mu_0 / sig2_0)
     rnorm(n=1, mean=mu_1, sd=sqrt(sig2_1))
     }
   ```
   <img src="https://user-images.githubusercontent.com/31917400/48294103-446df500-e47a-11e8-8069-f5864e53fb44.jpg" />
   
   ```
   update_sig2 = function(n, y, mu, nu_0, beta_0) {
     nu_1 = nu_0 + n / 2.0
     sumsq = sum( (y - mu)^2 )                        # vectorized
     beta_1 = beta_0 + sumsq / 2.0
     out_gamma = rgamma(n=1, shape=nu_1, rate=beta_1) # rate for gamma is shape for inv-gamma
     1.0 / out_gamma                                  # reciprocal of a gamma random variable is distributed inv-gamma
     }
   ```
 - With functions for drawing from the full conditionals, we are ready to write a function to perform Gibbs sampling!!!!
```
gibbs = function(y, n_iter, init, prior) {
  ybar = mean(y)
  n = length(y)
  
  ## initialize
  mu_out = numeric(n_iter)
  sig2_out = numeric(n_iter)
  
  mu_now = init$mu
  
  ## Here Gibbs sampler is!!!!
  for (i in 1:n_iter) {
    # Since we started with the current draw from 'init$mu' above, let's update variance first!!!
    sig2_now = update_sig2(n, y, mu_now, nu_0 = prior$nu_0, beta_0 = prior$beta_0)
    mu_now = update_mu(n, ybar, sig2_now, mu_0 = prior$mu_0, sig2_0 = prior$sig2_0)
    
    sig2_out[i] = sig2_now
    mu_out[i] = mu_now
  }
  cbind(mu=mu_out, sig2=sig2_out)   # we're going to have a matrix of two columns with 'n_iter' rows. 
}
```
 - Next, we execute this. Here is our data.
 - Gibbs sampling function accepts our **prior** as a `list`.
 - We set `sig2_0`(initial_var) to 1.0, so that this prior is similar to the t-prior.
 - the particular parameterization of the `inverse gamma` is called the `scaled inverse chi-square`, where the two parameters are `n_0`(effective sample size for `sig2`) and `s2_0`(point estimate for `sig2`).???? Once we specify these two numbers, the parameters for our `inverse gamma` are automatically determined. 
```
y = c(1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9)
ybar = mean(y)
n = length(y)

## prior ##--------------------------------------------------------------------
prior = list()

prior$mu_0 = 0.0
prior$sig2_0 = 1.0

prior$n_0 = 2.0                             # prior effective sample size for sig2
prior$s2_0 = 1.0                            # prior point estimate for sig2

prior$nu_0 = prior$n_0 / 2.0                # prior parameter for inverse-gamma
prior$beta_0 = prior$n_0 * prior$s2_0 / 2.0 # prior parameter for inverse-gamma
#-------------------------------------------------------------------------------
hist(y, freq=FALSE, xlim=c(-1.0, 3.0))

# add individual data point!!(let them show up on x axis..) so..in Y-axis: rep(0,n)
points(y, rep(0,n), pch=1) # samples
points(ybar, 0, pch=19) # sample mean

# let's plot our prior distribution of the mean 
curve(dnorm(x, mean = prior$mu_0, sd = sqrt(prior$sig2_0)), lty=2, add=TRUE) # prior for mu..looks like t(0,1,1)..but it's normal
```
<img src="https://user-images.githubusercontent.com/31917400/48300886-a52e1980-e4dc-11e8-8cde-96b82f8dcd0c.jpg" />

 - Then let's run Gibbs sampling. We have the `data`, `init` and we have `prior`
```
set.seed(53)

init = list()
init$mu = 0.0
post = gibbs(y=y, n_iter=1e3, init=init, prior=prior)

library("coda")
plot(as.mcmc(post))  # plot()??? not traceplot()??? 
summary(as.mcmc(post))
```
<img src="https://user-images.githubusercontent.com/31917400/48300964-086c7b80-e4de-11e8-8dbe-cce6dca91043.jpg" />

## 4. Assessing Convergence
We've simulated a Markov chain whose stationary distribution is the target(posterior) distribution. Before using the simulated chain **to obtain Monte Carlo estimates**, we should first ask ourselves: `Has our simulated Markov chain converged to its **stationary distribution yet?` We don't know..
> Trace_plot
 - It shows the history of a parameter value across iterations of the chain. It shows you precisely where the chain has been exploring.
 - If the chain is stationary, it should not be showing any long-term trends. If this is the case, you need to run the chain many more iterations. 
```
post0 = MH(n, ybar, n_iter=1e3, mu_init=0.0, cand_sd=0.9)
coda::traceplot(as.mcmc(post0$mu[-c(1:500)]))

post1 = MH(n, ybar, n_iter=1e3, mu_init=0.0, cand_sd=0.04)
coda::traceplot(as.mcmc(post1$mu[-c(1:500)]))
```
<img src="https://user-images.githubusercontent.com/31917400/48550245-22390480-e8ca-11e8-9e72-f992faeb7f0a.jpg" />

> autocorrelation_plot & **Effective Sample_Size**
 - Looking at the level of autocorrelation in each. Autocorrelation is a number between −1 and +1 which measures `how linearly dependent the current value of the chain is on past values`. We call this `lags`!!!
```
coda::autocorr.plot(as.mcmc(post0$mu))
coda::autocorr.diag(as.mcmc(post0$mu))

coda::autocorr.plot(as.mcmc(post1$mu))
coda::autocorr.diag(as.mcmc(post1$mu))
```
<img src="https://user-images.githubusercontent.com/31917400/48553062-66c89e00-e8d2-11e8-8edb-bf15aee892e9.jpg" />

 - Autocorrelation is important because it tells us **how much information is available** in our Markov chain. 
   - Of course, sampling 1000 iterations from a **highly correlated Markov chain** yields less information about the stationary distribution than we would obtain from 1000 samples independently drawn from the stationary distribution.
 - Autocorrelation is a major component in calculating the `Monte Carlo Effective Sample_Size` of your chain which saying how many independent samples from the **stationary distribution** you would have to draw to have equivalent information in your **Markov chain**. Essentially it is the `m`(sample size) we chose on Monte Carlo estimation.
```
str(post0) # contains 1,000 iterations
str(post1) # contains 1,000 iterations

coda::effectiveSize(as.mcmc(post0$mu)) # effective sample size of ~170
coda::effectiveSize(as.mcmc(post1$mu)) # effective sample size of ~3
```
<img src="https://user-images.githubusercontent.com/31917400/48551842-95447a00-e8ce-11e8-803b-8854aa870051.jpg" />

 - **Thin out** the samples until autocorrelation is essentially `0`(for example, run the chain many more iterations..). This will leave you with approximately independent samples, and the number of samples remaining is similar to the `effective sample size`.
<img src="https://user-images.githubusercontent.com/31917400/48551842-95447a00-e8ce-11e8-803b-8854aa870051.jpg" />
 
 - The chain from `post0` has 1,000 iterations, but an effective sample size of about `170`. That is, this chain essentially provides the equivalent of `170` **independent Monte Carlo samples**.
 - It is usually a good idea to check the Monte Carlo effective sample size of your chain. If all you seek is a **posterior mean estimate**, then an effective sample size of a few hundred to a few thousand should be enough. However, if you want to create something like a **95% posterior interval**, you may need many thousands of effective samples to produce a reliable estimate of the outer edges of the distribution. The number you need can be quickly calculated using the `Raftery and Lewis diagnostic`.
```
raftery.diag(as.mcmc(post0$mu))
raftery.diag(as.mcmc(post0$mu), q=0.005, r=0.001, s=0.95)
```
<img src="https://user-images.githubusercontent.com/31917400/48552975-249f5c80-e8d2-11e8-894d-67e3a9579e41.jpg" />

> Burn-in
 - We have also seen how the `initial value` of the chain can affect how **quickly** the chain converges. If our initial value is far from the bulk of the posterior distribution, then it may take a while for the chain to travel there. In our earlier example, 
 <img src="https://user-images.githubusercontent.com/31917400/48793722-ca4a3580-ecef-11e8-839f-b4131f30d946.jpg" />

 - Clearly, **the first 100 or so iterations** do not reflect draws from the **stationary distribution**, so they should be discarded before we use this chain for Monte Carlo estimates. This is called the `“burn-in” period`. You should always discard early iterations that do not appear to be coming from the stationary distribution. Even if the chain appears to have converged early on, it is safer practice to discard an initial burn-in.

> Gelman-Rubin & Mulitple chains
 - If we want to be more confident that we have converged to the true stationary distribution, **we can simulate multiple chains, each with a different starting value**.
```
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

pmc = mcmc.list(as.mcmc(post1$mu), as.mcmc(post2$mu), as.mcmc(post3$mu), as.mcmc(post4$mu), as.mcmc(post5$mu))
str(pmc)
coda::traceplot(pmc)
```
<img src="https://user-images.githubusercontent.com/31917400/48794630-742ac180-ecf2-11e8-875a-e4d2188165dc.jpg" />

 - It appears that after about iteration 200, all chains are exploring the stationary (posterior) distribution. We can back up our visual results with the Gelman and Rubin diagnostic. This diagnostic statistic calculates the variability within chains, comparing that to the variability between chains. If all chains have converged to the stationary distribution, the variability between chains should be relatively small, and the potential scale reduction factor, reported by the the diagnostic, should be close to one. If the values are much higher than one, then we would conclude that the chains have not yet converged.
```
coda::gelman.diag(pmc)
coda::gelman.plot(pmc)
```

## 5. Monte Carlo estimation
If we are reasonably confident that our Markov chain has converged, then we can go ahead and treat it as a Monte Carlo sample from the posterior distribution. Calculate posterior quantities like the posterior mean and posterior intervals from the samples directly.
```
nburn = 1000 # remember to discard early iterations
post0$mu_keep = post0$mu[-c(1:1000)]
summary(as.mcmc(post0$mu_keep))

mean(post$mu_keep > 1.0) # posterior probability that mu  > 1.0
```

------------------------------------------------------------------------------------------------------
# Inverse Monte Carlo Method and Inverse Problem 
Let's consider the "Inverse Problems" for which we have incomplete knowledge of the relationship between data and model parameters. This is the case for many highly nonlinear problems, where the forward relation is insusceptible to mathematical analysis, and is only given by a complex algorithm. 

MonteCarlo methods can be divided into two groups
 - the first of which is devoted to sampling from a probability density
 - the second is designed to search for near optimal solutions to the problem. 

Sampling discrete models in a uniform random fashion between pairs of upper and lower bounds, which were chosen a priori. Each generated  model was tested for its fit to the available data and then accepted or rejected. 
 - It can avoid all assumptions (such as linearity between the observables and the unknowns representing the model upon which most previous techniques relied).
 - A measure of uniqueness of the solutions would be obtained by examining the degree to which the successful models agreed or disagreed. 
 - We can map out a region of acceptable models in parameter space. This was done by deterministically sampling all models in the vicinity of an acceptable model, which had previously been determined by MCI. The whole process could then be repeated many times over. 
 -  One problem was that it is never known whether sufficient number of models had been tested. It was always possible that acceptable models may exist that bear no resemblance to the satisfactory models obtained. 

Probabilistic formulation of inverse problems leads to the definition of a probability distribution in the model space. This probability distribution combines a priori information with new information obtained by measuring some observable parameters (data). As, in the general case, the theory linking data with model parameters is nonlinear, the a posteriori probability in the model space may not be easy to describe (it may be multimodal, some moments may not be defined, etc.). When analysing an inverse problem, obtaining a maximum likelihood model is usually not sufficient, as we normally also wish to have information on the resolution power of the data. In the general case we may have a large number of model parameters, and an inspection of the marginal probability densities of interest may be impractical, or even useless. But it is possible to pseudorandomly generate a large collection of models according to the posterior probability distribution and to analyse and display the models in such a way that information on the relative likelihoods of model properties is conveyed to the spectator. This can be accomplished by means of an efficient Monte Carlo method, even in cases where no explicit formula for the a priori distribution is available. The most well known importance sampling method, the Metropolis algorithm, can be generalized, and this gives a method that allows analysis of (possibly highly nonlinear) inverse problems with complex a priori information and data with an arbitrary noise distribution.









