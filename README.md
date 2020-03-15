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

MonteCarlo methods can be divided into two groups????
 - the first of which is devoted to sampling from a probability density
 - the second is to search for near optimal solutions to the problem ??????????????????too broad?? 
 
------------------------------------------------------------------------------------------------------
# Inverse Problem 
 - __Forward relation:__ `From reliable Data -> obtain posterior -> predictive` Let's say we know the position of particles(we know model parameters: `source`). This makes it easy to predict the collective behaviour of the particles(we can predict obv: `output`)
   - Mapping from the model space M into the **data space D**
     - ![formula](https://render.githubusercontent.com/render/math?math=d_\predict=g(m))
 
 - __Backward relation:__ `From posterior with unreliable Data -> obtain adjusted Data` However, even though we observe the particles' collective behavior(we know obv: `output`, but we are not clear), it's not to say we always understand the position of particles(we can't understand model parameters: `source`) to better predict the collective behaviour of the particles. 
   - Mapping from the Data space D into the **model space M**
     - ![formula](https://render.githubusercontent.com/render/math?math=d_\obv=g(m))  

Inverse problem theory is describing how information about a **parameterized physical system** can be derived from 1.`observational data`, 2.`prior information`, and 3.`theoretical relationships` between model parameters and data. 

Let's consider the "Inverse Problems" for which we have incomplete knowledge of the relationship between `data` and `model parameters`.
[Linear relations] between **data** and **model parameters**, for example, [assumptions] about `Gaussian noise` and `Gaussian A priori probability density` result in analytical expressions for resolution indicators like, for instance, "resolution kernels?" and a "posteriori covariance matrices?". But the situation is quite different for the general inverse problem.  

The general inverse problem is characterized by at least one of the following two complications:
 - crazy Likelihood: **Data** can only be computed from the model by means of a numerical algorithm. We don't 100% rely on the data..it might have a crazy noise..
 - crazy Prior: **A priori model parameter constraints** can only be expressed via numerical algorithms. The prior information is only available as an algorithm)
 - Non-linear relation b/w `data` and `model parameters`: `d` = g(`m`)??? `f(d) = ∫ L(m)σ(m)dm`???  

### So...We cannot 100% rely on our data and we don't have the prior. We apparently cannot get the Joint function! The only way to proceed is to use sampling methods that **`collect information on the posterior`**. Then we can get the adjusted predictive?

?? This is the case for many highly **non-linear problems**, where the forward relation is insusceptible to mathematical analysis. When analysing an inverse problem, obtaining a maximum likelihood model is usually not sufficient, as we normally also wish to have `information on the resolution power`? of the data. Once a collection of models sampled according to the **posterior** is available, it is possible to estimate `not only posterior model parameter covariances, but also resolution measures`(The resolution analysis may also provide new insight into problems that are usually treated by means of analytical methods).
> The resolution measures "non-uniqueness" and "uncertainty" of solution. No more point estimates! In the past, the analysis of nonlinearizable inverse problems were mainly focused on the construction of best fitting models, but today it is widely acknowledged that uncertainty and nonuniqueness analysis is very important for the assessment of scientific conclusions based on inverse calculations).  

# Intro to Inverse MonteCarlo
MC ultimately concerns `θ` based on `Data_x`. However, IMC ultimately concerns `Adjusting Data_x` by controlling `θ`. As an adaptive procedure, by controlling model parameter `θ` sampling, IMC makes the corresponding data sampling follows some desired distribution ? we need. In the general case, an inspection of the marginal probability densities of interest maybe impractical, but it is possible to **`pseudo-randomly generate a large collection of models according to the posterior probability distribution`**. This can be accomplished by means of an Inverse Monte Carlo method, even in cases where **no explicit formula for the a priori distribution is available**! `Metropolis algorithm`(importance sampling) gives a framework that allows analysis of (possibly highly nonlinear) inverse problems with `complex a priori information`(crazy prior) and `data with an arbitrary noise distribution`(crazy likelihood).
### Get your prior from Random Process..{prior ρ(m)} → {posterior σ(m)}
IMC first does Sampling `discrete models` in a **uniform random fashion** between pairs of upper and lower bounds, **which were chosen a priori**. Each generated model was tested for `its fit to the available data` and then accepted or rejected (based on data-model relation & data-noise distribution..which means **probabilities**(L1/L2) that depend on the models ability to reproduce observations). We can map out a region of acceptable models in parameter space. This was done by deterministically **sampling all models** in the vicinity(neighbouring parameters) of an acceptable model(candidate parameters), which had previously been determined by IMC. The whole process could then be repeated many times over(so keep generating different metagrams?). 
<img src="https://user-images.githubusercontent.com/31917400/76683610-62fb7780-65fd-11ea-979d-d5660d746c46.jpg" />

> One problem was that it is never known whether sufficient number of models had been tested. It was always possible that acceptable models may exist that bear no resemblance to the satisfactory models obtained. But it has some advantanges like: 
 - It can avoid all assumptions (such as linearity between the observables and the unknowns representing the model upon which most previous techniques relied).
 - A measure of uniqueness of the solutions would be obtained by **examining the degree** to which the successful models agreed or disagreed. 

### What's the matter with the old researches?
**σ(m)** represents our a posteriori information, deduced from `σ(m)` = k*`ρ(m)`*`L(m)`
 - **ρ(m)** represents our a priori information on models
 - **L(m)** represents our likelihood information as a measure of **degree of fit** between `data` predicted from models with `m` and `actual observed data`. Typically, this is done through the introduction of a **misfit function S(m)**, connected to **L(m)** through an expression: `L(m)` = k*exp(−`S(m)`) 
> When analyzing **nonlinear inverse problems of high dimensionality**, it is necessary to severely restrict the number of **misfit calculations**. Of course, it maybe ideal that all model parameters within the considered model subspace are visited, but the task is computationally unfeasible when problems with **many model parameters** are considered. In this regards, `Monte Carlo Search`, which consists of a **random walk** in the model space, extensively samples the model space, avoids **entrapment in local maxima** by taking advantage of the fact that **all local maxima** are sampled. 

One of the problems is that it requires an explicit formula for the a **priori distribution**! 

## New Approach General Sampling concept
So...We want the model solutions are sampled at a rate proportional to their a **posteriori probabilities**, that is, **`models` consistent with a "priori information" as well as "observations" are picked most often**, whereas models that are in incompatible with either a priori information or observations (or both) are rarely sampled. The general sampling algorithm can be described as consisting of two components: 
 - __Step 01:__ **generates** a `priori models`, that is, we aim to obtain some model parameters sampled with a frequency distribution (transition probability) that reaches the stationarity(equalibrium) which is accomplished by means of a random walk. This step may consist of a large number of mutually dependent sub-processes, each of which generates part of the a priori models. 
 - __Step 02:__ **accepts or rejects** attempted moves of the a `priori random walk` with probabilities that depend on the models ability to reproduce observations (Use your likelihood to examine). 


### Part 01. Probabilistic Formulation
__[A] Model parameters taking continuous values and Prior__

The prior knowledge is important! In the inverse problem, the actual values of the model parameter can be underdetermined, due to lack of **significant data** or due to **`experimental uncertainties`**...so many noise. It can also be overdetermined, if we repeat similar measurements. A better question would have been: **What information can we infer on the actual value of the model parameter vector m**? The Bayesian approach to inverse problems, describes the necessity of **`a priori information`**. We may have on the model vector m, by a **priori** density `ρ(m)`. As an example, when we describe experimental results(obv) `L(m)` by a vector of observed values ![formula](https://render.githubusercontent.com/render/math?math=d_\obv) with **`Gaussian experimental uncertainties`** described by a **covariance matrix `C`**, <img src="https://user-images.githubusercontent.com/31917400/76145629-05a67a00-6083-11ea-992b-91f52fa67db2.jpg" /> But how to introduce realistic a **priori** information in this model space? 

__[B] Model parameter discretization and Prior__

If ![formula](https://render.githubusercontent.com/render/math?math=m_i) is continuous, f(![formula](https://render.githubusercontent.com/render/math?math=m_i)) is a density, thus we can say the probability defined for a region of the space is: <img src="https://user-images.githubusercontent.com/31917400/76164915-854b4c00-614a-11ea-8af6-fb0af773d3a4.jpg" /> For numerical computations, we discretize the space by defining a grid of points, where each point represents a surrounding region ∆![formula](https://render.githubusercontent.com/render/math?math=m_a), ∆![formula](https://render.githubusercontent.com/render/math?math=m_b),....small enough for the probability densities under consideration to be almost constant inside it. Then, we say **P(![formula](https://render.githubusercontent.com/render/math?math=m_i))** is "the probability of the region ∆![formula](https://render.githubusercontent.com/render/math?math=m_a), ∆![formula](https://render.githubusercontent.com/render/math?math=m_b)... surrounding the point ![formula](https://render.githubusercontent.com/render/math?math=m_i)". In the limit of an infinitely dense grid and assuming a continuous f(m) , "the probability of the point ![formula](https://render.githubusercontent.com/render/math?math=m_i)" tends to ![formula](https://render.githubusercontent.com/render/math?math=f_i=f(m_i))∆![formula](https://render.githubusercontent.com/render/math?math=m_a)∆![formula](https://render.githubusercontent.com/render/math?math=m_b)...
If this is the case...
<img src="https://user-images.githubusercontent.com/31917400/76166280-571f3980-6155-11ea-9d74-7f3f58521ec3.jpg" />

> ### First sample the prior probability ![formula](https://render.githubusercontent.com/render/math?math=\rho_i) then we will modify this sampling procedure in such a way that the probability ![formula](https://render.githubusercontent.com/render/math?math=\sigma_i) is eventually sampled (One never creates a probability ex-nihilo(out of nothing) but rather modifies some prior into a posterior).

__[C] Designing an Equilibrium distribution of ![formula](https://render.githubusercontent.com/render/math?math=\p_i) and Prior__

**Question:** 
 - Given a set of points in the model space, with a probability ![formula](https://render.githubusercontent.com/render/math?math=\p_i) attached to every ![formula](https://render.githubusercontent.com/render/math?math=m_i), how can we define random rules to select points such that the probability of selecting point ![formula](https://render.githubusercontent.com/render/math?math=m_i) is ![formula](https://render.githubusercontent.com/render/math?math=\p_i)? **How to obtain the prior("support":![formula](https://render.githubusercontent.com/render/math?math=m_i), "frequency":![formula](https://render.githubusercontent.com/render/math?math=\p_i))?**

**Equilibrium:** 
 - Consider a random process that selects points in the model space. If the probability of selecting **point_i** is ![formula](https://render.githubusercontent.com/render/math?math=p_i), then the points ![formula](https://render.githubusercontent.com/render/math?math=m_i) selected by the process are called **"samples"** of the probability distribution {![formula](https://render.githubusercontent.com/render/math?math=p_i)}. The sampling methods is the random walks. The **possible paths** of a random walk define a graph in the model space. It obeys some probabilistic rules that allow it to jump from one model to a connected model in each step. This will, asymptotically, have some probability: ![formula](https://render.githubusercontent.com/render/math?math=p_i) to be `at` **point_i** `at` a **given step**. The neighborhood of given model is defined as the models to which a random walker can go in one step, if it starts at the given model. We say it's the transitional probability ![formula](https://render.githubusercontent.com/render/math?math=p_\ij) for the random walker to go to **point_i** if it currently is at the neighboring **point_j**, and ![formula](https://render.githubusercontent.com/render/math?math=\Sigma\p_\ij)=![formula](https://render.githubusercontent.com/render/math?math=p_1\j)+![formula](https://render.githubusercontent.com/render/math?math=p_2\j)+...=1. Let's say there is a probability ![formula](https://render.githubusercontent.com/render/math?math=q_i) that the random walk is initiated at **point_i**. Then, when the number of steps tends to infinity, the probability that the random walker is at **point_i** will converge to some other probability **"![formula](https://render.githubusercontent.com/render/math?math=\p_i)"** which is an equilibrium distribution of {"![formula](https://render.githubusercontent.com/render/math?math=\P_\ij)"}...so ![formula](https://render.githubusercontent.com/render/math?math=\Sigma\P_\ij\p_i=\p_i). Instead of letting ![formula](https://render.githubusercontent.com/render/math?math=\p_i) represent the probability that a (single) random walker is at **point_i**, we can let ![formula](https://render.githubusercontent.com/render/math?math=\p_i) be the number of "particles" at **point_i**. Then ![formula](https://render.githubusercontent.com/render/math?math=\Sigma\p_i) represents the "total number of particles"...and not 1.   
 
 - The design of a random walk that equilibrates at a desired distribution of {![formula](https://render.githubusercontent.com/render/math?math=\p_i)} (a.k.a our prior) can be formulated as the design of an equilibrium flow having a throughput of ![formula](https://render.githubusercontent.com/render/math?math=\p_i) particles at point_i. The simplest equilibrium flows are symmetric: ![formula](https://render.githubusercontent.com/render/math?math=\f_\ij=\f_\ji). See this flow: ![formula](https://render.githubusercontent.com/render/math?math=\f_\ij=\P_\ij\p_i). 
 - ![formula](https://render.githubusercontent.com/render/math?math=\P_\ij\p_i) is the probability that the next transition will be from j to i where ![formula](https://render.githubusercontent.com/render/math?math=\P_\ij) is the **conditional probability** of going to point_i if the random walker is at j. 
 - ![formula](https://render.githubusercontent.com/render/math?math=\f_\ij) is the **unconditional probability** that the next step will be a transition to i from j 

It is a "flow" as it can be interpreted as the number of particles going to point_i from point_j in a single step((The flow corresponding to an equilibrated random walk has the property that the number of particles ![formula](https://render.githubusercontent.com/render/math?math=\f_\ij=\p_i) at point_i is constant in time). Thus that a random walk has equilibrated at a distribution ![formula](https://render.githubusercontent.com/render/math?math=\f_\ij=\p_i) means that in each step, the total flow into a given point is equal to the total flow out from the point. And the flow has the property that the total flow out from point_i and hence the total flow into the another point must equal: ![formula](https://render.githubusercontent.com/render/math?math=\Sigma\f_\ij=\Sigma\f_\ki=\p_i)   

__[D] Two ways to process the metagraph__

Let's define P(`m`) for each metagraph we have!
 - __Option 01. Naive Random Walks:__ P(![formula](https://render.githubusercontent.com/render/math?math=m_i))=![formula](https://render.githubusercontent.com/render/math?math=\p_i) ∝ ![formula](https://render.githubusercontent.com/render/math?math=\n_i)
   - All points ![formula](https://render.githubusercontent.com/render/math?math=\m_i) have a probability proportional to their own number of neighbors(edges)...so it's an one and off case and this is the equilibrium distribution to obtain **`a single prior sample`**.  
   - ![formula](https://render.githubusercontent.com/render/math?math=\p_i=\n_i/\Sigma\n_j) where ![formula](https://render.githubusercontent.com/render/math?math=\n_i) is the number of neighbors of ![formula](https://render.githubusercontent.com/render/math?math=\m_i) (including the point ![formula](https://render.githubusercontent.com/render/math?math=\m_i) itself)...so the probability is obtained by the comparison of the neighbor sizes between you and your neighbors?    
    <img src="https://user-images.githubusercontent.com/31917400/76547840-c793c680-6485-11ea-8dab-dcfbb4da2c39.jpg" />
 
 - __Option 02. Uniform Random Walks:__ P(![formula](https://render.githubusercontent.com/render/math?math=m_i))=![formula](https://render.githubusercontent.com/render/math?math=\p_i) = C
   - All points ![formula](https://render.githubusercontent.com/render/math?math=\m_i) have the same probability, which implies too many output cases are possible. We might need to see the equilibrium distribution to obtain **`a single prior sample`**.  
   - The equilibrium distribution can be obtained by followings
     - if the "new" point has less neighbors than the "old" point (or the same number), then always accept the move to the "new" point!
     - if the "new" point has more neighbors than the "old" point, then make a random decision to move to the "new", or to stay at the "old", with the probability `alpha` = ![formula](https://render.githubusercontent.com/render/math?math=\n_j/\n_i)
     - ...`old/new` is big? then accept the move. `old/new` is small? then reject the move. 
   
### Part 02. Prior Sampling
There are two ways of defining the a **priori probability distribution**:
  - (1) By defining a (pseudo) **`random process`** (i.e., a set of pseudo **random rules**) whose output is models assumed to represent pseudo random realizations of `ρ(m)`: the **metagraph!**
    - If, for some reason, we are not able to directly design a random walk that samples the prior, but we have an expression that gives the value of the prior probability ![formula](https://render.githubusercontent.com/render/math?math=\rho_i) for any model ![formula](https://render.githubusercontent.com/render/math?math=\m_i), we can, for instance, start a **`Uniform Random Walk`** that samples the **model space** with uniform probability.
      - Using the Metropolis framework, replacing the likelihood values ![formula](https://render.githubusercontent.com/render/math?math=\L_i) by the prior probabilities ![formula](https://render.githubusercontent.com/render/math?math=\rho_i), we will produce a random walk that samples the prior (as clearly described above). 
    
  - (2) By explicitly giving a formula for the a priori probability density `ρ(m)`
    - We may choose the probability density: <img src="https://user-images.githubusercontent.com/31917400/76333067-02142c80-62e9-11ea-982b-75fde0f26607.jpg" />
    - In this example, where we only have an expression for `ρ(m)` , we have to generate samples from this distribution. This can be done in many different ways. One way is to start with a **`Naive Random Walk`** and then use the Metropolis rule to modify it, in order to sample the prior distribution `ρ(m)`.  


### Part 03. Posterior Sampling
- __[Decision_1] Use Data & Model parameter relation (use your Likelihood)__ 
   - Assume that some **random rules** are given that define a random walk having **prior** {![formula](https://render.githubusercontent.com/render/math?math=\rho_i)} as its equilibrium probability(unif or not). How can this rules be modified so that the **new random walk** converge at **posterior** {![formula](https://render.githubusercontent.com/render/math?math=\sigma_i)}? 
   - First, if that "proposed transition" i ← j was always accepted, then the random walker would sample the **prior probability**: {![formula](https://render.githubusercontent.com/render/math?math=\rho_i)}! 
   - Let us, however, instead of always accepting the proposed transition i ← j , sometimes thwart it by using the following rule to decide `if he is allowed to move to i` or `if he is forced to stay at j`: <img src="https://user-images.githubusercontent.com/31917400/76234932-4983b580-6222-11ea-80a1-e906d8122a8f.jpg" /> then the random walker will sample the **posterior probability** {![formula](https://render.githubusercontent.com/render/math?math=\sigma_i)}. This modification rule, reminiscent of the Metropolis algorithm, is not the only one possible.   
  
 - __[Decision_2] Use Data-Noise Distribution__ 
   - Let us consider the case of i.i.d Gaussian uncertainties. Then the likelihood function describing the experimental uncertainties degenerates into: <img src="https://user-images.githubusercontent.com/31917400/76240374-0843d380-622b-11ea-9c14-19aeb870163e.jpg" /> The acceptance probability for a perturbed model becomes in this case: <img src="https://user-images.githubusercontent.com/31917400/76240889-e7c84900-622b-11ea-96d5-b12bd6fc593d.jpg" />


### Part 04. IMC with Uniform Random Walk Prior example
- If we decide to discretize the model at constant **∆z** intervals, m ={ρ(z1),ρ(z2),...} will have some probability distribution (representing our a priori knowledge) for the parameters {ρ(z1),ρ(z2),...} which we may not need to characterize explicitly.
- In this example, the pseudo random procedure produces, by its very definition, samples m1,m2,... the a priori probability density ρ(m). These samples will be the input to the next modifying Metropolis decision rule.

 - What's your Problem?
   - We want to know most likely configuration of layers under the ground.
   - Noisy data: We don't believe the data collected coz..the measured data values are assumed to be contaminated. 
     <img src="https://user-images.githubusercontent.com/31917400/76699516-fc7e6400-66a5-11ea-91e7-30112c354974.jpg" />
   
   - We are not sure about data, so we are not sure about the parameter distribution of course...perhaps, we have to sample all possible parameter distributions..
   
   - __Dataset:__ gravity observation data
     - ![formula](https://render.githubusercontent.com/render/math?math=\d_i)=d(![formula](https://render.githubusercontent.com/render/math?math=\x_i))=![formula](https://render.githubusercontent.com/render/math?math=g'(\x_i)) where `g(x)` is the vertical component of the gravity.
     - the 20 triangle, the location of data collection at 20 equispaced points to the right of the fault, the first point being located 2 km from the fault, and the last point being located 40 km from the fault.
     - ### the mass density `k` across the **vertical fault** produce a **`gravity anomaly`** at the surface.
 - Prior Knowledge?
   - __parameters:__ depth `z`, log of mass density for each layer `k`, thinkness for each layer `l`
     - `l ~ Exp(λ)`, and `k ~ N(μ,σ2)`
     - total thinkness **Σ`l`** = 100 km
       <img src="https://user-images.githubusercontent.com/31917400/76699738-5bdd7380-66a8-11ea-9c61-5dd8d0f11da9.jpg" />

     - vertical profile of mass density: **ρ(`z`)** = ρ(`k`)ρ(`l`)
     - ∆**ρ(`z`)** is the horizontal density contrast across the fault at depth `z`
   - __predictive distribution__ where **G** is the gravitational constant
     <img src="https://user-images.githubusercontent.com/31917400/76699199-06529800-66a3-11ea-8f38-d3ff5af73530.jpg" />

 - Likelihood Knowledge?
   - We know that the measured data values are assumed to be contaminated by statistically independent, random errors ![formula](https://render.githubusercontent.com/render/math?math=\epsilon_i) modeled by some gaussian mixture, then the likelihood function **L(m)**, measuring the degree of fit between synthetic and observed data is... 
     <img src="https://user-images.githubusercontent.com/31917400/76700003-dc9d6f00-66aa-11ea-9063-9089df84dd2f.jpg" />















