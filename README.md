# Study-V003-MCMC-Python-R-II

### (B) Example for Metropolis Hastings II.(continuous MarkovChain - multiple parameters)
 - I have a model that is not conjugate. What should I do?
## 3. Gibbs Sampling
Sample the two parameters one at a time? `P(θ,ϕ|y) ∝ g(θ,ϕ)`
 - We would draw a **candidate for ϕ** using some proposal distribution `q( )` and use this `g(θ,ϕ)` where we plug in the value of **θ** to compute our `Metropolis-Hastings ratio`: **α**. We pretend we know the value of **θ** by substituting in `its current value` or `current iteration` from the Markov chain. Once we've drawn for both **θ** and **ϕ**, that completes one iteration and we begin the next iteration by drawing a new **θ** (we are just going back and forth, updating the parameters one at a time, plugging in the current value of the other parameter into the `g(θ,ϕ)`function. This idea of **one at a time updates** is used in what we call Gibbs sampling. 
 - Again, using the **chain rule** of probability, the joint posterior distribution can be factored: `P(θ,ϕ|y) = P(θ|ϕ,y)*P(ϕ|y)`
   - Notice that the only difference between this full **joint** posterior `P(θ,ϕ|y)` and this full **conditional** `P(θ|ϕ,y)` here, is multiplication by a factor `P(ϕ|y)` that does not involve **θ** at all.
 - When `g(θ,ϕ)` is viewed as a function of **θ**, it is proportional to both: 
   - full posterior **joint**: `P(θ,ϕ|y)`
   - **full-conditional** for θ: `P(θ|ϕ,y)`(meaning that `θ given everyting else`)
 - *Therefore, we can replace `g(θ,ϕ)` with the **full-conditional** `P(θ|ϕ,y)` when we performe the update for **θ**. *Why do we use this instead of `g(θ,ϕ)`? 
   - In some cases, the **full-conditional** is a **standard distribution** that we know how to sample!!! If that happens, we no longer need to draw a candidate and decide whether to accept it or not. In fact, if we treat the **full-conditional** a candidate proposal distribution `q( )`, the resulting Metropolis-Hastings acceptance probability **α** becomes exactly `1`. 
 - Gibbs Samplers require to find the **full-conditional** for each parameter beforehand. The good news is, that all **full-conditional** have the same starting point: the `full posterior joint distribution`! 
   - For θ: `g(θ,ϕ)` **∝** `P(θ|ϕ,y)` **∝** `P(θ,ϕ|y)`
   - For ϕ: `g(θ,ϕ)` **∝** `P(ϕ|θ,y)` **∝** `P(θ,ϕ|y)`
 - We always start with the `full posterior joint distribution`, thus the process of finding **full-conditional** is the same as finding the **posterior distribution of each parameter** and pretend that all of the other parameters are known constants. The idea of Gibbs sampling is that we can update multiple parameters by sampling just one parameter at a time and cycling through all parameters and then repeating.
   <img src="https://user-images.githubusercontent.com/31917400/48234880-651f4780-e3b3-11e8-9482-5f75b1fa19a5.jpg" />

 - Let's say we have a normal likelihood, with unknown mean `μ` and unknown variance `σ^2`. 
   <img src="https://user-images.githubusercontent.com/31917400/48290317-59428c80-e46a-11e8-9abe-f6f05c70e80a.jpg" />
   - In this case, we chose a **normal prior** for `μ` because when **σ^2** is a known constant, the normal distribution is the conjugate prior for `μ`. Likewise, in the case where **μ** is known, the **inverse gamma** is the conjugate prior for `σ^2`. This will give us the **full-conditional** in a Gibbs sampler. 
   - **FIRST**, If we work out the form of the `full posterior distribution`(JAGS software will do this step for us),...   
     <img src="https://user-images.githubusercontent.com/31917400/48291210-8ba1b900-e46d-11e8-8122-cf2fc3e0be8c.jpg" />
   
   - **SECOND**, If we continue on to find the two **full-conditional** distributions,...
   











































