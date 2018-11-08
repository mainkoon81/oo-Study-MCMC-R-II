# Study-V003-MCMC-Python-R-II

### (B) Example for Metropolis Hastings II.(continuous MarkovChain - multiple parameters)
 - I have a model that is not conjugate. What should I do?
## 3. Gibbs Sampling
Sample the two parameters one at a time? `P(θ,ϕ|y) ∝ g(θ,ϕ)`
 - We would draw a **candidate for ϕ** using some proposal distribution and use this `g( )` where we plug in the value of **θ** to compute our `Metropolis-Hastings ratio`. We pretend we know the value of **θ** by substituting in `its current value` or `current iteration` from the Markov chain. Once we've drawn for both **θ** and **ϕ**, that completes one iteration and we begin the next iteration by drawing a new **θ** (we are just going back and forth, updating the parameters one at a time, plugging in the current value of the other parameter into the `g( )`function. This idea of **one at a time updates** is used in what we call Gibbs sampling. 
 - Note! Again using the **chain rule** of probability, the joint posterior distribution can be factored: `P(θ,ϕ|y) = P(ϕ|y)*P(θ|ϕ,y)`












































