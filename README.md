# LandscapeEvolution
Code developed for the manuscript "Evolution in alternating environments with tunable inter-landscape correlations" by [Jeff Maltas](https://scholar.google.com/citations?hl=en&user=Hk1ymawAAAAJ), Douglas M. McNally and [Kevin B. Wood](https://scholar.google.com/citations?user=GODI0AEAAAAJ&hl=en).

- [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/803619v1)

## Sections:

- [Motivation](#motivation)
- [Main takeaway](#main-takeaway)
- [Using the code](#using-the-code)

## Motivation

Natural populations are often exposed to temporally varying environments. The first step to understanding these dynamics is studying them with a well-understood model system. We construct pairs of fitness landscapes that share global fitness features (i.e. identical mean and variance) but are correlatied with one another in a tunable way, resulting in landscape pairs that range from perfectly anti-correlated to perfectly correlated. By studying how properties such as fitness change as a function of correlation due to periodic cycling we can begin to undertand the effect of temporally varying environments.

## Main takeaway

We find that for largely positively correlated landscapes, periodic cycling increases the avarage evolved fitness compared to single landscape evolution. However, for largely anti-correlated landscapes, we find their evolutionary dynamics begin to demonstrate ergodic-like behavior as they explore the entire fitness landscape indefinitely. This effect dramatically lowers the steady-state fitness in comparison to single landscape evolution.

## Using the code

The code defines a class "Landscape". 

<ins>Attributes:</ins>

N : int  
  Lenth of the bit sequence. A fitness landscape has 2^N genotypes.

sigma : float  
  Width of the normal distribution used to add noise to the landscape.

Bs : list  
  A list of landscapes correlated with the current landscape. This attribute is not initialized until the method "generate_correlated_landscapes" is called.
  
ls : ndarray (dim 2^N)  
  This is the array of fitness values for the current landsacpe. 
  
TM : ndarray (dim 2^N x 2^N)  
  The transition matrix for the landscape. As N gets large TMs can grow to be huge. Due to this, TM will not be set unless the method "get_TM" is called with store=True.
  
<ins>Methods:</ins>

get_TM(store=False)  
  Returns the transition matrix for the landscape. Generates the TM if it was not already stored.
  
find_max_indices()  
  Returns a list of indices of maximum fitness values in the landscape.
  
find_min_indices()  
  Returns a list of indices of minimum fitness values in the landscape.
  
evolve(steps, store_TM=False)  
  Impelements single landscape evolution on this landscape; returns a vector of genotype occupation probabilities.

evolve_switching(B, steps, store_TM=False)  
  Implements paired landscape evolution on this landscape and a B landscape; returns a vector of genotype occupation probabilities.
  
calc_fitness(steps, store_TMs=True)  
  Calculates fitness achieved after steps rounds of evolution in the single landscape case and paired landscape cases for each of the Bs.
  
graph(p=None, verbose=False)  
  Generates a graph representation of this landscape on the currently active matplotlib figure.
  
generate_correlated_landscapes(correl, without_shared_max=False, only_shared_max=False)  
  Generates and returns a list of paired B landscapes with correlations specified in correl; sets the Bs attribute of this landscape.

calc_nonzero_steadystate_prob(steps)  
  Computes the fraction of nonzero probability genotypes (out of the total number of genotypes) in the probability vector after steps rounds of evolution.
  
average_mutations(steps)  
  Returns the average number of mutations away from the initial state with nonzero probabilities. Uses smaller epsilon than calc_nonzero_steadystate_prob.
  
only_max_fit(self, Bs=None)  
  Calculates average fitness of the maximums in the A landscape if Bs=None, or a list of the average fitness of shared maximums in the A and B landscapes for each B in Bs.
  
get_steadystate_rounds(correl)  
  Calculates number of steps to reach steady state for paired landscape evolution.


