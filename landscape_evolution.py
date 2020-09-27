import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.optimize as op
from scipy.stats.stats import pearsonr
import time
import copy
import matplotlib.pyplot as plt
import math
import itertools
import networkx as nx

class Landscape:
    """
    This class represents Landscapes which are used as the central objects for Markov evolutions and other calculations.

    ...

    Attributes
    ----------
    N : int
        The length of the bit sequences used to model genotypes. There will be 2^N genotypes in the Landscape.
    sigma : float
        Width of the normal distribution used to generate noise in the Landscape.
    Bs : list
        A list of Landscapes correlated with the current Landscape. This attribute won't be initialized until generate_correlated_landscapes is called.
    ls : ndarray (dim 2^N)
        The array of fitness values for the current Landscape. This is the essential data representing the Landscape.
    TM : ndarray (dim 2^N x 2^N)
        The Markov transition matrix for the landscape. Because TMs can be quite large, this attribute will not be set unless get_TM is called with store=True

    Methods
    -------
    get_TM(store=False)
        Returns the transition matrix for the Landscape. Generates the TM if it was not already stored.
    find_max_indices()
        Returns a list of indicies of maximum fitness values in the Landscape.
    find_min_indices()
        Returns a list of indicies of minimum fitness values in the Landscape.
    evolve(steps, store_TM=False)
        Implements single landscape evolution on this landscape; returns a vector of genotype occupation probabilities
    evolve_switching(B, steps, store_TM=False)
        Implements paired landscape evolution on this Landscape and a B Landscape; returns a vector of genotype occupation probabilities
    calc_fitness(steps, store_TMs=True)
        Calculates fitness achieved after steps rounds of evolution in the single landscape case and paired landscape cases for each of the Bs
    graph(p=None, verbose=False)
        Generates a graph representation of this Landscape on the currently active matplotlib figure.
    generate_correlated_landscapes(correl, without_shared_max=False, only_shared_max=False)
        Generates and returns a list of paired B landscapes with correlations specified in correl; sets the Bs attribute of this Landscape
    calc_nonzero_steadystate_prob(steps)
        Computes the fraction of nonzero probability genotypes (out of the total number of genotypes) in the probability vector after steps rounds of evolution
    average_mutations(steps)
        Returns the average number of mutations away from the initial state for states with nonzero probabilities. Uses smaller epsilon than calc_nonzero_steadystate_prob
    only_max_fit(self, Bs=None)
        Calculates average fitness of the maximums in the A landscape if Bs=None, or a list of the average fitness of shared maximums in the A and B landscapes for each B in Bs
    get_steadystate_rounds(correl)
        Calculates number of steps to reach steady state for paired landscape evolution
    """
    def __init__(self, N, sigma, ls=None, parent=None):
        """
        Initializes landscape objects with given N and sigma to simulate epistasis (zero sigma produces an additive landscape with exactly one global maximum).
        """
        self.N = N
        self.sigma = sigma
        self.Bs = None
        if ls is None:
            self.ls = np.array([0]) # Initializes landscape vector with fitness 0 in the first (wildtype) index
            fitness = np.random.uniform(-1, 1, N) # Generates N fitness values between -1 and 1 that will be used to generate the additive landscape.
            for mut in range(N):                                         # Loop that generates additive fitness landscape from the above "fitness" uniform random numbers.
                self.ls = np.append(self.ls, self.ls + fitness[mut])
            noise = np.random.normal(0, sigma, 2**N)                     # Generates array of gaussian niose length 2^N to match size of A landscape.
            self.ls = self.ls + noise                                    # Adds gaussian generated noise to A landscape at each A genotype
        else: self.ls = ls
        if parent is not None: self.parent = parent

    def get_TM(self, store=False):
        """
        Returns the transition matrix for this landscape. If store=True, it will
        be saved in a field of this object (TM) for later use. If a stored copy already
        exists for this landscape, it will be returned with no wasted computation.
        """
        if not hasattr(self, 'TM'):
            mut = range(self.N)                                               # Creates a list (0, 1, ..., N) to call for bitshifting mutations.
            TM = np.zeros((2**self.N,2**self.N))                                   # Transition matrix will be sparse (most genotypes unaccessible in one step) so initializes a TM with mostly 0s to do most work for us.

            for i in range(2**self.N):
                adjMut = [i ^ (1 << m) for m in mut]                     # For the current genotype i, creates list of genotypes that are 1 mutation away.

                adjFit = [self.ls[j] for j in adjMut]                         # Creates list of fitnesses for each corresponding genotype that is 1 mutation away.

                fitter = list(filter(lambda x: adjFit[x]>self.ls[i], mut))                      # Finds which indices of adjFit are more fit than the current genotype and thus available for mutation.

                fitLen = len(fitter)
                if fitLen == 0:                                          # If no mutations are more fit, stay in current genotype.
                    TM[i][i] = 1
                else:
                    tranVal = 1.0 / fitLen                                                   # If at least one mutation is more fit, assign a fitness of 1/(# of more fit mutatnts) to each accessible genotype.
                    for f in fitter:
                        TM[adjMut[f]][i] = tranVal
            if store: self.TM = TM # store the transition matrix for this landscape object
            return TM
        else: return self.TM

    def get_TM_phenom(self, phenom, store=False):
        """
        Returns the transition matrix for this landscape, with phenomenological stepping (see Tan and Gore 2012). If store=True, it will
        be saved in a field of this object (TM) for later use. If a stored copy already
        exists for this landscape, it will be returned with no wasted computation.
        """
        if not hasattr(self, 'TM'):
            mut = range(self.N)                                               # Creates a list (0, 1, ..., N) to call for bitshifting mutations.
            TM = np.zeros((2**self.N,2**self.N))                              # Transition matrix will be sparse (most genotypes unaccessible in one step) so initializes a TM with mostly 0s to do most work for us.

            for i in range(2**self.N):
                adjMut = [i ^ (1 << m) for m in mut]                          # For the current genotype i, creates list of genotypes that are 1 mutation away.

                adjFit = [self.ls[j] for j in adjMut]                         # Creates list of fitnesses for each corresponding genotype that is 1 mutation away.

                fitter = list(filter(lambda x: adjFit[x]>self.ls[i], mut))    # Finds which indices of adjFit are more fit than the current genotype and thus available for mutation.

                fitLen = len(fitter)
                if fitLen == 0:                                               # If no mutations are more fit, stay in current genotype.
                    TM[i][i] = 1
                else:
                    dfit = np.power([adjFit[f] - self.ls[i] for f in fitter], phenom)
                    prob_mut = np.divide(dfit,np.sum(dfit))
                    count = 0
                    for f in fitter:
                        TM[adjMut[f]][i] = prob_mut[count]
                        count += 1
            if store: self.TM = TM # store the transition matrix for this landscape object
            return TM
        else: return self.TM

    def get_TM_phenom_inf(self, store=False):
        """
        Returns the transition matrix for this landscape, with phenomenological stepping (see Tan and Gore 2012). If store=True, it will
        be saved in a field of this object (TM) for later use. If a stored copy already
        exists for this landscape, it will be returned with no wasted computation.
        """
        if not hasattr(self, 'TM'):
            mut = range(self.N)                                               # Creates a list (0, 1, ..., N) to call for bitshifting mutations.
            TM = np.zeros((2**self.N,2**self.N))                              # Transition matrix will be sparse (most genotypes unaccessible in one step) so initializes a TM with mostly 0s to do most work for us.

            for i in range(2**self.N):
                adjMut = [i ^ (1 << m) for m in mut]                          # For the current genotype i, creates list of genotypes that are 1 mutation away.

                adjFit = [self.ls[j] for j in adjMut]                         # Creates list of fitnesses for each corresponding genotype that is 1 mutation away.

                fitter = list(filter(lambda x: adjFit[x]>self.ls[i], mut))    # Finds which indices of adjFit are more fit than the current genotype and thus available for mutation.

                fitLen = len(fitter)
                if fitLen == 0:                                               # If no mutations are more fit, stay in current genotype.
                    TM[i][i] = 1
                else:
                    fitMax = np.argmax(adjFit)
                    TM[adjMut[fitMax]][i] = 1

            if store: self.TM = TM # store the transition matrix for this landscape object
            return TM
        else: return self.TM

    def find_one_step_neighbors(self, index):
        """
        Returns a list of indicies and a list of fitnesses in this landscape
        which are 1 mutational step away from the given index.
        """
        mut = range(self.N)
        adjMut = [index ^ (1 << m) for m in mut]                     # For the current genotype index, creates list of genotypes that are 1 mutation away.
        adjFit = [self.ls[j] for j in adjMut]
        return adjMut, adjFit

    def find_two_step_neighbors(self, index):
        """
        Returns a list of indicies and a list of fitnesses in this landscape
        which are 2 mutational step away from the given index.
        """
        mut = range(self.N)
        one_step, _ = self.find_one_step_neighbors(index)
        adjMut = set()
        for neighbor in one_step:
            for m in mut:
                adjMut.add(neighbor ^ (1 << m))
        adjMut.remove(index)
        adjFit = [self.ls[j] for j in adjMut]
        return adjMut, adjFit

    def find_two_step_neighbors2(self, index):
        """
        Alternate implementation for find_two_step_neighbors which is more
        generalizeable to finding n-step neighbors
        """
        adjMut = []
        for mut in range(2**self.N):
            count = 0
            for i in range(self.N):
                if ((index >> i) & 1) != (mut >> i) & 1:
                    count += 1
            if count == 2 and mut not in adjMut:
                adjMut.append(mut)
        adjFit = [self.ls[j] for j in adjMut]
        return adjMut, adjFit


    def find_max_indices(self):
        """
        Returns a list of indicies of maxes in this landscape
        """
        mut = range(self.N)
        maxes = []
        for i in range(2**self.N):
            adjMut = [i ^ (1 << m) for m in mut]
            adjFit = [self.ls[i] for i in adjMut]
            fitter = list(filter(lambda x: adjFit[x]>self.ls[i], mut))
            fitLen = len(fitter)
            if fitLen == 0:
                maxes.append(i)
        return maxes

    def find_min_indices(self):
        """
        Returns a list of indicies of mins in this landscape
        """
        mut = range(self.N)
        mins = []
        for i in range(2**self.N):
            adjMut = [i ^ (1 << m) for m in mut]
            adjFit = [self.ls[i] for i in adjMut]
            fitter = list(filter(lambda x: adjFit[x]>self.ls[i], mut))
            fitLen = len(fitter)
            if fitLen == self.N:
                mins.append(i)
        return mins

    def evolve(self, steps, store_TM=False):
        """
        Returns an array of genotype occupation probabilities after stepping in
        this landscape steps times.
        """
        TM = self.get_TM(store_TM)
        p0 = np.zeros((2**self.N,1))
        p0[0][0] = 1
        return np.dot(np.linalg.matrix_power(TM, steps), p0)

    def evolve_switching(self, B, steps, store_TM=False):
        """
        Returns an array of genotype occupation probabilities after alternating
        stepping in this landscape and the <B> landscape steps times. Note steps
        must be odd to ensure the last step is always in the A landscape.
        """
        if steps % 2 == 0: raise Exception("Only odd step counts allowed")
        ATM = self.get_TM()
        BTM = B.get_TM(store_TM)
        p0 = np.zeros((2**self.N,1))
        p0[0][0] = 1
        p0 = np.dot(ATM, p0)
        if steps == 1:
            return p0
        else:
            ABTM = np.dot(ATM,BTM)
            return np.dot(np.linalg.matrix_power(ABTM, (steps-1)//2), p0)

    def evolve_phenom(self, steps, phenom, store_TM=False):
        """
        Returns an array of genotype occupation probabilities after stepping in
        this landscape steps times.
        """
        TM = self.get_TM_phenom(phenom, store_TM)
        p0 = np.zeros((2**self.N,1))
        p0[0][0] = 1
        return np.dot(np.linalg.matrix_power(TM, steps), p0)

    def evolve_phenom_inf(self, steps, store_TM=False):
        """
        Returns an array of genotype occupation probabilities after stepping in
        this landscape steps times.
        """
        TM = self.get_TM_phenom_inf(store_TM)
        p0 = np.zeros((2**self.N,1))
        p0[0][0] = 1
        return np.dot(np.linalg.matrix_power(TM, steps), p0)

    def evolve_switching_phenom(self, B, steps, phenom, store_TM=False):
        """
        Returns an array of genotype occupation probabilities after alternating
        stepping in this landscape and the <B> landscape steps times. Note steps
        must be odd to ensure the last step is always in the A landscape.
        """
        if steps % 2 == 0: raise Exception("Only odd step counts allowed")
        ATM = self.get_TM_phenom(phenom)
        BTM = B.get_TM_phenom(phenom, store_TM)
        p0 = np.zeros((2**self.N,1))
        p0[0][0] = 1
        p0 = np.dot(ATM, p0)
        if steps == 1:
            return p0
        else:
            ABTM = np.dot(ATM,BTM)
            return np.dot(np.linalg.matrix_power(ABTM, (steps-1)//2), p0)

    def evolve_switching_phenom_inf(self, B, steps, store_TM=False):
        """
        Returns an array of genotype occupation probabilities after alternating
        stepping in this landscape and the <B> landscape steps times. Note steps
        must be odd to ensure the last step is always in the A landscape.
        """
        if steps % 2 == 0: raise Exception("Only odd step counts allowed")
        ATM = self.get_TM_phenom_inf()
        BTM = B.get_TM_phenom_inf(store_TM)
        p0 = np.zeros((2**self.N,1))
        p0[0][0] = 1
        p0 = np.dot(ATM, p0)
        if steps == 1:
            return p0
        else:
            ABTM = np.dot(ATM,BTM)
            return np.dot(np.linalg.matrix_power(ABTM, (steps-1)//2), p0)

    def calc_fitness(self, steps, store_TMs=True):
        """
        Returns:
        fitA-the average fitness in this landscape after <steps> rounds of evolution (scalar)
        fitAB_A-the average fitness achieved in this landscape when switching landscapes with each of the B landscapes
        fitAB_B-the average fitness achieved in each of the B landscapes in Bs when switching with this landscape
        """
        if self.Bs is None: raise Exception("Must call generate_correlated_landscapes() first.")
        Bs = self.Bs
        p = self.evolve(steps, store_TM=store_TMs)
        fitA = np.dot(self.ls, p)[0]
        fitAB_A = []
        fitAB_B = []
        for i in range(len(Bs)):
            p = self.evolve_switching(Bs[i], steps, store_TM=store_TMs)
            fitAB_A.append(np.dot(self.ls, p)[0])
            fitAB_B.append(np.dot(Bs[i].ls, p)[0])
        return (fitA, fitAB_A, fitAB_B)

    def calc_fitness_phenom(self, steps, phenom, store_TMs=True):
        """
        Returns:
        fitA-the average fitness in this landscape after <steps> rounds of evolution (scalar)
        fitAB_A-the average fitness achieved in this landscape when switching landscapes with each of the B landscapes
        fitAB_B-the average fitness achieved in each of the B landscapes in Bs when switching with this landscape
        """
        if self.Bs is None: raise Exception("Must call generate_correlated_landscapes() first.")
        Bs = self.Bs
        p = self.evolve_phenom(steps, phenom, store_TM=store_TMs)
        fitA = np.dot(self.ls, p)[0]
        fitAB_A = []
        fitAB_B = []
        for i in range(len(Bs)):
            p = self.evolve_switching_phenom(Bs[i], steps, phenom, store_TM=store_TMs)
            fitAB_A.append(np.dot(self.ls, p)[0])
            fitAB_B.append(np.dot(Bs[i].ls, p)[0])
        return (fitA, fitAB_A, fitAB_B)

    def calc_fitness_phenom_inf(self, steps, store_TMs=True):
        """
        Returns:
        fitA-the average fitness in this landscape after <steps> rounds of evolution (scalar)
        fitAB_A-the average fitness achieved in this landscape when switching landscapes with each of the B landscapes
        fitAB_B-the average fitness achieved in each of the B landscapes in Bs when switching with this landscape
        """
        if self.Bs is None: raise Exception("Must call generate_correlated_landscapes() first.")
        Bs = self.Bs
        p = self.evolve_phenom_inf(steps, store_TM=store_TMs)
        fitA = np.dot(self.ls, p)[0]
        fitAB_A = []
        fitAB_B = []
        for i in range(len(Bs)):
            p = self.evolve_switching_phenom_inf(Bs[i], steps, store_TM=store_TMs)
            fitAB_A.append(np.dot(self.ls, p)[0])
            fitAB_B.append(np.dot(Bs[i].ls, p)[0])
        return (fitA, fitAB_A, fitAB_B)

    def graph(self, p=None, verbose=False):
        """
        Plots a graph representation of this landscape on the current matplotlib figure.
        If p is set to a vector of occupation probabilities, the edges in the graph will
        have thickness proportional to the transition probability between nodes.
        """
        TM = self.get_TM()
        # Transpose TM because draw functions uses transposed version.
        TM = list(map(list, zip(*TM)))

        # Figure out the length of the bit sequences we're working with
        N = self.N

        # Generate all possible N-bit sequences
        genotypes = ["".join(seq) for seq in itertools.product("01", repeat=N)]

        # Turn the unique bit sequences array into a list of tuples with the bit sequence and its corresponding fitness
        # The tuples can still be used as nodes because they are hashable objects
        genotypes = [(genotypes[i], self.ls[i]) for i in range(len(genotypes))]

        # Build hierarchical structure for N-bit sequences that differ by 1 bit at each level
        hierarchy = [[] for i in range(N+1)]
        for g in genotypes: hierarchy[g[0].count("1")].append(g)

        # Add all unique bit sequences as nodes to the graph
        G = nx.DiGraph()
        G.add_nodes_from(genotypes)

        # Add edges with appropriate weights depending on the TM
        sf = 5 # edge thickness scale factor
        for i in range(len(TM)):
            for j in range(len(TM[i])):
                if TM[i][j] != 0 and i != j:
                    G.add_edge(genotypes[i], genotypes[j], weight=sf*TM[i][j])

        # Find the local & global min/max
        maxes = []
        mins = []
        for node in G:
            if len(G[node]) == 0:
                maxes.append(node)
            elif len(G[node]) == N:
                mins.append(node)

        # Determine which is global min/max
        # this algorithm can probably be simplified because...
        # global max will always have fitness = 1, and global min fitness = 0
        globalmax = 0
        globalmin = 0
        for i in range(1,len(maxes)):
            if maxes[i][1] > maxes[globalmax][1]:
                globalmax = i
        for i in range(1,len(mins)):
            if mins[i][1] < mins[globalmin][1]:
                globalmin = i
        globalmax = maxes[globalmax]
        globalmin = mins[globalmin]
        maxes.remove(globalmax)
        mins.remove(globalmin)

        # Create label dict for max/min nodes
        labels = {}
        labels[globalmax] = "+"
        labels[globalmin] = "-"
        for n in maxes:
            labels[n] = "+"
        for n in mins:
            labels[n] = "-"

        # Store all the edge weights in a list so they can be used to control the edge widths when drawn
        edges = G.edges()
        weights = [G[u][v]['weight'] for u,v in edges]

        # just using spring layout to generate an initial dummy pos dict
        pos = nx.spring_layout(G)

        # calculate how many entires in the longest row, it will be N choose N/2
        # because the longest row will have every possible way of putting N/2 1s (or 0s) into N bits
        maxLen = math.factorial(N) / math.factorial(N//2)**2

        # Position the nodes in a layered hierarchical structure by modifying pos dict
        y = 1
        for row in hierarchy:
            if len(row) > maxLen: maxLen = len(row)
        for i in range(len(hierarchy)):
            levelLen = len(hierarchy[i])
            # algorithm for horizontal spacing.. may not be 100% correct?
            offset = (maxLen - levelLen + 1) / maxLen
            xs = np.linspace(0 + offset / 2, 1 - offset / 2, levelLen)
            for j in range(len(hierarchy[i])):
                pos[hierarchy[i][j]] = (xs[j], y)
            y -= 1 / N

        # Print node structure to console
        if verbose:
            for i in range(len(hierarchy)):
                print(("Row {}: " + str([h[0] for h in hierarchy[i]]).strip('[]')).format(i+1))
            print()

        node_size = 500
        if p is not None:
            node_size = [75 + 1000*val for val in p]

        # Draw the graph
        plt.axis('off')
        node_vals = [g[1] for g in G.nodes()]
        nx.draw(G, pos, with_labels=False, width=weights, linewidths=1, cmap=plt.get_cmap('Greys'), node_color=node_vals,node_size=node_size)
        nx.draw_networkx_labels(G,pos,labels,font_size=16,font_color='red') # labels for min/max nodes
        ax = plt.gca()
        ax.collections[0].set_edgecolor("#000000")

    def graphTraj(self, TM, N, p=None, verbose=False):
        """
        Modified version of graph(). Depreciated.
        """
        #TM = self.get_TM()
        # Transpose TM because draw functions uses transposed version.
        TM = list(map(list, zip(*TM)))

        # Figure out the length of the bit sequences we're working with
        N = N

        # Generate all possible N-bit sequences
        genotypes = ["".join(seq) for seq in itertools.product("01", repeat=N)]

        # Turn the unique bit sequences array into a list of tuples with the bit sequence and its corresponding fitness
        # The tuples can still be used as nodes because they are hashable objects
        genotypes = [(genotypes[i], self.ls[i]) for i in range(len(genotypes))]

        # Build hierarchical structure for N-bit sequences that differ by 1 bit at each level
        hierarchy = [[] for i in range(N+1)]
        for g in genotypes: hierarchy[g[0].count("1")].append(g)

        # Add all unique bit sequences as nodes to the graph
        G = nx.DiGraph()
        G.add_nodes_from(genotypes)

        # Add edges with appropriate weights depending on the TM
        sf = 5 # edge thickness scale factor
        for i in range(len(TM)):
            for j in range(len(TM[i])):
                if TM[i][j] != 0 and i != j:
                    G.add_edge(genotypes[i], genotypes[j], weight=sf*TM[i][j])

        # Find the local & global min/max
        maxes = []
        mins = []
        for node in G:
            if len(G[node]) == 0:
                maxes.append(node)
            elif len(G[node]) == N:
                mins.append(node)

        # Create label dict for max/min nodes
        labels = {}
        for n in maxes:
            labels[n] = " "
        for n in mins:
            labels[n] = " "

        # Store all the edge weights in a list so they can be used to control the edge widths when drawn
        edges = G.edges()
        weights = [G[u][v]['weight'] for u,v in edges]

        # just using spring layout to generate an initial dummy pos dict
        pos = nx.spring_layout(G)

        # calculate how many entires in the longest row, it will be N choose N/2
        # because the longest row will have every possible way of putting N/2 1s (or 0s) into N bits
        maxLen = math.factorial(N) / math.factorial(N//2)**2

        # Position the nodes in a layered hierarchical structure by modifying pos dict
        y = 1
        for row in hierarchy:
            if len(row) > maxLen: maxLen = len(row)
        for i in range(len(hierarchy)):
            levelLen = len(hierarchy[i])
            # algorithm for horizontal spacing.. may not be 100% correct?
            offset = (maxLen - levelLen + 1) / maxLen
            xs = np.linspace(0 + offset / 2, 1 - offset / 2, levelLen)
            for j in range(len(hierarchy[i])):
                pos[hierarchy[i][j]] = (xs[j], y)
            y -= 1 / N

        # Print node structure to console
        if verbose:
            for i in range(len(hierarchy)):
                print(("Row {}: " + str([h[0] for h in hierarchy[i]]).strip('[]')).format(i+1))
            print()

        node_size = 500
        if p is not None:
            node_size = [10 + 1000*val for val in p]

        # Draw the graph
        plt.axis('off')
        node_vals = [g[1] for g in G.nodes()]
        nx.draw(G, pos, with_labels=False, width=weights, linewidths=1, cmap=plt.get_cmap('Greys'), node_color=node_vals,node_size=node_size)
        nx.draw_networkx_labels(G,pos,labels,font_size=16,font_color='red') # labels for min/max nodes
        ax = plt.gca()
        ax.collections[0].set_edgecolor("#000000")

    def generate_correlated_landscapes(self, correl, without_shared_max=False, only_shared_max=False, count_tries=False):
        """
        Returns landscapes correlated to A for each correlation value specified in correl.
        The B landscapes will also be stored as a list in the self.Bs attribute
        If without_shared_max is set, the correlated landscapes will be guaranteed to not have
        shared maximums with the A landscape. NOTE: running time will dramatically increase with this flag
        If only_shared_max is set, the correlated landscapes will be guaranteed to have at least
        one shared maximum with the A landscape. Note both without_shared_max and only_shared_max cannot both be set.
        """
        if without_shared_max and only_shared_max:
            raise Exception("You cannot set both without_shared_max and only_shared_max")
        Bs = [None]*len(correl)
        Astd = np.std(self.ls, ddof=1) # have to use ddof=1 to match matlab sample std
        Amean = np.mean(self.ls)
        ls = (self.ls - Amean)/Astd
        M = la.orth(np.array([np.ones(2**self.N), ls]).T)
        if not count_tries:
            y0 = np.random.uniform(0,1,(2**self.N,1))
            dp = np.dot(y0.T, M)
            y0 = y0 - np.dot(M, dp.T)
            y0_std = np.std(y0, ddof=1)
            y0 /= y0_std
            y0 = np.array(y0.T[0])
        else: tries = np.zeros(len(correl))

        rhos = np.array([])
        for i in range(len(correl)):
            rhos = np.append(rhos, correl[i])
        if without_shared_max or only_shared_max:
            Amaxes = self.find_max_indices()
        for i in range(len(rhos)):
            if count_tries:
                y0 = np.random.uniform(0,1,(2**self.N,1))
                dp = np.dot(y0.T, M)
                y0 = y0 - np.dot(M, dp.T)
                y0_std = np.std(y0, ddof=1)
                y0 /= y0_std
                y0 = np.array(y0.T[0])
                tries[i] += 1
            r = rhos[i]
            Als = np.array(copy.deepcopy(self.ls))
            if without_shared_max and r != -1:
                if r == 1: raise Exception("It is not possible to have a landscape with 1 correlation that doesn't share a maximum")
                shared_max = True
                while shared_max:
                    shared_max = False
                    if abs(r) < np.finfo(np.float64).eps: # zero correlation case
                        y = Amean + y0 * Astd
                        Bs[i] = Landscape(self.N, self.sigma, ls=y, parent=self)
                        r = 0
                    elif r < 0:
                        r = -r
                        Als = -Als
                    if r < 1 and r > 0:
                        fun = lambda beta : r - pearsonr(Als, y0 + beta*(Als-y0))[0]
                        beta = op.brentq(fun, 0, 1)
                        y = y0 + beta * (Als-y0)
                        y = Amean + y * Astd
                        Bs[i] = Landscape(self.N, self.sigma, ls=y.T, parent=self)
                    Bmaxes = Bs[i].find_max_indices()
                    for m in Amaxes:
                        if m in Bmaxes:
                            shared_max = True
                            break
                    if shared_max: # re-roll y0
                        if count_tries: tries[i] += 1
                        y0 = np.random.uniform(0,1,(2**self.N,1))
                        dp = np.dot(y0.T, M)
                        y0 = y0 - np.dot(M, dp.T)
                        y0_std = np.std(y0, ddof=1)
                        y0 /= y0_std
                        y0 = np.array(y0.T[0])
                        #if tries > 100 and tries % 100 == 0: print("Correlation {:.2f} has taken {} tries".format(r, tries))
            elif only_shared_max and r != 1:
                if r == -1: raise Exception("It is not possible to have a landscape with -1 correlation that shares a maximum")
                shared_max = False
                while not shared_max:
                    if abs(r) < np.finfo(np.float64).eps: # zero correlation case
                        y = Amean + y0 * Astd
                        Bs[i] = Landscape(self.N, self.sigma, ls=y, parent=self)
                        r = 0
                    elif r < 0:
                        r = -r
                        Als = -Als
                    if r < 1 and r > 0:
                        fun = lambda beta : r - pearsonr(Als, y0 + beta*(Als-y0))[0]
                        beta = op.brentq(fun, 0, 1)
                        y = y0 + beta * (Als-y0)
                        y = Amean + y * Astd
                        Bs[i] = Landscape(self.N, self.sigma, ls=y.T, parent=self)
                    Bmaxes = Bs[i].find_max_indices()
                    for m in Amaxes:
                        if m in Bmaxes:
                            shared_max = True
                            break
                    if not shared_max: # re-roll y0
                        if count_tries: tries[i] += 1
                        y0 = np.random.uniform(0,1,(2**self.N,1))
                        dp = np.dot(y0.T, M)
                        y0 = y0 - np.dot(M, dp.T)
                        y0_std = np.std(y0, ddof=1)
                        y0 /= y0_std
                        y0 = np.array(y0.T[0])
                        #if tries > 100 and tries % 100 == 0: print("Correlation {:.2f} has taken {} tries".format(r, tries))
            else:
                if abs(r) < np.finfo(np.float64).eps: # zero correlation case
                    y = Amean + y0 * Astd
                    Bs[i] = Landscape(self.N, self.sigma, ls=y, parent=self)
                    r = 0
                elif r == 1: # guaranteed to have a shared max, so the only_shared_max case is handled here already
                    Bs[i] = Landscape(self.N, self.sigma, ls=Als, parent=self)
                elif r == -1: # guaranteed to not have a shared max, so the without_shared_max case is handled here already
                    Bs[i] = Landscape(self.N, self.sigma, ls=-Als, parent=self)
                elif r < 0:
                    r = -r
                    Als = -Als
                if r < 1 and r > 0:
                    fun = lambda beta : r - pearsonr(Als, y0 + beta*(Als-y0))[0]
                    beta = op.brentq(fun, 0, 1)
                    y = y0 + beta * (Als-y0)
                    y = Amean + y * Astd
                    Bs[i] = Landscape(self.N, self.sigma, ls=y.T, parent=self)
        self.Bs = Bs
        if count_tries:
            return Bs, tries
        return Bs

    def calc_nonzero_steadystate_prob(self, steps):
        """
        Computes the fraction of nonzero probability genotypes (out of the total number of genotypes) in the probability vector after steps rounds of evolution
        Returns:
        Aonly_nonzero - The fraction of nonzero states for evolution in only the A landscape_evolution
        AB_nonzero - The fraction of nonzero states for switching evolution between this landscape and each landscape in self.Bs
        pAonly - The result of self.evolve(steps)
        pAB - The average probability vector between the A and B landscapes for switching evoltion for each landscape in Bs
        """
        if self.Bs is None: raise Exception("Must call generate_correlated_landscapes() first.")
        Bs = self.Bs
        N = self.N
        epsilon = np.finfo(np.float64).eps # numpy float64 precision
        pAonly = self.evolve(steps)
        Aonly_nonzero = 0
        for i in range(len(pAonly)):
            if pAonly[i] > epsilon:
                Aonly_nonzero += 1
        Aonly_nonzero /= 2**N
        #np.count_nonzero(pAonly) / 2**N

        pA = np.zeros((len(Bs),2**N))
        pB = np.zeros((len(Bs),2**N))
        AB = np.zeros(len(Bs))

        for i in range(len(Bs)):
            pA[i] = self.evolve_switching(Bs[i], steps).flatten()
            pB[i] = np.dot(Bs[i].get_TM(), pA[i]).flatten()

        pAB = (pA + pB) / 2.0
        #AB = np.count_nonzero(pAB, axis = 1) / 2**N
        AB_nonzero = np.zeros(len(pAB))
        for i in range(len(AB)):
            for j in range(len(pAB[i])):
                if pAB[i][j] > 1e-10:#epsilon:
                    AB_nonzero[i] += 1
        AB_nonzero = np.divide(AB_nonzero, 2**N)

        return (Aonly_nonzero, AB_nonzero, pAonly, pAB)

    def average_mutations(self, steps, single_landscape=False):
        """
        Returns the average number of mutations away from the initial state
        for states with nonzero probabilities.
        """
        genotypes = ["".join(seq) for seq in itertools.product("01", repeat=self.N)]
        epsilon = np.finfo(np.float64).eps
        A_mutations = 0
        _, _, pAonly, pAB = self.calc_nonzero_steadystate_prob(steps)

        for i in range(len(pAonly)):
            if pAonly[i] > epsilon:
                A_mutations += pAonly[i] * genotypes[i].count("1")
        AB_mutations = np.zeros(len(pAB))
        for i in range(len(pAB)):
            for j in range(len(pAB[i])):
                if pAB[i][j] > epsilon:
                    AB_mutations[i] += pAB[i][j] * genotypes[j].count("1")

        return (A_mutations, AB_mutations)

    def only_max_fit(self, Bs=None):
        """
        Returns either the average fitness of the maximums in the A landscape if Bs=None,
        or a list of the average fitness of shared maximums in the A and B landscapes for each B in Bs
        """
        Amaxes = self.find_max_indices()
        if Bs is None:
            totalmaxfit = 0
            for m in Amaxes:
                totalmaxfit += self.ls[m]
            return totalmaxfit / len(Amaxes)
        else:
            switching_avg_max_fit = []
            for B in Bs:
                Bmaxes = B.find_max_indices()
                totalmaxfit = 0
                count = 0
                for m in Bmaxes:
                    if m in Amaxes:
                        totalmaxfit += self.ls[m]
                        count += 1
                if totalmaxfit != 0: switching_avg_max_fit.append(totalmaxfit / count)
                else: switching_avg_max_fit.append(float('nan'))
            return np.array(switching_avg_max_fit)

    def get_steadystate_rounds(self, correl):
        """
        Calculates number of steps to reach steady state for paired landscape evolution
        Returns a list of steps to steady state for PLE for each of the correlations in correl
        """
        epsilon = 0.001
        ss_found = [False for _ in range(len(correl))]
        steps_list = np.zeros(len(correl))
        Bs = self.generate_correlated_landscapes(correl)
        # calculate steps to steady state:
        steps = 1
        prev = []
        for i in range(len(correl)):
            B = Bs[i]
            prev.append(self.evolve_switching(B, steps, store_TM=True)) # evolve 1 step first for comparison
        flag = True
        while flag:
            p = [[] for _ in range(len(correl))]
            steps += 2 # only odd step counts possible for switching
            for i in range(len(correl)):
                if not ss_found[i]: # only do the calculation for Bs that haven't found SS yet
                    B = Bs[i]
                    p[i] = self.evolve_switching(B, steps, store_TM=True)
                    if la.norm(prev[i] - p[i]) < epsilon: # condition for steady state
                        ss_found[i] = True # found steady state for this correlation
                        steps_list[i] = steps # record number of steps to SS for this correlation
                    prev[i] = p[i]
            flag = False
            for v in ss_found: # check if SS has been found for all the correlations
                if not v:
                    flag = True
                    break
        return steps_list

    def __repr__(self):
        return str(self.ls)

    def __str__(self):
        return self.__repr__()
