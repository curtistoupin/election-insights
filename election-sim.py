# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 00:21:57 2024

@author: curti
"""
import random as rd
from math import comb as choose
from itertools import product
from tqdm import tqdm

def polls_conditional_prob(N, n, *polls):
    p = 1
    for poll in polls:
        p *= choose(n, sum(poll))*choose(N-n, len(poll)-sum(poll))/choose(N, len(poll))
    return p

def run_trial(print_flag = True):
    pop_pc = round(rd.gauss(0.5, 0.005), 5)
    DEM = 1
    REP = 0
    pop_total = 1000000
    pop = [DEM]*int(pop_pc*pop_total) + [REP]*int(round((1-pop_pc), 5)*pop_total)
    
    n_poll = 10
    polls = []
    for i in range(n_poll):
        poll_size = rd.randint(800, 1200)
        polls.append(rd.sample(pop, poll_size))
    
    N_resolution = 10000
    possible_N = list(range(int(round(0.9*pop_total/N_resolution)*N_resolution), 
                            int(round(1.1*pop_total/N_resolution)*N_resolution) + N_resolution, 
                            10000))                                                          
    n_resolution = 5000
    possible_n = list(range(int(round(0.4*pop_total/n_resolution)*n_resolution), 
                            int(round(0.6*pop_total/n_resolution)*n_resolution) + n_resolution, 
                            10000))
    
    p_N_n = 1/(len(possible_N)*len(possible_n))
    posterior = {}
    
    p_range = tqdm(product(possible_N, possible_n)) if print_flag else product(possible_N, possible_n)
    marginal_likelihood = 0
    for N, n in p_range:
        marginal_likelihood += polls_conditional_prob(N, n, *polls)*p_N_n
    
    p_range = tqdm(product(possible_N, possible_n)) if print_flag else product(possible_N, possible_n)  
    for N, n in p_range:
        prior = polls_conditional_prob(N, n, *polls) * p_N_n
        posterior[(N, n)] = prior/marginal_likelihood
        
    highest_likelihood = max(posterior, key=posterior.get)
    est_N = highest_likelihood[0]
    est_n = highest_likelihood[1]
    est_p = round(100*est_n/est_N, 2)
    if print_flag:
        print(f"There were {n_poll} polls condudcted, with the following results:")
        for poll in polls:
            print(f"\t{len(poll)} people polled, finding {sum(poll)} dem voters (or {round(100*sum(poll)/len(poll), 2)}%")
        print(f"The most likely outcome for this data is {est_N} total voters with {est_n} dem voters ({est_p}%).")
        print(f"The true number of voters is {pop_total}, with {round(100*pop_pc, 2)}% support for democrats.")
    return (est_N, est_n, est_p, pop_pc)

results = []
n_trial = 1500
for trial in tqdm(range(n_trial)):
    results.append(run_trial(False))
