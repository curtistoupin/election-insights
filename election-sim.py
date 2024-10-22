# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 00:21:57 2024

@author: curti
"""
import random as rd
from math import comb as choose
from itertools import product
from tqdm import tqdm

DEM = 1
REP = 0

def polls_conditional_prob(N, n, *polls):
    p = 1
    for poll in polls:
        p *= choose(n, sum(poll))*choose(N-n, len(poll)-sum(poll))/choose(N, len(poll))
    return p

def calculate_posterior(polls, 
                        N_min = 900000,
                        N_max = 1100000,
                        N_resolution = 10000,
                        pc_min = 0.4,
                        pc_max = 0.6,
                        n_resolution = 5000,
                        print_flag = True):
    possible_N = list(range(int(N_min/N_resolution)*N_resolution,
                           int(N_max/N_resolution)*N_resolution + N_resolution,
                           N_resolution))
    n_min = pc_min*N_min
    n_max = pc_max*N_max
    possible_n = list(range(int(n_min/n_resolution)*n_resolution,
                            int(n_max/n_resolution)*n_resolution,
                            n_resolution))
    p_N_n = 1/(len(possible_N)*len(possible_n))
    posterior_dist = {}
    
    p_range = tqdm(product(possible_N, possible_n)) if print_flag else product(possible_N, possible_n)
    marginal_likelihood = 0
    for N, n in p_range:
        marginal_likelihood += polls_conditional_prob(N, n, *polls)*p_N_n
        
    p_range = tqdm(product(possible_N, possible_n)) if print_flag else product(possible_N, possible_n)  
    for N, n in p_range:
        prior = polls_conditional_prob(N, n, *polls) * p_N_n
        posterior_dist[(N, n)] = prior/marginal_likelihood
    
    

def run_trial(print_flag = True):
    pop_pc = round(rd.gauss(0.5, 0.005), 5)
    DEM = 1
    REP = 0
    pop_total = 1000000
    pop = [DEM]*int(pop_pc*pop_total) + [REP]*int(round((1-pop_pc), 5)*pop_total)
    
    n_poll = 100
    polls = []
    for i in range(n_poll):
        poll_size = rd.randint(800, 1200)
        polls.append(rd.sample(pop, poll_size))
    
    posterior = calculate_posterior(polls)
        
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
n_trial = 1
print_flag = True

trial_range = range(n_trial) if print_flag else tqdm(range(n_trial))
for trial in trial_range:
    results.append(run_trial(True))
