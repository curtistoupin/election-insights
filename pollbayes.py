# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 22:41:41 2024

@author: curti
"""

import random as rd
from math import comb as choose
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

def polls_conditional_prob(N, n, *polls):
    p = 1
    for poll in polls:
        p *= (choose(n, sum(poll))*choose(N-n, len(poll)-sum(poll))/choose(N, len(poll)))
    return p

def calculate_posterior(polls, 
                        N_min,
                        N_max,
                        N_resolution = 10000,
                        pc_min = 0.4,
                        pc_max = 0.6,
                        n_resolution = 5000,
                        print_flag = True):
    
    
    possible_N = list(range(int(N_min/N_resolution)*N_resolution,
                           int(N_max/N_resolution)*N_resolution + N_resolution,
                           N_resolution))
    total_N_n_pairs = 0
    marginal_likelihood = 0
    posterior_dist = {}
    for N in tqdm(possible_N):
        possible_n = list(range(int(pc_min*N/n_resolution)*n_resolution,
                                int(pc_max*N/n_resolution)*n_resolution,
                                n_resolution))
        total_N_n_pairs += len(possible_n)
        for n in possible_n:
            marginal_likelihood += polls_conditional_prob(N, n, *polls)
            prior = polls_conditional_prob(N, n, *polls)
            posterior_dist[(N, n)] = prior
        
    p_N_n = 1/total_N_n_pairs
    marginal_likelihood *= p_N_n
    for k,v in posterior_dist.items():
        posterior_dist[k] = v*p_N_n/marginal_likelihood
    return posterior_dist

def get_p_dist(posterior):
    p_dist = {}
    for k,v in posterior.items():
        pc = int(1000*k[1]/k[0])/10
        p_dist[pc] = p_dist.get(pc, 0) + v
    return p_dist

def get_cts_ci(dist, confidence = 0.95):
    line_height = 1
    total = 0
    while total < confidence:
        remaining = {k:v for k,v in dist.items() if v < line_height}
        line_height = max([v for k,v in remaining.items()])
        inf = min([k for k,v in dist.items() if v >= line_height])
        sup = max([k for k,v in dist.items() if v >= line_height])
        total = sum([v for k,v in dist.items() if k >= inf and k <= sup])
    return inf, sup, total

def get_ci(dist, confidence = 0.95):
    by_density = sorted([(k,v) for k,v in dist.items()], key=lambda x: x[1], reverse=True)
    for i in range(len(by_density)):
        if sum([pair[1] for pair in by_density[:i]]) >= confidence:
            break
    return sorted(by_density[:i], key = lambda x: x[0])

def plot_p_dist(p_dist, ci=None, state_name=None):
    keys = list(p_dist.keys())
    values = list(p_dist.values())
    
    # Set default value for ci if it's None
    if ci is None:
        ci = []
    
    red_keys = [key for key in p_dist if key < 50]

    # Set colors for each bar based on whether the key is in red_keys
    colors = ['red' if key in red_keys else 'blue' for key in keys]

    # Define the width of the bars
    bar_width = 0.1

    # Adjust the x positions to align the left side of the bars
    adjusted_keys = [k for k in keys]

    # Create the bar chart with adjusted x positions
    plt.bar(adjusted_keys, values, color=colors, width=bar_width, align='edge')
    
    p_dem_win = round(100 * sum([v for k, v in p_dist.items() if k >= 50]), 1)
    
    # Add labels and title
    plt.xlabel('Democrat Vote Share')
    plt.ylabel('Probability Density')
    plt.title(('' if state_name is None else f'{state_name} ') + f'Vote Share -- {p_dem_win}% Chance of Democrat Win')
    
    # Show the plot
    plt.show()

def json_p_dist_data(state_name, states_data):
    p_dist = states_data[state_name]['p_dist']
    # Sort the dictionary by keys (x-values)
    sorted_p_dist = dict(sorted(p_dist.items()))
    
    keys = list(sorted_p_dist.keys())
    values = list(sorted_p_dist.values())

    red_keys = [key for key in sorted_p_dist if key < 50]
    DEM = 'Harris'
    REP = 'Trump'
    # Set colors based on whether the key is in red_keys
    colors = ['red' if key in red_keys else 'blue' for key in keys]

    p_dem_win = round(100 * sum([v for k, v in sorted_p_dist.items() if k >= 50]), 1)
    highest_p = max(p_dist, key=p_dist.get)
    highest_p = max(highest_p, 100-highest_p)
    highest_p_winner = REP if highest_p < 50 else DEM
    winner = REP if p_dem_win < 50 else DEM
    winner_p = max(p_dem_win, 100-p_dem_win)
    ci = get_cts_ci(p_dist)
    winner_possessive = winner + ("'" if winner[-1] == "s" else "'s")
    winner_ci_min = ci[0] if winner == DEM else 100 - ci[1]
    winner_ci_max = ci[1] if winner == DEM else 100 - ci[0]
    return {
        'keys': keys,
        'values': values,
        'colors': colors,
        'ci_min': ci[0],
        'ci_max': ci[1],
        'ci_p': round(100*ci[2], 1),
        'p_dem_win': p_dem_win,
        'title': f"2024 Presidential Election Forecast for {state_name}",
        'projection': f"There is a projected {winner_p}% chance that {winner} wins {state_name}.",
        'most_likely': f"The most likely outcome is {highest_p_winner} winning with {highest_p}% of the head-to-head vote share.",
        'confidence': f"With {round(100*ci[2], 1)}% confidence, {winner_possessive} head-to-head vote share will be between {winner_ci_min}% and {winner_ci_max}%.",
        'state_name': state_name,
        'abbrev': states_data[state_name]['abbrev']
    }
