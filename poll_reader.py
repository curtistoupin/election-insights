# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:58:58 2024

@author: curti
"""

import pandas as pd
import os
from pollbayes import polls_conditional_prob, calculate_posterior, get_p_dist, get_ci, plot_p_dist
from lxml import etree, html
import pyperclip

DEM = 1
REP = 0

df = pd.read_csv(f'{os.path.dirname(os.path.abspath(__file__))}/polls.csv')
df['unq'] = [str(df.poll_id[i]) + str(df.question_id[i]) for i in df.index]
df['start_date'] = ['20' + '-'.join([df.start_date[i].split('/')[-1]] + [s.zfill(2) for s in df.start_date[i].split('/')[:2]]) for i in df.index]
df['end_date'] = ['20' + '-'.join([df.end_date[i].split('/')[-1]] + [s.zfill(2) for s in df.end_date[i].split('/')[:2]]) for i in df.index]
df = df.loc[[i for i in df.index if df.start_date[i] > '2024-09-10']]
df = df.loc[[i for i in df.index if isinstance(df.state[i], str)]]
poll_df = df.copy()
# states = {df.state[i]: {'name': df.state[i],
#                         'polls': [],
#                         'votes': None} for i in df.index}
with open('states_data.txt', 'r') as file:
    states = eval(file.read())
df = df.loc[[i for i in df.index if df.numeric_grade[i] >= 2.3]]
good_poll_df = df.copy()
df = df.loc[[i for i in df.index if df.office_type[i] == 'U.S. President']]
df = df.loc[[i for i in df.index if df.candidate_name[i] in ['Kamala Harris', 'Donald Trump']]]

for state_name in []:
    print('-----------', state_name, '------------')
    pa = df.loc[[i for i in df.index if df.state[i] == state_name]]
    
    for uid in set(pa.unq):
        d = pa.loc[[i for i in pa.index if pa.candidate_name[i] == 'Kamala Harris' and pa.unq[i] == uid]]
        r = pa.loc[[i for i in pa.index if pa.candidate_name[i] == 'Donald Trump' and pa.unq[i] == uid]]
        poll = [DEM]*(int(d.sample_size.iloc[0]*d.pct.iloc[0]/100 + 1)) + [REP]*(int(r.sample_size.iloc[0]*r.pct.iloc[0]/100 + 1))
        states[state_name]['polls'].append(poll)
        
    posterior = calculate_posterior(states[state_name]['polls'],
                                    states[state_name]['pop_min'],
                                    states[state_name]['pop_max'],
                                    10000,
                                    states[state_name]['pc_min'],
                                    states[state_name]['pc_max'],
                                    5000,
                                    True)
    states[state_name]['posterior'] = posterior
    p_dist = get_p_dist(posterior)
    states[state_name]['p_dist'] = p_dist
    plot_p_dist(p_dist, get_ci(p_dist), state_name)
    with open('states_data.txt', 'w') as file:
        file.write(str(states))
        
poll_state = 'Wisconsin'

state_df = df.loc[[i for i in df.index if df.state[i] == poll_state]]
pop_key = {'lv': 'likely voters',
           'rv': 'registered voters'}
month_key = {'01': 'Jan',
             '02': 'Feb',
             '03': 'Mar',
             '04': 'Apr',
             '05': 'May',
             '06': 'Jun',
             '07': 'Jul',
             '08': 'Aug',
             '09': 'Sep',
             '10': 'Oct',
             '11': 'Nov',
             '12': 'Dec'}
state_polls = []
for unq in set(state_df.unq):
    poll_df = state_df.loc[[i for i in state_df.index if state_df.unq[i] == unq]]
    sample_size = int(poll_df.sample_size.iloc[0])
    population = pop_key[poll_df.population.iloc[0]]
    start_year, start_month, start_day = poll_df.start_date.iloc[0].split('-')
    end_year, end_month, end_day = poll_df.end_date.iloc[0].split('-')
    if start_year == end_year and start_month == end_month:
        poll_date = month_key[start_month] + ' ' + start_day + ' - ' + end_day
    else:
        poll_date = month_key[start_month] + ' ' + start_day + ' - ' + month_key[end_month] + ' ' + end_day

    pollster = poll_df.pollster.iloc[0]
    dem_candidate = poll_df[poll_df.party == 'DEM'].candidate_name.iloc[0]
    rep_candidate = poll_df[poll_df.party == 'REP'].candidate_name.iloc[0]
    dem_vote = poll_df[poll_df.party == 'DEM'].pct.iloc[0]
    if dem_vote == int(dem_vote):
        dem_vote = int(dem_vote)
    rep_vote = poll_df[poll_df.party == 'REP'].pct.iloc[0]
    if rep_vote == int(rep_vote):
        rep_vote = int(rep_vote)
    dem_head_to_head = round(100*dem_vote/(rep_vote + dem_vote), 1)
    rep_head_to_head = round(100*rep_vote/(rep_vote + dem_vote), 1)
    pollster_rating = poll_df.numeric_grade.iloc[0]
    pr_int = pollster_rating//1
    pr_frac = pollster_rating - pr_int
    pr_frac += 0.25*(1-2*pr_frac) if pr_frac > 0 else pr_frac
    pollster_rating_visual = pr_int + pr_frac
    if pollster_rating_visual < 2:
        break
    state_polls.append({
        'pollster_name': pollster,
        'poll_date': poll_date,
        'pollster_rating': pollster_rating,
        'pollster_rating_visual': pollster_rating_visual,
        'dem_candidate': dem_candidate.split(' ')[-1],
        'dem_vote': f"{dem_vote}%",
        'rep_candidate': rep_candidate.split(' ')[-1],
        'rep_vote': f"{rep_vote}%",
        'dem_head_to_head': f"{dem_head_to_head}%",
        'rep_head_to_head': f"{rep_head_to_head}%",
        'date': poll_df.end_date.iloc[0]})
    
def generate_table_body(poll_data):
    # Create the tbody element
    tbody = etree.Element("tbody")

    # Loop through poll data to create rows and cells
    for poll in poll_data:
        # Create a new row (tr)
        tr = etree.SubElement(tbody, "tr")

        # Add pollster name (td)
        pollster_name_td = etree.SubElement(tr, "td")
        pollster_name_td.text = poll['pollster_name']

        # Add poll date (td)
        poll_date_td = etree.SubElement(tr, "td")
        poll_date_td.text = poll['poll_date']

        # Add pollster rating (td with stars-container div)
        pollster_rating_td = etree.SubElement(tr, "td")
        stars_div = etree.SubElement(pollster_rating_td, "div", attrib={"class": "stars-container", "data-rating": str(poll['pollster_rating_visual']), "title": f"{poll['pollster_rating']}/3 Stars"})

        # Add Popular Vote for Democrat
        dem_vote_td = etree.SubElement(tr, "td")
        dem_vote_div = etree.SubElement(dem_vote_td, "div", attrib={"class": "vote-container"})
        dem_name_span = etree.SubElement(dem_vote_div, "span", attrib={"class": "candidate-name"})
        dem_name_span.text = poll['dem_candidate']
        dem_vote_span = etree.SubElement(dem_vote_div, "span", attrib={"class": "dem-vote"})
        dem_vote_span.text = poll['dem_vote']

        # Add Popular Vote for Republican
        rep_vote_td = etree.SubElement(tr, "td")
        rep_vote_div = etree.SubElement(rep_vote_td, "div", attrib={"class": "vote-container"})
        rep_name_span = etree.SubElement(rep_vote_div, "span", attrib={"class": "candidate-name"})
        rep_name_span.text = poll['rep_candidate']
        rep_vote_span = etree.SubElement(rep_vote_div, "span", attrib={"class": "rep-vote"})
        rep_vote_span.text = poll['rep_vote']

        # Add Head to Head for Democrat
        dem_head_to_head_td = etree.SubElement(tr, "td")
        dem_head_to_head_div = etree.SubElement(dem_head_to_head_td, "div", attrib={"class": "vote-container"})
        dem_head_to_head_name = etree.SubElement(dem_head_to_head_div, "span", attrib={"class": "candidate-name"})
        dem_head_to_head_name.text = poll['dem_candidate']
        dem_head_to_head_vote = etree.SubElement(dem_head_to_head_div, "span", attrib={"class": "dem-vote"})
        dem_head_to_head_vote.text = poll['dem_head_to_head']

        # Add Head to Head for Republican
        rep_head_to_head_td = etree.SubElement(tr, "td")
        rep_head_to_head_div = etree.SubElement(rep_head_to_head_td, "div", attrib={"class": "vote-container"})
        rep_head_to_head_name = etree.SubElement(rep_head_to_head_div, "span", attrib={"class": "candidate-name"})
        rep_head_to_head_name.text = poll['rep_candidate']
        rep_head_to_head_vote = etree.SubElement(rep_head_to_head_div, "span", attrib={"class": "rep-vote"})
        rep_head_to_head_vote.text = poll['rep_head_to_head']

    # Convert the lxml ElementTree to a string with pretty print
    return html.tostring(tbody, pretty_print=True).decode("utf-8")

state_polls = reversed(sorted(state_polls, key=lambda x: x['date']))
tbody_html = generate_table_body(state_polls)
pyperclip.copy(tbody_html)

# for state, dat in states.items():
#     if dat['safe_blue']:
#         states[state]['p_dem_win'] = 1
#     elif dat['safe_red']:
#         states[state]['p_dem_win'] = 0
#     else:
#         states[state]['p_dem_win'] = sum([v for k,v in dat['p_dist'].items() if k >= 50])
# with open('states_data.txt', 'w') as file:
#     file.write(str(states))