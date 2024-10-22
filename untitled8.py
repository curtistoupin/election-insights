# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:24:17 2024

@author: curti
"""

from pollbayes import json_p_dist_data
import json
import pyperclip

with open('states_data.txt', 'r') as file:
    states = eval(file.read())


state_name = 'Georgia'
plot_data = json_p_dist_data(state_name, states)
json_data = json.dumps(plot_data)
pyperclip.copy(json_data)
print(plot_data)