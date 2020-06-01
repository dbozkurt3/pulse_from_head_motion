"""
Gets the ground truth BPM from an ECG file
"""

import sys
import numpy as np
import heartpy as hp
import matplotlib.pyplot as plt
import json

# Pulse oximeter sample rate = 60 Hz
sample_rate = 60.0
  
# Opening JSON file 
f = open(sys.argv[1]) 
 
# Returns JSON object as a dictionary 
data = json.load(f) 
  
# Iterating through the json list 
y = np.empty(len(data['/FullPackage']))

for i in range(len(data['/FullPackage'])):
    y[i] = data['/FullPackage'][i]['Value']['waveform']

working_data, measures = hp.process(y, sample_rate)
print(measures['bpm'])

hp.plotter(working_data, measures)

# Closing file 
f.close() 

