# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:27:22 2024

@author: theov
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:50:51 2023

@author: tvanneau
"""

#loading needed toolboxes 
import mne
import numpy as np
import copy
import tkinter as tk
import pandas as pd
from tkinter import filedialog
import logging
import os
from matplotlib import pyplot as plt
from autoreject import Ransac

#%% Reading BDF files 

# For subject TD: 10158 and 12386 no trigger 99 for JTLA

#Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD')
Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD')

sb = 10

list_file = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/'+ Subjects_name[sb])

file_path_2 = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/'+ Subjects_name[sb]

bdf_files = [s for s in list_file if '.bdf' in s]

#%%

# S1
montage = mne.channels.make_standard_montage('standard_1020')

raw_1 = mne.io.read_raw_bdf(file_path_2 + "/" + bdf_files[0], eog=None, misc = None, stim_channel='auto',
                    exclude = (), infer_types = True, preload= True)
raw_1 = raw_1.set_montage(montage, on_missing='ignore')
events_1 = mne.find_events(raw_1, stim_channel="Status", shortest_event = 1)

raw_2 = mne.io.read_raw_bdf(file_path_2 + "/" + bdf_files[1], eog=None, misc = None, stim_channel='auto',
                    exclude = (), infer_types = True, preload= True)
raw_2 = raw_2.set_montage(montage, on_missing='ignore')
events_2 = mne.find_events(raw_2, stim_channel="Status", shortest_event = 1)

raw_3 = mne.io.read_raw_bdf(file_path_2 + "/" + bdf_files[2], eog=None, misc = None, stim_channel='auto',
                    exclude = (), infer_types = True, preload= True)
raw_3 = raw_3.set_montage(montage, on_missing='ignore')
events_3 = mne.find_events(raw_3, stim_channel="Status", shortest_event = 1)

raw_4 = mne.io.read_raw_bdf(file_path_2 + "/" + bdf_files[3], eog=None, misc = None, stim_channel='auto',
                    exclude = (), infer_types = True, preload= True)
raw_4 = raw_4.set_montage(montage, on_missing='ignore')
events_4 = mne.find_events(raw_4, stim_channel="Status", shortest_event = 1)

raw_5 = mne.io.read_raw_bdf(file_path_2 + "/" + bdf_files[4], eog=None, misc = None, stim_channel='auto',
                    exclude = (), infer_types = True, preload= True)
raw_5 = raw_5.set_montage(montage, on_missing='ignore')
events_5 = mne.find_events(raw_5, stim_channel="Status", shortest_event = 1)

raw_6 = mne.io.read_raw_bdf(file_path_2 + "/" + bdf_files[5], eog=None, misc = None, stim_channel='auto',
                    exclude = (), infer_types = True, preload= True)
raw_6 = raw_6.set_montage(montage, on_missing='ignore')
events_6 = mne.find_events(raw_6, stim_channel="Status", shortest_event = 1)

raw_7 = mne.io.read_raw_bdf(file_path_2 + "/" + bdf_files[6], eog=None, misc = None, stim_channel='auto',
                    exclude = (), infer_types = True, preload= True)
raw_7 = raw_7.set_montage(montage, on_missing='ignore')
events_7 = mne.find_events(raw_7, stim_channel="Status", shortest_event = 1)

raw_8 = mne.io.read_raw_bdf(file_path_2 + "/" + bdf_files[7], eog=None, misc = None, stim_channel='auto',
                    exclude = (), infer_types = True, preload= True)
raw_8 = raw_8.set_montage(montage, on_missing='ignore')
events_8 = mne.find_events(raw_8, stim_channel="Status", shortest_event = 1)



# raw_9 = mne.io.read_raw_bdf(file_path_2 + "/" + bdf_files[8], eog=None, misc = None, stim_channel='auto',
#                     exclude = (), infer_types = True, preload= True)
# raw_9 = raw_9.set_montage(montage, on_missing='ignore')
# events_9 = mne.find_events(raw_9, stim_channel="Status", shortest_event = 1)


# raw_10 = mne.io.read_raw_bdf(file_path_2 + "/" + bdf_files[9], eog=None, misc = None, stim_channel='auto',
#                     exclude = (), infer_types = True, preload= True)
# raw_10 = raw_10.set_montage(montage, on_missing='ignore')
# events_10 = mne.find_events(raw_10, stim_channel="Status", shortest_event = 1)


# Filter

#Low pass (selects data below 30Hz, not relevant here)
lowpass_epochs = 30
highpass_epochs = 0.1

raw_f1 = raw_1.copy().filter(l_freq=highpass_epochs, h_freq=lowpass_epochs)
raw_f2 = raw_2.copy().filter(l_freq=highpass_epochs, h_freq=lowpass_epochs)
raw_f3 = raw_3.copy().filter(l_freq=highpass_epochs, h_freq=lowpass_epochs)
raw_f4 = raw_4.copy().filter(l_freq=highpass_epochs, h_freq=lowpass_epochs)
raw_f5 = raw_5.copy().filter(l_freq=highpass_epochs, h_freq=lowpass_epochs)
raw_f6 = raw_6.copy().filter(l_freq=highpass_epochs, h_freq=lowpass_epochs)
raw_f7 = raw_7.copy().filter(l_freq=highpass_epochs, h_freq=lowpass_epochs)
raw_f8 = raw_8.copy().filter(l_freq=highpass_epochs, h_freq=lowpass_epochs)

# raw_f9 = raw_9.copy().filter(l_freq=highpass_epochs, h_freq=lowpass_epochs)
# raw_f10 = raw_10.copy().filter(l_freq=highpass_epochs, h_freq=lowpass_epochs)


# # 1-5 raws files

# raw, events= mne.concatenate_raws([raw_f1, raw_f2,raw_f3,raw_f4,raw_f5], preload=True, 
#                                     events_list=[events_1, events_2, events_3, events_4, events_5])


# # 1-6 raws files

# raw, events= mne.concatenate_raws([raw_f1, raw_f2,raw_f3,raw_f4,raw_f5,raw_f6], preload=True, 
#                                     events_list=[events_1, events_2, events_3, events_4, events_5, events_6])


# # 1-7 raws files

# raw, events= mne.concatenate_raws([raw_f1, raw_f2,raw_f3,raw_f4,raw_f5,raw_f6,raw_f7], preload=True, 
#                                     events_list=[events_1, events_2, events_3, events_4, events_5, events_6, events_7])



# # 1-8 raws files

raw, events= mne.concatenate_raws([raw_f1, raw_f2,raw_f3,raw_f4,raw_f5,raw_f6,raw_f7,raw_f8], preload=True, 
                                    events_list=[events_1, events_2, events_3, events_4, events_5, events_6, events_7,events_8])


# # 1 -9 raws files

# raw, events= mne.concatenate_raws([raw_f1, raw_f2,raw_f3,raw_f4,raw_f5,raw_f6,raw_f7,raw_f8,raw_f9], preload=True, 
#                                     events_list=[events_1, events_2, events_3, events_4, events_5, events_6, events_7,events_8,
#                                                   events_9])

# 1 -10 raws files

# raw, events= mne.concatenate_raws([raw_f1, raw_f2,raw_f3,raw_f4,raw_f5,raw_f6,raw_f7,raw_f8,raw_f9,raw_f10], preload=True, 
#                                     events_list=[events_1, events_2, events_3, events_4, events_5, events_6, events_7,events_8,
#                                                   events_9,events_10])

#

# # For subject without trigger 99 for JTLA
# for i in range(0,len(events)-1,1):
#     if events[i,2] == 42 and events[i+1,2] == 42:
#         events[i,2]=99
#         events[i+1,2]=99
#     elif events[i,2] == 42 and events[i+1,2] == 53:
#         events[i,2]=99
#         events[i+1,2]=110
#     elif events[i,2]== 53:
#         events[i,2]=110
#     elif events[i+1,2] == 42 and events[i+2,2] == 100 and (events[i,2] == 53 or events[i-1,2] == 53):
#         events[i,2] = 99
#         events[i+1,2] = 99
#     elif events[i+1,2] == 42 and events[i+2,2] == 100 and (events[i,2] != 31 or events[i-1,2] != 31):
#         events[i,2] = 99
#         events[i+1,2] = 99
#     elif events[i,2] == 42 and events[i+1,2] == 19:
#         events[i,2] = 99

events[events==42] = 31
events[events==52] = 41
events[events==50] = 39
events[events==110] = 99

events_trial = events[np.where((events[:,2]==31) | (events[:,2]==41) | (events[:,2]==99) | (events[:,2]==39) | (events[:,2]==101))] # Triggers with the video number

# Delete consecutive 101
for i in range (len(events_trial)-2,-1,-1):
    if events_trial[i,2]==101 and events_trial[i+1,2]==101:
        events_trial = np.delete(events_trial, i,0)
        
# Delete trial of less than 35 seconds 
events_start = events_trial[np.where(events_trial[:,2]==101)]
test_int = np.zeros((len(events_start)-1,1))
for i in range(0,len(events_start)-1,1):
    test_int[i,0] = events_start[i+1,0] - events_start[i,0]
    
False_trial = events_start[np.where(test_int<(512*35))[0]+1]
    
for i in range(0,len(False_trial),1):
    events_trial = np.delete(events_trial, events_trial[:,0]==False_trial[i,0], axis=0)

# Replace 101 triggers depending on stimulation type
for i in range (0,len(events_trial)-1,1):
    if events_trial[i,2]==101 and all(events_trial[(i+1):(i+40),2]==31):
        events_trial[i,2] = 1
    if events_trial[i,2]==101 and all(events_trial[(i+1):(i+40),2]==41):
        events_trial[i,2] = 2
    if events_trial[i,2]==101 and all(events_trial[(i+1):(i+40),2]==99):
        events_trial[i,2] = 3
    if events_trial[i,2]==101 and all(events_trial[(i+1):(i+40),2]==39):
        events_trial[i,2] = 4
    
event_dict_TRF = {
    "ISO": 1,
    "JTSM": 2,
    "JTLA": 3,
    "RDM": 4}

tmin_epochs = -5.0
tmax_epochs = 40.0

#Create epoch file without baseline correction and decim=4 to resample the data at 128Hz
epochs = mne.Epochs(raw, events_trial, tmin=tmin_epochs, tmax=tmax_epochs, event_id = event_dict_TRF, baseline= None ,detrend=1, preload=True, decim=4)

epochs.set_channel_types({"EXG1":"emg"})
epochs.set_channel_types({"EXG2":"emg"})
epochs.set_channel_types({"EXG3":"emg"})
epochs.set_channel_types({"EXG4":"emg"})
epochs.set_channel_types({"EXG5":"emg"})
epochs.set_channel_types({"EXG6":"emg"})
epochs.set_channel_types({"EXG7":"emg"})
epochs.set_channel_types({"EXG8":"emg"})

#%% First use of autoreject package to detect bads channel before CAR

#Without interpolation to not bias the ICA
import autoreject
ar = autoreject.AutoReject(n_interpolate= [0],
                           n_jobs = 1,
                           verbose = True)

ar.fit(epochs)

reject_log = ar.get_reject_log(epochs)

#%% Visualisation tools

# Visualize the dropped epochs
epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))

# Visualize the rejected log
reject_log.plot('horizontal')

#%% Drop bads channels if more than 50% epochs dropper in this experiment

#Percent of dropped epochs per channels
Detect_badschan = ( np.sum(reject_log.labels[:,:64], axis=0) / len(epochs) ) * 100

#Mark bad channel if more than 50% of bads epochs dropped for CAR
bads_channels = np.array(epochs.ch_names)[np.where(Detect_badschan>50)[0].astype(int)]
for i in range (0,len(bads_channels),1):
    epochs.info['bads'].append(bads_channels[i])
    
# Bads channels dropped
epochs.info['bads']

#epochs_lowpass.plot()

#%% CAR without bads channels

epochs.set_eeg_reference(ref_channels='average') # CAR

#%%#INDEPENDENT COMPONENTS ANALYSIS (ICA)

epochs_ica = epochs.filter(l_freq=1., h_freq=None)
#ICA est sensible aux dérives basse fréquence donc 1Hz + charge données

#Parameters ICA
n_components = None #0.99 # % de vraiance expliquée ou alors le nombre d'électrodes
max_pca_components = 64 # disparaitra dans une future version de MNE, nombre de PCA a faire avant ICA
random_state = 42 #* pour que l'ICA donne la même chose sur les mêmes données
method = 'fastica' # méthode de l'ICA (si 'picard' nécessite pip install python-picard)
fit_params = None # fastica_it=5 paramètre lié à la methode picard
max_iter = 1000 # nombre d'iterations de l'ICA

ica = mne.preprocessing.ICA(n_components=n_components, method = method, max_iter = max_iter, fit_params= fit_params, random_state=random_state)

ica.fit(epochs_ica)
ica.plot_sources(epochs)
ica.plot_components()

#%%APPLICATION ICA COMMUN AVERAGE REFERENCE
ica.apply(epochs)
ica.apply(raw)

#%% Second use of autoreject package after ICA to remove bads epochs

ar_2 = autoreject.AutoReject(n_interpolate= [1, 4, 32],
                           n_jobs = 1,
                           verbose = True)

ar_2.fit(epochs)

epochs_ar_2, reject_log_2 = ar_2.transform(epochs, return_log=True)

#%% Visualisation tools

# Visualize the dropped epochs
epochs[reject_log_2.bad_epochs].plot(scalings=dict(eeg=100e-6))

# Visualize the rejected log
reject_log_2.plot('horizontal')

epochs_ar_2.info["bads"]

#%% Interpolate bads channels
epochs_ar_2.interpolate_bads(reset_bads=True, mode='accurate')
epochs_ar_2.info["bads"]
#%% Calculating lowpass.highpass epochs files

# Lowpass EEG 
lowpass_epochs = 3
highpass_epochs = 0.5

epochs_lowpass = epochs_ar_2.copy().filter(l_freq=highpass_epochs, h_freq=lowpass_epochs)

# Highpass EEG
lowpass_epochs = 30
highpass_epochs = 5

epochs_highpass = epochs_ar_2.copy().filter(l_freq=highpass_epochs, h_freq=lowpass_epochs)

#%% ERP epochs

event_dict_ERP = {
    "ISO": 31,
    "JTSM": 41,
    "JTLA": 99,
    "RDM": 39}

tmin_epochs_ERP = -0.2
tmax_epochs_ERP = 0.5

epochs_ERP = mne.Epochs(raw, events_trial, tmin=tmin_epochs_ERP, tmax=tmax_epochs_ERP, event_id = event_dict_ERP, baseline=(None,0), detrend=1, preload=True, decim=4)

epochs_ERP.set_channel_types({"EXG1":"emg"})
epochs_ERP.set_channel_types({"EXG2":"emg"})
epochs_ERP.set_channel_types({"EXG3":"emg"})
epochs_ERP.set_channel_types({"EXG4":"emg"})
epochs_ERP.set_channel_types({"EXG5":"emg"})
epochs_ERP.set_channel_types({"EXG6":"emg"})
epochs_ERP.set_channel_types({"EXG7":"emg"})
epochs_ERP.set_channel_types({"EXG8":"emg"})

ar_3 = autoreject.AutoReject(n_interpolate= [1, 4, 32],
                           n_jobs = 1,
                           verbose = True)
ar_3.fit(epochs_ERP)

epochs_ERP_ar, reject_log_3 = ar_3.transform(epochs_ERP, return_log=True)

ISO_ar = epochs_ERP_ar['ISO'].average()
JTSM_ar = epochs_ERP_ar['JTSM'].average()
JTLA_ar = epochs_ERP_ar['JTLA'].average()
RDM_ar = epochs_ERP_ar['RDM'].average()

Diff = mne.viz.plot_compare_evokeds(dict(ISO=ISO_ar, JTSM=JTSM_ar, JTLA=JTLA_ar, RDM=RDM_ar), picks=['Cz'])

epochs_ERP_ar.save('Epochs_ERP_AR_' + Subjects_name[sb] + '-epo.fif')

#%% Calculate stimulation vector and EEG/Stim matrices

tmax_epochs = 35.0
Iso_events = epochs_ar_2['ISO'].events 
Stim_ISO = []
for i in range(0,len(Iso_events),1):
    tmin = Iso_events[i,0]-1
    tmax = tmin + (tmax_epochs * raw.info['sfreq'])
    Timing_iso = events_trial[(events_trial[:,0]>tmin) & (events_trial[:,0]<tmax)]
    Dis_vect = ((Timing_iso[:,0]) - (Timing_iso[0,0])) / 4
    Vector = np.zeros((int(tmax_epochs * epochs_ar_2.info['sfreq']),1))
    for i in range(1,len(Dis_vect),1):
        Vector[round(Dis_vect[i]),0] = 1
    Stim_ISO.append(Vector)
    
JTSM_events = epochs_ar_2['JTSM'].events 
Stim_JTSM = []
for i in range(0,len(JTSM_events),1):
    tmin = JTSM_events[i,0]-1
    tmax = tmin + (tmax_epochs * raw.info['sfreq'])
    Timing_JTSM = events_trial[(events_trial[:,0]>tmin) & (events_trial[:,0]<tmax)]
    Dis_vect = ((Timing_JTSM[:,0]) - (Timing_JTSM[0,0])) / 4
    Vector = np.zeros((int(tmax_epochs * epochs_ar_2.info['sfreq']),1))
    for i in range(1,len(Dis_vect),1):
        Vector[round(Dis_vect[i]),0] = 1
    Stim_JTSM.append(Vector)
    
JTLA_events = epochs_ar_2['JTLA'].events 
Stim_JTLA = []
for i in range(0,len(JTLA_events),1):
    tmin = JTLA_events[i,0]-1
    tmax = tmin + (tmax_epochs * raw.info['sfreq'])
    Timing_JTLA = events_trial[(events_trial[:,0]>tmin) & (events_trial[:,0]<tmax)]
    Dis_vect = ((Timing_JTLA[:,0]) - (Timing_JTLA[0,0])) / 4
    Vector = np.zeros((int(tmax_epochs * epochs_ar_2.info['sfreq']),1))
    for i in range(1,len(Dis_vect),1):
        Vector[round(Dis_vect[i]),0] = 1
    Stim_JTLA.append(Vector)
    
RDM_events = epochs_ar_2['RDM'].events 
Stim_RDM = []
for i in range(0,len(RDM_events),1):
    tmin = RDM_events[i,0]-1
    tmax = tmin + (tmax_epochs * raw.info['sfreq'])
    Timing_RDM = events_trial[(events_trial[:,0]>tmin) & (events_trial[:,0]<tmax)]
    Dis_vect = ((Timing_RDM[:,0]) - (Timing_RDM[0,0])) / 4
    Vector = np.zeros((int(tmax_epochs * epochs_ar_2.info['sfreq']),1))
    for i in range(1,len(Dis_vect),1):
        Vector[round(Dis_vect[i]),0] = 1
    Stim_RDM.append(Vector)
    
EEG_ISO=[]
for i in range(0, len(epochs_ar_2['ISO']), 1):
    EEG_ISO.append(epochs_ar_2['ISO']._data[i,:64,640:5120].T)
    
EEG_JTSM=[]
for i in range(0, len(epochs_ar_2['JTSM']), 1):
    EEG_JTSM.append(epochs_ar_2['JTSM']._data[i,:64,640:5120].T)
    
EEG_JTLA=[]
for i in range(0, len(epochs_ar_2['JTLA']), 1):
    EEG_JTLA.append(epochs_ar_2['JTLA']._data[i,:64,640:5120].T)
    
EEG_RDM=[]
for i in range(0, len(epochs_ar_2['RDM']), 1):
    EEG_RDM.append(epochs_ar_2['RDM']._data[i,:64,640:5120].T)
    
    
EEG_ISO_lowpass=[]
for i in range(0, len(epochs_lowpass['ISO']), 1):
    EEG_ISO_lowpass.append(epochs_lowpass['ISO']._data[i,:64,640:5120].T)
    
EEG_JTSM_lowpass=[]
for i in range(0, len(epochs_lowpass['JTSM']), 1):
    EEG_JTSM_lowpass.append(epochs_lowpass['JTSM']._data[i,:64,640:5120].T)
    
EEG_JTLA_lowpass=[]
for i in range(0, len(epochs_lowpass['JTLA']), 1):
    EEG_JTLA_lowpass.append(epochs_lowpass['JTLA']._data[i,:64,640:5120].T)
    
EEG_RDM_lowpass=[]
for i in range(0, len(epochs_lowpass['RDM']), 1):
    EEG_RDM_lowpass.append(epochs_lowpass['RDM']._data[i,:64,640:5120].T)
    
    
EEG_ISO_highpass=[]
for i in range(0, len(epochs_highpass['ISO']), 1):
    EEG_ISO_highpass.append(epochs_highpass['ISO']._data[i,:64,640:5120].T)
    
EEG_JTSM_highpass=[]
for i in range(0, len(epochs_highpass['JTSM']), 1):
    EEG_JTSM_highpass.append(epochs_highpass['JTSM']._data[i,:64,640:5120].T)
    
EEG_JTLA_highpass=[]
for i in range(0, len(epochs_highpass['JTLA']), 1):
    EEG_JTLA_highpass.append(epochs_highpass['JTLA']._data[i,:64,640:5120].T)
    
EEG_RDM_highpass=[]
for i in range(0, len(epochs_highpass['RDM']), 1):
    EEG_RDM_highpass.append(epochs_highpass['RDM']._data[i,:64,640:5120].T)

import pickle
        
with open('Matrices_TRF_Models_'+Subjects_name[sb]+'.pkl', 'wb') as f:
    pickle.dump([EEG_ISO, EEG_JTSM, EEG_JTLA, EEG_RDM, 
                 EEG_ISO_lowpass, EEG_JTSM_lowpass, EEG_JTLA_lowpass, EEG_RDM_lowpass,
                 EEG_ISO_highpass, EEG_JTSM_highpass, EEG_JTLA_highpass, EEG_RDM_highpass,
                 Stim_ISO, Stim_JTSM, Stim_JTLA, Stim_RDM], f)
 


#%% Import and concatenate individuals matrices

import pickle
import mne
import numpy as np
import os

Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/')

Matrices_EEG_ISO_TD = []
Matrices_EEG_JTSM_TD = []
Matrices_EEG_JTLA_TD = []
Matrices_EEG_RDM_TD = []

Matrices_EEG_Highpass_ISO_TD = []
Matrices_EEG_Highpass_JTSM_TD = []
Matrices_EEG_Highpass_JTLA_TD = []
Matrices_EEG_Highpass_RDM_TD = []

Matrices_EEG_Lowpass_ISO_TD = []
Matrices_EEG_Lowpass_JTSM_TD = []
Matrices_EEG_Lowpass_JTLA_TD = []
Matrices_EEG_Lowpass_RDM_TD = []

Matrices_Stim_ISO_TD = []
Matrices_Stim_JTSM_TD = []
Matrices_Stim_JTLA_TD = []
Matrices_Stim_RDM_TD = []


for sb in range(0,len(Subjects_name),1):
    
    file_path_2 = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/'
    
    with open(file_path_2 + Subjects_name[sb], 'rb') as f:
        [EEG_ISO, EEG_JTSM, EEG_JTLA, EEG_RDM, 
                     EEG_ISO_lowpass, EEG_JTSM_lowpass, EEG_JTLA_lowpass, EEG_RDM_lowpass,
                     EEG_ISO_highpass, EEG_JTSM_highpass, EEG_JTLA_highpass, EEG_RDM_highpass,
                     Stim_ISO, Stim_JTSM, Stim_JTLA, Stim_RDM] = pickle.load(f)
    
    Matrices_EEG_ISO_TD.append(EEG_ISO)
    Matrices_EEG_JTSM_TD.append(EEG_JTSM)
    Matrices_EEG_JTLA_TD.append(EEG_JTLA)
    Matrices_EEG_RDM_TD.append(EEG_RDM)
    
    Matrices_EEG_Highpass_ISO_TD.append(EEG_ISO_highpass)
    Matrices_EEG_Highpass_JTSM_TD.append(EEG_JTSM_highpass)
    Matrices_EEG_Highpass_JTLA_TD.append(EEG_JTLA_highpass)
    Matrices_EEG_Highpass_RDM_TD.append(EEG_RDM_highpass)
    
    Matrices_EEG_Lowpass_ISO_TD.append(EEG_ISO_lowpass)
    Matrices_EEG_Lowpass_JTSM_TD.append(EEG_JTSM_lowpass)
    Matrices_EEG_Lowpass_JTLA_TD.append(EEG_JTLA_lowpass)
    Matrices_EEG_Lowpass_RDM_TD.append(EEG_RDM_lowpass)
    
    Matrices_Stim_ISO_TD.append(Stim_ISO)
    Matrices_Stim_JTSM_TD.append(Stim_JTSM)
    Matrices_Stim_JTLA_TD.append(Stim_JTLA)
    Matrices_Stim_RDM_TD.append(Stim_RDM)


#%%

# ASD 19 Subjects

EEG_ISO_All_ASD = Matrices_EEG_ISO_TD[0] + Matrices_EEG_ISO_TD[1] + Matrices_EEG_ISO_TD[2] + Matrices_EEG_ISO_TD[3] + Matrices_EEG_ISO_TD[4] + Matrices_EEG_ISO_TD[5] + Matrices_EEG_ISO_TD[6] + Matrices_EEG_ISO_TD[7] + Matrices_EEG_ISO_TD[8] + Matrices_EEG_ISO_TD[9] + Matrices_EEG_ISO_TD[10] + Matrices_EEG_ISO_TD[11] + Matrices_EEG_ISO_TD[12] + Matrices_EEG_ISO_TD[13] + Matrices_EEG_ISO_TD[14] + Matrices_EEG_ISO_TD[15] + Matrices_EEG_ISO_TD[16]   + Matrices_EEG_ISO_TD[17]  + Matrices_EEG_ISO_TD[18]   
EEG_JTSM_All_ASD = Matrices_EEG_JTSM_TD[0] + Matrices_EEG_JTSM_TD[1] + Matrices_EEG_JTSM_TD[2] + Matrices_EEG_JTSM_TD[3] + Matrices_EEG_JTSM_TD[4] + Matrices_EEG_JTSM_TD[5] + Matrices_EEG_JTSM_TD[6] + Matrices_EEG_JTSM_TD[7] + Matrices_EEG_JTSM_TD[8] + Matrices_EEG_JTSM_TD[9] + Matrices_EEG_JTSM_TD[10] + Matrices_EEG_JTSM_TD[11] + Matrices_EEG_JTSM_TD[12] + Matrices_EEG_JTSM_TD[13] + Matrices_EEG_JTSM_TD[14] + Matrices_EEG_JTSM_TD[15]  + Matrices_EEG_JTSM_TD[16] + Matrices_EEG_JTSM_TD[17] + Matrices_EEG_JTSM_TD[18]  
EEG_JTLA_All_ASD = Matrices_EEG_JTLA_TD[0] + Matrices_EEG_JTLA_TD[1] + Matrices_EEG_JTLA_TD[2] + Matrices_EEG_JTLA_TD[3] + Matrices_EEG_JTLA_TD[4] + Matrices_EEG_JTLA_TD[5] + Matrices_EEG_JTLA_TD[6] + Matrices_EEG_JTLA_TD[7] + Matrices_EEG_JTLA_TD[8] + Matrices_EEG_JTLA_TD[9] + Matrices_EEG_JTLA_TD[10] + Matrices_EEG_JTLA_TD[11] + Matrices_EEG_JTLA_TD[12] + Matrices_EEG_JTLA_TD[13] + Matrices_EEG_JTLA_TD[14] + Matrices_EEG_JTLA_TD[15] + Matrices_EEG_JTLA_TD[16]  + Matrices_EEG_JTLA_TD[17]  + Matrices_EEG_JTLA_TD[18]    
EEG_RDM_All_ASD = Matrices_EEG_RDM_TD[0] + Matrices_EEG_RDM_TD[1] + Matrices_EEG_RDM_TD[2] + Matrices_EEG_RDM_TD[3] + Matrices_EEG_RDM_TD[4] + Matrices_EEG_RDM_TD[5] + Matrices_EEG_RDM_TD[6] + Matrices_EEG_RDM_TD[7] + Matrices_EEG_RDM_TD[8] + Matrices_EEG_RDM_TD[9] + Matrices_EEG_RDM_TD[10] + Matrices_EEG_RDM_TD[11] + Matrices_EEG_RDM_TD[12] + Matrices_EEG_RDM_TD[13] + Matrices_EEG_RDM_TD[14]  + Matrices_EEG_RDM_TD[15] + Matrices_EEG_RDM_TD[16] + Matrices_EEG_RDM_TD[17] + Matrices_EEG_RDM_TD[18] 

EEG_ISO_Highpass_All_ASD = Matrices_EEG_Highpass_ISO_TD[0] + Matrices_EEG_Highpass_ISO_TD[1] + Matrices_EEG_Highpass_ISO_TD[2] + Matrices_EEG_Highpass_ISO_TD[3] + Matrices_EEG_Highpass_ISO_TD[4] + Matrices_EEG_Highpass_ISO_TD[5] + Matrices_EEG_Highpass_ISO_TD[6] + Matrices_EEG_Highpass_ISO_TD[7] + Matrices_EEG_Highpass_ISO_TD[8] + Matrices_EEG_Highpass_ISO_TD[9] + Matrices_EEG_Highpass_ISO_TD[10] + Matrices_EEG_Highpass_ISO_TD[11] + Matrices_EEG_Highpass_ISO_TD[12] + Matrices_EEG_Highpass_ISO_TD[13] + Matrices_EEG_Highpass_ISO_TD[14] + Matrices_EEG_Highpass_ISO_TD[15]  + Matrices_EEG_Highpass_ISO_TD[16] + Matrices_EEG_Highpass_ISO_TD[17] + Matrices_EEG_Highpass_ISO_TD[18]  
EEG_JTSM_Highpass_All_ASD = Matrices_EEG_Highpass_JTSM_TD[0] + Matrices_EEG_Highpass_JTSM_TD[1] + Matrices_EEG_Highpass_JTSM_TD[2] + Matrices_EEG_Highpass_JTSM_TD[3] + Matrices_EEG_Highpass_JTSM_TD[4] + Matrices_EEG_Highpass_JTSM_TD[5] + Matrices_EEG_Highpass_JTSM_TD[6] + Matrices_EEG_Highpass_JTSM_TD[7] + Matrices_EEG_Highpass_JTSM_TD[8] + Matrices_EEG_Highpass_JTSM_TD[9] + Matrices_EEG_Highpass_JTSM_TD[10] + Matrices_EEG_Highpass_JTSM_TD[11] + Matrices_EEG_Highpass_JTSM_TD[12] + Matrices_EEG_Highpass_JTSM_TD[13] + Matrices_EEG_Highpass_JTSM_TD[14] + Matrices_EEG_Highpass_JTSM_TD[15] + Matrices_EEG_Highpass_JTSM_TD[16]+ Matrices_EEG_Highpass_JTSM_TD[17]+ Matrices_EEG_Highpass_JTSM_TD[18]  
EEG_JTLA_Highpass_All_ASD = Matrices_EEG_Highpass_JTLA_TD[0] + Matrices_EEG_Highpass_JTLA_TD[1] + Matrices_EEG_Highpass_JTLA_TD[2] + Matrices_EEG_Highpass_JTLA_TD[3] + Matrices_EEG_Highpass_JTLA_TD[4] + Matrices_EEG_Highpass_JTLA_TD[5] + Matrices_EEG_Highpass_JTLA_TD[6] + Matrices_EEG_Highpass_JTLA_TD[7] + Matrices_EEG_Highpass_JTLA_TD[8] + Matrices_EEG_Highpass_JTLA_TD[9] + Matrices_EEG_Highpass_JTLA_TD[10] + Matrices_EEG_Highpass_JTLA_TD[11] + Matrices_EEG_Highpass_JTLA_TD[12] + Matrices_EEG_Highpass_JTLA_TD[13] + Matrices_EEG_Highpass_JTLA_TD[14] + Matrices_EEG_Highpass_JTLA_TD[15] + Matrices_EEG_Highpass_JTLA_TD[16]+ Matrices_EEG_Highpass_JTLA_TD[17]+ Matrices_EEG_Highpass_JTLA_TD[18]  
EEG_RDM_Highpass_All_ASD = Matrices_EEG_Highpass_RDM_TD[0] + Matrices_EEG_Highpass_RDM_TD[1] + Matrices_EEG_Highpass_RDM_TD[2] + Matrices_EEG_Highpass_RDM_TD[3] + Matrices_EEG_Highpass_RDM_TD[4] + Matrices_EEG_Highpass_RDM_TD[5] + Matrices_EEG_Highpass_RDM_TD[6] + Matrices_EEG_Highpass_RDM_TD[7] + Matrices_EEG_Highpass_RDM_TD[8] + Matrices_EEG_Highpass_RDM_TD[9] + Matrices_EEG_Highpass_RDM_TD[10] + Matrices_EEG_Highpass_RDM_TD[11] + Matrices_EEG_Highpass_RDM_TD[12] + Matrices_EEG_Highpass_RDM_TD[13] + Matrices_EEG_Highpass_RDM_TD[14]  + Matrices_EEG_Highpass_RDM_TD[15]  + Matrices_EEG_Highpass_RDM_TD[16]  + Matrices_EEG_Highpass_RDM_TD[17] + Matrices_EEG_Highpass_RDM_TD[18]    

EEG_ISO_Lowpass_All_ASD = Matrices_EEG_Lowpass_ISO_TD[0] + Matrices_EEG_Lowpass_ISO_TD[1] + Matrices_EEG_Lowpass_ISO_TD[2] + Matrices_EEG_Lowpass_ISO_TD[3] + Matrices_EEG_Lowpass_ISO_TD[4] + Matrices_EEG_Lowpass_ISO_TD[5] + Matrices_EEG_Lowpass_ISO_TD[6] + Matrices_EEG_Lowpass_ISO_TD[7] + Matrices_EEG_Lowpass_ISO_TD[8] + Matrices_EEG_Lowpass_ISO_TD[9] + Matrices_EEG_Lowpass_ISO_TD[10] + Matrices_EEG_Lowpass_ISO_TD[11] + Matrices_EEG_Lowpass_ISO_TD[12] + Matrices_EEG_Lowpass_ISO_TD[13] + Matrices_EEG_Lowpass_ISO_TD[14] + Matrices_EEG_Lowpass_ISO_TD[15] + Matrices_EEG_Lowpass_ISO_TD[16]+ Matrices_EEG_Lowpass_ISO_TD[17]+ Matrices_EEG_Lowpass_ISO_TD[18]  
EEG_JTSM_Lowpass_All_ASD = Matrices_EEG_Lowpass_JTSM_TD[0] + Matrices_EEG_Lowpass_JTSM_TD[1] + Matrices_EEG_Lowpass_JTSM_TD[2] + Matrices_EEG_Lowpass_JTSM_TD[3] + Matrices_EEG_Lowpass_JTSM_TD[4] + Matrices_EEG_Lowpass_JTSM_TD[5] + Matrices_EEG_Lowpass_JTSM_TD[6] + Matrices_EEG_Lowpass_JTSM_TD[7] + Matrices_EEG_Lowpass_JTSM_TD[8] + Matrices_EEG_Lowpass_JTSM_TD[9] + Matrices_EEG_Lowpass_JTSM_TD[10] + Matrices_EEG_Lowpass_JTSM_TD[11] + Matrices_EEG_Lowpass_JTSM_TD[12] + Matrices_EEG_Lowpass_JTSM_TD[13] + Matrices_EEG_Lowpass_JTSM_TD[14] + Matrices_EEG_Lowpass_JTSM_TD[15] + Matrices_EEG_Lowpass_JTSM_TD[16]+ Matrices_EEG_Lowpass_JTSM_TD[17]+ Matrices_EEG_Lowpass_JTSM_TD[18]  
EEG_JTLA_Lowpass_All_ASD = Matrices_EEG_Lowpass_JTLA_TD[0] + Matrices_EEG_Lowpass_JTLA_TD[1] + Matrices_EEG_Lowpass_JTLA_TD[2] + Matrices_EEG_Lowpass_JTLA_TD[3] + Matrices_EEG_Lowpass_JTLA_TD[4] + Matrices_EEG_Lowpass_JTLA_TD[5] + Matrices_EEG_Lowpass_JTLA_TD[6] + Matrices_EEG_Lowpass_JTLA_TD[7] + Matrices_EEG_Lowpass_JTLA_TD[8] + Matrices_EEG_Lowpass_JTLA_TD[9] + Matrices_EEG_Lowpass_JTLA_TD[10] + Matrices_EEG_Lowpass_JTLA_TD[11] + Matrices_EEG_Lowpass_JTLA_TD[12] + Matrices_EEG_Lowpass_JTLA_TD[13] + Matrices_EEG_Lowpass_JTLA_TD[14] + Matrices_EEG_Lowpass_JTLA_TD[15]  + Matrices_EEG_Lowpass_JTLA_TD[16]+ Matrices_EEG_Lowpass_JTLA_TD[17]+ Matrices_EEG_Lowpass_JTLA_TD[18] 
EEG_RDM_Lowpass_All_ASD = Matrices_EEG_Lowpass_RDM_TD[0] + Matrices_EEG_Lowpass_RDM_TD[1] + Matrices_EEG_Lowpass_RDM_TD[2] + Matrices_EEG_Lowpass_RDM_TD[3] + Matrices_EEG_Lowpass_RDM_TD[4] + Matrices_EEG_Lowpass_RDM_TD[5] + Matrices_EEG_Lowpass_RDM_TD[6] + Matrices_EEG_Lowpass_RDM_TD[7] + Matrices_EEG_Lowpass_RDM_TD[8] + Matrices_EEG_Lowpass_RDM_TD[9] + Matrices_EEG_Lowpass_RDM_TD[10] + Matrices_EEG_Lowpass_RDM_TD[11] + Matrices_EEG_Lowpass_RDM_TD[12] + Matrices_EEG_Lowpass_RDM_TD[13] + Matrices_EEG_Lowpass_RDM_TD[14] + Matrices_EEG_Lowpass_RDM_TD[15]+ Matrices_EEG_Lowpass_RDM_TD[16]+ Matrices_EEG_Lowpass_RDM_TD[17]+ Matrices_EEG_Lowpass_RDM_TD[18]

Stim_ISO_All_ASD = Matrices_Stim_ISO_TD[0] + Matrices_Stim_ISO_TD[1] + Matrices_Stim_ISO_TD[2] + Matrices_Stim_ISO_TD[3] + Matrices_Stim_ISO_TD[4] + Matrices_Stim_ISO_TD[5] + Matrices_Stim_ISO_TD[6] + Matrices_Stim_ISO_TD[7] + Matrices_Stim_ISO_TD[8] + Matrices_Stim_ISO_TD[9] + Matrices_Stim_ISO_TD[10] + Matrices_Stim_ISO_TD[11] + Matrices_Stim_ISO_TD[12] + Matrices_Stim_ISO_TD[13] + Matrices_Stim_ISO_TD[14] + Matrices_Stim_ISO_TD[15] + Matrices_Stim_ISO_TD[16]+ Matrices_Stim_ISO_TD[17]+ Matrices_Stim_ISO_TD[18]  
Stim_JTSM_All_ASD = Matrices_Stim_JTSM_TD[0] + Matrices_Stim_JTSM_TD[1] + Matrices_Stim_JTSM_TD[2] + Matrices_Stim_JTSM_TD[3] + Matrices_Stim_JTSM_TD[4] + Matrices_Stim_JTSM_TD[5] + Matrices_Stim_JTSM_TD[6] + Matrices_Stim_JTSM_TD[7] + Matrices_Stim_JTSM_TD[8] + Matrices_Stim_JTSM_TD[9] + Matrices_Stim_JTSM_TD[10] + Matrices_Stim_JTSM_TD[11] + Matrices_Stim_JTSM_TD[12] + Matrices_Stim_JTSM_TD[13] + Matrices_Stim_JTSM_TD[14] + Matrices_Stim_JTSM_TD[15]  + Matrices_Stim_JTSM_TD[16]+ Matrices_Stim_JTSM_TD[17]+ Matrices_Stim_JTSM_TD[18] 
Stim_JTLA_All_ASD = Matrices_Stim_JTLA_TD[0] + Matrices_Stim_JTLA_TD[1] + Matrices_Stim_JTLA_TD[2] + Matrices_Stim_JTLA_TD[3] + Matrices_Stim_JTLA_TD[4] + Matrices_Stim_JTLA_TD[5] + Matrices_Stim_JTLA_TD[6] + Matrices_Stim_JTLA_TD[7] + Matrices_Stim_JTLA_TD[8] + Matrices_Stim_JTLA_TD[9] + Matrices_Stim_JTLA_TD[10] + Matrices_Stim_JTLA_TD[11] + Matrices_Stim_JTLA_TD[12] + Matrices_Stim_JTLA_TD[13] + Matrices_Stim_JTLA_TD[14] + Matrices_Stim_JTLA_TD[15] + Matrices_Stim_JTLA_TD[16]+ Matrices_Stim_JTLA_TD[17]+ Matrices_Stim_JTLA_TD[18]  
Stim_RDM_All_ASD = Matrices_Stim_RDM_TD[0] + Matrices_Stim_RDM_TD[1] + Matrices_Stim_RDM_TD[2] + Matrices_Stim_RDM_TD[3] + Matrices_Stim_RDM_TD[4] + Matrices_Stim_RDM_TD[5] + Matrices_Stim_RDM_TD[6] + Matrices_Stim_RDM_TD[7] + Matrices_Stim_RDM_TD[8] + Matrices_Stim_RDM_TD[9] + Matrices_Stim_RDM_TD[10] + Matrices_Stim_RDM_TD[11] + Matrices_Stim_RDM_TD[12] + Matrices_Stim_RDM_TD[13] + Matrices_Stim_RDM_TD[14] + Matrices_Stim_RDM_TD[15]+ Matrices_Stim_RDM_TD[16]+ Matrices_Stim_RDM_TD[17]+ Matrices_Stim_RDM_TD[18]


# ASD 18 Subjects

EEG_ISO_All_ASD = Matrices_EEG_ISO_TD[0] + Matrices_EEG_ISO_TD[1] + Matrices_EEG_ISO_TD[2] + Matrices_EEG_ISO_TD[3] + Matrices_EEG_ISO_TD[4] + Matrices_EEG_ISO_TD[5] + Matrices_EEG_ISO_TD[6] + Matrices_EEG_ISO_TD[7] + Matrices_EEG_ISO_TD[8] + Matrices_EEG_ISO_TD[9] + Matrices_EEG_ISO_TD[10] + Matrices_EEG_ISO_TD[11] + Matrices_EEG_ISO_TD[12] + Matrices_EEG_ISO_TD[13] + Matrices_EEG_ISO_TD[14] + Matrices_EEG_ISO_TD[15] + Matrices_EEG_ISO_TD[16]   + Matrices_EEG_ISO_TD[17]   
EEG_JTSM_All_ASD = Matrices_EEG_JTSM_TD[0] + Matrices_EEG_JTSM_TD[1] + Matrices_EEG_JTSM_TD[2] + Matrices_EEG_JTSM_TD[3] + Matrices_EEG_JTSM_TD[4] + Matrices_EEG_JTSM_TD[5] + Matrices_EEG_JTSM_TD[6] + Matrices_EEG_JTSM_TD[7] + Matrices_EEG_JTSM_TD[8] + Matrices_EEG_JTSM_TD[9] + Matrices_EEG_JTSM_TD[10] + Matrices_EEG_JTSM_TD[11] + Matrices_EEG_JTSM_TD[12] + Matrices_EEG_JTSM_TD[13] + Matrices_EEG_JTSM_TD[14] + Matrices_EEG_JTSM_TD[15]  + Matrices_EEG_JTSM_TD[16] + Matrices_EEG_JTSM_TD[17] 
EEG_JTLA_All_ASD = Matrices_EEG_JTLA_TD[0] + Matrices_EEG_JTLA_TD[1] + Matrices_EEG_JTLA_TD[2] + Matrices_EEG_JTLA_TD[3] + Matrices_EEG_JTLA_TD[4] + Matrices_EEG_JTLA_TD[5] + Matrices_EEG_JTLA_TD[6] + Matrices_EEG_JTLA_TD[7] + Matrices_EEG_JTLA_TD[8] + Matrices_EEG_JTLA_TD[9] + Matrices_EEG_JTLA_TD[10] + Matrices_EEG_JTLA_TD[11] + Matrices_EEG_JTLA_TD[12] + Matrices_EEG_JTLA_TD[13] + Matrices_EEG_JTLA_TD[14] + Matrices_EEG_JTLA_TD[15] + Matrices_EEG_JTLA_TD[16]  + Matrices_EEG_JTLA_TD[17]     
EEG_RDM_All_ASD = Matrices_EEG_RDM_TD[0] + Matrices_EEG_RDM_TD[1] + Matrices_EEG_RDM_TD[2] + Matrices_EEG_RDM_TD[3] + Matrices_EEG_RDM_TD[4] + Matrices_EEG_RDM_TD[5] + Matrices_EEG_RDM_TD[6] + Matrices_EEG_RDM_TD[7] + Matrices_EEG_RDM_TD[8] + Matrices_EEG_RDM_TD[9] + Matrices_EEG_RDM_TD[10] + Matrices_EEG_RDM_TD[11] + Matrices_EEG_RDM_TD[12] + Matrices_EEG_RDM_TD[13] + Matrices_EEG_RDM_TD[14]  + Matrices_EEG_RDM_TD[15] + Matrices_EEG_RDM_TD[16] + Matrices_EEG_RDM_TD[17] 

EEG_ISO_Highpass_All_ASD = Matrices_EEG_Highpass_ISO_TD[0] + Matrices_EEG_Highpass_ISO_TD[1] + Matrices_EEG_Highpass_ISO_TD[2] + Matrices_EEG_Highpass_ISO_TD[3] + Matrices_EEG_Highpass_ISO_TD[4] + Matrices_EEG_Highpass_ISO_TD[5] + Matrices_EEG_Highpass_ISO_TD[6] + Matrices_EEG_Highpass_ISO_TD[7] + Matrices_EEG_Highpass_ISO_TD[8] + Matrices_EEG_Highpass_ISO_TD[9] + Matrices_EEG_Highpass_ISO_TD[10] + Matrices_EEG_Highpass_ISO_TD[11] + Matrices_EEG_Highpass_ISO_TD[12] + Matrices_EEG_Highpass_ISO_TD[13] + Matrices_EEG_Highpass_ISO_TD[14] + Matrices_EEG_Highpass_ISO_TD[15]  + Matrices_EEG_Highpass_ISO_TD[16] + Matrices_EEG_Highpass_ISO_TD[17] 
EEG_JTSM_Highpass_All_ASD = Matrices_EEG_Highpass_JTSM_TD[0] + Matrices_EEG_Highpass_JTSM_TD[1] + Matrices_EEG_Highpass_JTSM_TD[2] + Matrices_EEG_Highpass_JTSM_TD[3] + Matrices_EEG_Highpass_JTSM_TD[4] + Matrices_EEG_Highpass_JTSM_TD[5] + Matrices_EEG_Highpass_JTSM_TD[6] + Matrices_EEG_Highpass_JTSM_TD[7] + Matrices_EEG_Highpass_JTSM_TD[8] + Matrices_EEG_Highpass_JTSM_TD[9] + Matrices_EEG_Highpass_JTSM_TD[10] + Matrices_EEG_Highpass_JTSM_TD[11] + Matrices_EEG_Highpass_JTSM_TD[12] + Matrices_EEG_Highpass_JTSM_TD[13] + Matrices_EEG_Highpass_JTSM_TD[14] + Matrices_EEG_Highpass_JTSM_TD[15] + Matrices_EEG_Highpass_JTSM_TD[16]+ Matrices_EEG_Highpass_JTSM_TD[17]
EEG_JTLA_Highpass_All_ASD = Matrices_EEG_Highpass_JTLA_TD[0] + Matrices_EEG_Highpass_JTLA_TD[1] + Matrices_EEG_Highpass_JTLA_TD[2] + Matrices_EEG_Highpass_JTLA_TD[3] + Matrices_EEG_Highpass_JTLA_TD[4] + Matrices_EEG_Highpass_JTLA_TD[5] + Matrices_EEG_Highpass_JTLA_TD[6] + Matrices_EEG_Highpass_JTLA_TD[7] + Matrices_EEG_Highpass_JTLA_TD[8] + Matrices_EEG_Highpass_JTLA_TD[9] + Matrices_EEG_Highpass_JTLA_TD[10] + Matrices_EEG_Highpass_JTLA_TD[11] + Matrices_EEG_Highpass_JTLA_TD[12] + Matrices_EEG_Highpass_JTLA_TD[13] + Matrices_EEG_Highpass_JTLA_TD[14] + Matrices_EEG_Highpass_JTLA_TD[15] + Matrices_EEG_Highpass_JTLA_TD[16]+ Matrices_EEG_Highpass_JTLA_TD[17]  
EEG_RDM_Highpass_All_ASD = Matrices_EEG_Highpass_RDM_TD[0] + Matrices_EEG_Highpass_RDM_TD[1] + Matrices_EEG_Highpass_RDM_TD[2] + Matrices_EEG_Highpass_RDM_TD[3] + Matrices_EEG_Highpass_RDM_TD[4] + Matrices_EEG_Highpass_RDM_TD[5] + Matrices_EEG_Highpass_RDM_TD[6] + Matrices_EEG_Highpass_RDM_TD[7] + Matrices_EEG_Highpass_RDM_TD[8] + Matrices_EEG_Highpass_RDM_TD[9] + Matrices_EEG_Highpass_RDM_TD[10] + Matrices_EEG_Highpass_RDM_TD[11] + Matrices_EEG_Highpass_RDM_TD[12] + Matrices_EEG_Highpass_RDM_TD[13] + Matrices_EEG_Highpass_RDM_TD[14]  + Matrices_EEG_Highpass_RDM_TD[15]  + Matrices_EEG_Highpass_RDM_TD[16]  + Matrices_EEG_Highpass_RDM_TD[17] 

EEG_ISO_Lowpass_All_ASD = Matrices_EEG_Lowpass_ISO_TD[0] + Matrices_EEG_Lowpass_ISO_TD[1] + Matrices_EEG_Lowpass_ISO_TD[2] + Matrices_EEG_Lowpass_ISO_TD[3] + Matrices_EEG_Lowpass_ISO_TD[4] + Matrices_EEG_Lowpass_ISO_TD[5] + Matrices_EEG_Lowpass_ISO_TD[6] + Matrices_EEG_Lowpass_ISO_TD[7] + Matrices_EEG_Lowpass_ISO_TD[8] + Matrices_EEG_Lowpass_ISO_TD[9] + Matrices_EEG_Lowpass_ISO_TD[10] + Matrices_EEG_Lowpass_ISO_TD[11] + Matrices_EEG_Lowpass_ISO_TD[12] + Matrices_EEG_Lowpass_ISO_TD[13] + Matrices_EEG_Lowpass_ISO_TD[14] + Matrices_EEG_Lowpass_ISO_TD[15] + Matrices_EEG_Lowpass_ISO_TD[16]+ Matrices_EEG_Lowpass_ISO_TD[17]
EEG_JTSM_Lowpass_All_ASD = Matrices_EEG_Lowpass_JTSM_TD[0] + Matrices_EEG_Lowpass_JTSM_TD[1] + Matrices_EEG_Lowpass_JTSM_TD[2] + Matrices_EEG_Lowpass_JTSM_TD[3] + Matrices_EEG_Lowpass_JTSM_TD[4] + Matrices_EEG_Lowpass_JTSM_TD[5] + Matrices_EEG_Lowpass_JTSM_TD[6] + Matrices_EEG_Lowpass_JTSM_TD[7] + Matrices_EEG_Lowpass_JTSM_TD[8] + Matrices_EEG_Lowpass_JTSM_TD[9] + Matrices_EEG_Lowpass_JTSM_TD[10] + Matrices_EEG_Lowpass_JTSM_TD[11] + Matrices_EEG_Lowpass_JTSM_TD[12] + Matrices_EEG_Lowpass_JTSM_TD[13] + Matrices_EEG_Lowpass_JTSM_TD[14] + Matrices_EEG_Lowpass_JTSM_TD[15] + Matrices_EEG_Lowpass_JTSM_TD[16]+ Matrices_EEG_Lowpass_JTSM_TD[17]
EEG_JTLA_Lowpass_All_ASD = Matrices_EEG_Lowpass_JTLA_TD[0] + Matrices_EEG_Lowpass_JTLA_TD[1] + Matrices_EEG_Lowpass_JTLA_TD[2] + Matrices_EEG_Lowpass_JTLA_TD[3] + Matrices_EEG_Lowpass_JTLA_TD[4] + Matrices_EEG_Lowpass_JTLA_TD[5] + Matrices_EEG_Lowpass_JTLA_TD[6] + Matrices_EEG_Lowpass_JTLA_TD[7] + Matrices_EEG_Lowpass_JTLA_TD[8] + Matrices_EEG_Lowpass_JTLA_TD[9] + Matrices_EEG_Lowpass_JTLA_TD[10] + Matrices_EEG_Lowpass_JTLA_TD[11] + Matrices_EEG_Lowpass_JTLA_TD[12] + Matrices_EEG_Lowpass_JTLA_TD[13] + Matrices_EEG_Lowpass_JTLA_TD[14] + Matrices_EEG_Lowpass_JTLA_TD[15]  + Matrices_EEG_Lowpass_JTLA_TD[16]+ Matrices_EEG_Lowpass_JTLA_TD[17]
EEG_RDM_Lowpass_All_ASD = Matrices_EEG_Lowpass_RDM_TD[0] + Matrices_EEG_Lowpass_RDM_TD[1] + Matrices_EEG_Lowpass_RDM_TD[2] + Matrices_EEG_Lowpass_RDM_TD[3] + Matrices_EEG_Lowpass_RDM_TD[4] + Matrices_EEG_Lowpass_RDM_TD[5] + Matrices_EEG_Lowpass_RDM_TD[6] + Matrices_EEG_Lowpass_RDM_TD[7] + Matrices_EEG_Lowpass_RDM_TD[8] + Matrices_EEG_Lowpass_RDM_TD[9] + Matrices_EEG_Lowpass_RDM_TD[10] + Matrices_EEG_Lowpass_RDM_TD[11] + Matrices_EEG_Lowpass_RDM_TD[12] + Matrices_EEG_Lowpass_RDM_TD[13] + Matrices_EEG_Lowpass_RDM_TD[14] + Matrices_EEG_Lowpass_RDM_TD[15]+ Matrices_EEG_Lowpass_RDM_TD[16]+ Matrices_EEG_Lowpass_RDM_TD[17]

Stim_ISO_All_ASD = Matrices_Stim_ISO_TD[0] + Matrices_Stim_ISO_TD[1] + Matrices_Stim_ISO_TD[2] + Matrices_Stim_ISO_TD[3] + Matrices_Stim_ISO_TD[4] + Matrices_Stim_ISO_TD[5] + Matrices_Stim_ISO_TD[6] + Matrices_Stim_ISO_TD[7] + Matrices_Stim_ISO_TD[8] + Matrices_Stim_ISO_TD[9] + Matrices_Stim_ISO_TD[10] + Matrices_Stim_ISO_TD[11] + Matrices_Stim_ISO_TD[12] + Matrices_Stim_ISO_TD[13] + Matrices_Stim_ISO_TD[14] + Matrices_Stim_ISO_TD[15] + Matrices_Stim_ISO_TD[16]+ Matrices_Stim_ISO_TD[17]  
Stim_JTSM_All_ASD = Matrices_Stim_JTSM_TD[0] + Matrices_Stim_JTSM_TD[1] + Matrices_Stim_JTSM_TD[2] + Matrices_Stim_JTSM_TD[3] + Matrices_Stim_JTSM_TD[4] + Matrices_Stim_JTSM_TD[5] + Matrices_Stim_JTSM_TD[6] + Matrices_Stim_JTSM_TD[7] + Matrices_Stim_JTSM_TD[8] + Matrices_Stim_JTSM_TD[9] + Matrices_Stim_JTSM_TD[10] + Matrices_Stim_JTSM_TD[11] + Matrices_Stim_JTSM_TD[12] + Matrices_Stim_JTSM_TD[13] + Matrices_Stim_JTSM_TD[14] + Matrices_Stim_JTSM_TD[15]  + Matrices_Stim_JTSM_TD[16]+ Matrices_Stim_JTSM_TD[17]
Stim_JTLA_All_ASD = Matrices_Stim_JTLA_TD[0] + Matrices_Stim_JTLA_TD[1] + Matrices_Stim_JTLA_TD[2] + Matrices_Stim_JTLA_TD[3] + Matrices_Stim_JTLA_TD[4] + Matrices_Stim_JTLA_TD[5] + Matrices_Stim_JTLA_TD[6] + Matrices_Stim_JTLA_TD[7] + Matrices_Stim_JTLA_TD[8] + Matrices_Stim_JTLA_TD[9] + Matrices_Stim_JTLA_TD[10] + Matrices_Stim_JTLA_TD[11] + Matrices_Stim_JTLA_TD[12] + Matrices_Stim_JTLA_TD[13] + Matrices_Stim_JTLA_TD[14] + Matrices_Stim_JTLA_TD[15] + Matrices_Stim_JTLA_TD[16]+ Matrices_Stim_JTLA_TD[17]
Stim_RDM_All_ASD = Matrices_Stim_RDM_TD[0] + Matrices_Stim_RDM_TD[1] + Matrices_Stim_RDM_TD[2] + Matrices_Stim_RDM_TD[3] + Matrices_Stim_RDM_TD[4] + Matrices_Stim_RDM_TD[5] + Matrices_Stim_RDM_TD[6] + Matrices_Stim_RDM_TD[7] + Matrices_Stim_RDM_TD[8] + Matrices_Stim_RDM_TD[9] + Matrices_Stim_RDM_TD[10] + Matrices_Stim_RDM_TD[11] + Matrices_Stim_RDM_TD[12] + Matrices_Stim_RDM_TD[13] + Matrices_Stim_RDM_TD[14] + Matrices_Stim_RDM_TD[15]+ Matrices_Stim_RDM_TD[16]+ Matrices_Stim_RDM_TD[17]


#%% TD 17 subjects (=all subjects)

EEG_ISO_All_TD = Matrices_EEG_ISO_TD[0] + Matrices_EEG_ISO_TD[1] + Matrices_EEG_ISO_TD[2] + Matrices_EEG_ISO_TD[3] + Matrices_EEG_ISO_TD[4] + Matrices_EEG_ISO_TD[5] + Matrices_EEG_ISO_TD[6] + Matrices_EEG_ISO_TD[7] + Matrices_EEG_ISO_TD[8] + Matrices_EEG_ISO_TD[9] + Matrices_EEG_ISO_TD[10] + Matrices_EEG_ISO_TD[11] + Matrices_EEG_ISO_TD[12] + Matrices_EEG_ISO_TD[13] + Matrices_EEG_ISO_TD[14] + Matrices_EEG_ISO_TD[15]   
EEG_JTSM_All_TD = Matrices_EEG_JTSM_TD[0] + Matrices_EEG_JTSM_TD[1] + Matrices_EEG_JTSM_TD[2] + Matrices_EEG_JTSM_TD[3] + Matrices_EEG_JTSM_TD[4] + Matrices_EEG_JTSM_TD[5] + Matrices_EEG_JTSM_TD[6] + Matrices_EEG_JTSM_TD[7] + Matrices_EEG_JTSM_TD[8] + Matrices_EEG_JTSM_TD[9] + Matrices_EEG_JTSM_TD[10] + Matrices_EEG_JTSM_TD[11] + Matrices_EEG_JTSM_TD[12] + Matrices_EEG_JTSM_TD[13] + Matrices_EEG_JTSM_TD[14] + Matrices_EEG_JTSM_TD[15]   
EEG_JTLA_All_TD = Matrices_EEG_JTLA_TD[0] + Matrices_EEG_JTLA_TD[1] + Matrices_EEG_JTLA_TD[2] + Matrices_EEG_JTLA_TD[3] + Matrices_EEG_JTLA_TD[4] + Matrices_EEG_JTLA_TD[5] + Matrices_EEG_JTLA_TD[6] + Matrices_EEG_JTLA_TD[7] + Matrices_EEG_JTLA_TD[8] + Matrices_EEG_JTLA_TD[9] + Matrices_EEG_JTLA_TD[10] + Matrices_EEG_JTLA_TD[11] + Matrices_EEG_JTLA_TD[12] + Matrices_EEG_JTLA_TD[13] + Matrices_EEG_JTLA_TD[14] + Matrices_EEG_JTLA_TD[15]   
EEG_RDM_All_TD = Matrices_EEG_RDM_TD[0] + Matrices_EEG_RDM_TD[1] + Matrices_EEG_RDM_TD[2] + Matrices_EEG_RDM_TD[3] + Matrices_EEG_RDM_TD[4] + Matrices_EEG_RDM_TD[5] + Matrices_EEG_RDM_TD[6] + Matrices_EEG_RDM_TD[7] + Matrices_EEG_RDM_TD[8] + Matrices_EEG_RDM_TD[9] + Matrices_EEG_RDM_TD[10] + Matrices_EEG_RDM_TD[11] + Matrices_EEG_RDM_TD[12] + Matrices_EEG_RDM_TD[13] + Matrices_EEG_RDM_TD[14]  

EEG_ISO_Highpass_All_TD = Matrices_EEG_Highpass_ISO_TD[0] + Matrices_EEG_Highpass_ISO_TD[1] + Matrices_EEG_Highpass_ISO_TD[2] + Matrices_EEG_Highpass_ISO_TD[3] + Matrices_EEG_Highpass_ISO_TD[4] + Matrices_EEG_Highpass_ISO_TD[5] + Matrices_EEG_Highpass_ISO_TD[6] + Matrices_EEG_Highpass_ISO_TD[7] + Matrices_EEG_Highpass_ISO_TD[8] + Matrices_EEG_Highpass_ISO_TD[9] + Matrices_EEG_Highpass_ISO_TD[10] + Matrices_EEG_Highpass_ISO_TD[11] + Matrices_EEG_Highpass_ISO_TD[12] + Matrices_EEG_Highpass_ISO_TD[13] + Matrices_EEG_Highpass_ISO_TD[14] + Matrices_EEG_Highpass_ISO_TD[15] + Matrices_EEG_Highpass_ISO_TD[16]  
EEG_JTSM_Highpass_All_TD = Matrices_EEG_Highpass_JTSM_TD[0] + Matrices_EEG_Highpass_JTSM_TD[1] + Matrices_EEG_Highpass_JTSM_TD[2] + Matrices_EEG_Highpass_JTSM_TD[3] + Matrices_EEG_Highpass_JTSM_TD[4] + Matrices_EEG_Highpass_JTSM_TD[5] + Matrices_EEG_Highpass_JTSM_TD[6] + Matrices_EEG_Highpass_JTSM_TD[7] + Matrices_EEG_Highpass_JTSM_TD[8] + Matrices_EEG_Highpass_JTSM_TD[9] + Matrices_EEG_Highpass_JTSM_TD[10] + Matrices_EEG_Highpass_JTSM_TD[11] + Matrices_EEG_Highpass_JTSM_TD[12] + Matrices_EEG_Highpass_JTSM_TD[13] + Matrices_EEG_Highpass_JTSM_TD[14] + Matrices_EEG_Highpass_JTSM_TD[15] + Matrices_EEG_Highpass_JTSM_TD[16]   
EEG_JTLA_Highpass_All_TD = Matrices_EEG_Highpass_JTLA_TD[0] + Matrices_EEG_Highpass_JTLA_TD[1] + Matrices_EEG_Highpass_JTLA_TD[2] + Matrices_EEG_Highpass_JTLA_TD[3] + Matrices_EEG_Highpass_JTLA_TD[4] + Matrices_EEG_Highpass_JTLA_TD[5] + Matrices_EEG_Highpass_JTLA_TD[6] + Matrices_EEG_Highpass_JTLA_TD[7] + Matrices_EEG_Highpass_JTLA_TD[8] + Matrices_EEG_Highpass_JTLA_TD[9] + Matrices_EEG_Highpass_JTLA_TD[10] + Matrices_EEG_Highpass_JTLA_TD[11] + Matrices_EEG_Highpass_JTLA_TD[12] + Matrices_EEG_Highpass_JTLA_TD[13] + Matrices_EEG_Highpass_JTLA_TD[14] + Matrices_EEG_Highpass_JTLA_TD[15] + Matrices_EEG_Highpass_JTLA_TD[16]  
EEG_RDM_Highpass_All_TD = Matrices_EEG_Highpass_RDM_TD[0] + Matrices_EEG_Highpass_RDM_TD[1] + Matrices_EEG_Highpass_RDM_TD[2] + Matrices_EEG_Highpass_RDM_TD[3] + Matrices_EEG_Highpass_RDM_TD[4] + Matrices_EEG_Highpass_RDM_TD[5] + Matrices_EEG_Highpass_RDM_TD[6] + Matrices_EEG_Highpass_RDM_TD[7] + Matrices_EEG_Highpass_RDM_TD[8] + Matrices_EEG_Highpass_RDM_TD[9] + Matrices_EEG_Highpass_RDM_TD[10] + Matrices_EEG_Highpass_RDM_TD[11] + Matrices_EEG_Highpass_RDM_TD[12] + Matrices_EEG_Highpass_RDM_TD[13] + Matrices_EEG_Highpass_RDM_TD[14] + Matrices_EEG_Highpass_RDM_TD[15] + Matrices_EEG_Highpass_RDM_TD[16]   

EEG_ISO_Lowpass_All_TD = Matrices_EEG_Lowpass_ISO_TD[0] + Matrices_EEG_Lowpass_ISO_TD[1] + Matrices_EEG_Lowpass_ISO_TD[2] + Matrices_EEG_Lowpass_ISO_TD[3] + Matrices_EEG_Lowpass_ISO_TD[4] + Matrices_EEG_Lowpass_ISO_TD[5] + Matrices_EEG_Lowpass_ISO_TD[6] + Matrices_EEG_Lowpass_ISO_TD[7] + Matrices_EEG_Lowpass_ISO_TD[8] + Matrices_EEG_Lowpass_ISO_TD[9] + Matrices_EEG_Lowpass_ISO_TD[10] + Matrices_EEG_Lowpass_ISO_TD[11] + Matrices_EEG_Lowpass_ISO_TD[12] + Matrices_EEG_Lowpass_ISO_TD[13] + Matrices_EEG_Lowpass_ISO_TD[14] + Matrices_EEG_Lowpass_ISO_TD[15] + Matrices_EEG_Lowpass_ISO_TD[16]   
EEG_JTSM_Lowpass_All_TD = Matrices_EEG_Lowpass_JTSM_TD[0] + Matrices_EEG_Lowpass_JTSM_TD[1] + Matrices_EEG_Lowpass_JTSM_TD[2] + Matrices_EEG_Lowpass_JTSM_TD[3] + Matrices_EEG_Lowpass_JTSM_TD[4] + Matrices_EEG_Lowpass_JTSM_TD[5] + Matrices_EEG_Lowpass_JTSM_TD[6] + Matrices_EEG_Lowpass_JTSM_TD[7] + Matrices_EEG_Lowpass_JTSM_TD[8] + Matrices_EEG_Lowpass_JTSM_TD[9] + Matrices_EEG_Lowpass_JTSM_TD[10] + Matrices_EEG_Lowpass_JTSM_TD[11] + Matrices_EEG_Lowpass_JTSM_TD[12] + Matrices_EEG_Lowpass_JTSM_TD[13] + Matrices_EEG_Lowpass_JTSM_TD[14] + Matrices_EEG_Lowpass_JTSM_TD[15] + Matrices_EEG_Lowpass_JTSM_TD[16]   
EEG_JTLA_Lowpass_All_TD = Matrices_EEG_Lowpass_JTLA_TD[0] + Matrices_EEG_Lowpass_JTLA_TD[1] + Matrices_EEG_Lowpass_JTLA_TD[2] + Matrices_EEG_Lowpass_JTLA_TD[3] + Matrices_EEG_Lowpass_JTLA_TD[4] + Matrices_EEG_Lowpass_JTLA_TD[5] + Matrices_EEG_Lowpass_JTLA_TD[6] + Matrices_EEG_Lowpass_JTLA_TD[7] + Matrices_EEG_Lowpass_JTLA_TD[8] + Matrices_EEG_Lowpass_JTLA_TD[9] + Matrices_EEG_Lowpass_JTLA_TD[10] + Matrices_EEG_Lowpass_JTLA_TD[11] + Matrices_EEG_Lowpass_JTLA_TD[12] + Matrices_EEG_Lowpass_JTLA_TD[13] + Matrices_EEG_Lowpass_JTLA_TD[14] + Matrices_EEG_Lowpass_JTLA_TD[15] + Matrices_EEG_Lowpass_JTLA_TD[16]
EEG_RDM_Lowpass_All_TD = Matrices_EEG_Lowpass_RDM_TD[0] + Matrices_EEG_Lowpass_RDM_TD[1] + Matrices_EEG_Lowpass_RDM_TD[2] + Matrices_EEG_Lowpass_RDM_TD[3] + Matrices_EEG_Lowpass_RDM_TD[4] + Matrices_EEG_Lowpass_RDM_TD[5] + Matrices_EEG_Lowpass_RDM_TD[6] + Matrices_EEG_Lowpass_RDM_TD[7] + Matrices_EEG_Lowpass_RDM_TD[8] + Matrices_EEG_Lowpass_RDM_TD[9] + Matrices_EEG_Lowpass_RDM_TD[10] + Matrices_EEG_Lowpass_RDM_TD[11] + Matrices_EEG_Lowpass_RDM_TD[12] + Matrices_EEG_Lowpass_RDM_TD[13] + Matrices_EEG_Lowpass_RDM_TD[14] + Matrices_EEG_Lowpass_RDM_TD[15] + Matrices_EEG_Lowpass_RDM_TD[16] 

Stim_ISO_All_TD = Matrices_Stim_ISO_TD[0] + Matrices_Stim_ISO_TD[1] + Matrices_Stim_ISO_TD[2] + Matrices_Stim_ISO_TD[3] + Matrices_Stim_ISO_TD[4] + Matrices_Stim_ISO_TD[5] + Matrices_Stim_ISO_TD[6] + Matrices_Stim_ISO_TD[7] + Matrices_Stim_ISO_TD[8] + Matrices_Stim_ISO_TD[9] + Matrices_Stim_ISO_TD[10] + Matrices_Stim_ISO_TD[11] + Matrices_Stim_ISO_TD[12] + Matrices_Stim_ISO_TD[13] + Matrices_Stim_ISO_TD[14] + Matrices_Stim_ISO_TD[15] + Matrices_Stim_ISO_TD[16]
Stim_JTSM_All_TD = Matrices_Stim_JTSM_TD[0] + Matrices_Stim_JTSM_TD[1] + Matrices_Stim_JTSM_TD[2] + Matrices_Stim_JTSM_TD[3] + Matrices_Stim_JTSM_TD[4] + Matrices_Stim_JTSM_TD[5] + Matrices_Stim_JTSM_TD[6] + Matrices_Stim_JTSM_TD[7] + Matrices_Stim_JTSM_TD[8] + Matrices_Stim_JTSM_TD[9] + Matrices_Stim_JTSM_TD[10] + Matrices_Stim_JTSM_TD[11] + Matrices_Stim_JTSM_TD[12] + Matrices_Stim_JTSM_TD[13] + Matrices_Stim_JTSM_TD[14] + Matrices_Stim_JTSM_TD[15] + Matrices_Stim_JTSM_TD[16]   
Stim_JTLA_All_TD = Matrices_Stim_JTLA_TD[0] + Matrices_Stim_JTLA_TD[1] + Matrices_Stim_JTLA_TD[2] + Matrices_Stim_JTLA_TD[3] + Matrices_Stim_JTLA_TD[4] + Matrices_Stim_JTLA_TD[5] + Matrices_Stim_JTLA_TD[6] + Matrices_Stim_JTLA_TD[7] + Matrices_Stim_JTLA_TD[8] + Matrices_Stim_JTLA_TD[9] + Matrices_Stim_JTLA_TD[10] + Matrices_Stim_JTLA_TD[11] + Matrices_Stim_JTLA_TD[12] + Matrices_Stim_JTLA_TD[13] + Matrices_Stim_JTLA_TD[14] + Matrices_Stim_JTLA_TD[15] + Matrices_Stim_JTLA_TD[16]   
Stim_RDM_All_TD = Matrices_Stim_RDM_TD[0] + Matrices_Stim_RDM_TD[1] + Matrices_Stim_RDM_TD[2] + Matrices_Stim_RDM_TD[3] + Matrices_Stim_RDM_TD[4] + Matrices_Stim_RDM_TD[5] + Matrices_Stim_RDM_TD[6] + Matrices_Stim_RDM_TD[7] + Matrices_Stim_RDM_TD[8] + Matrices_Stim_RDM_TD[9] + Matrices_Stim_RDM_TD[10] + Matrices_Stim_RDM_TD[11] + Matrices_Stim_RDM_TD[12] + Matrices_Stim_RDM_TD[13] + Matrices_Stim_RDM_TD[14] + Matrices_Stim_RDM_TD[15] + Matrices_Stim_RDM_TD[16]

#%% Save big matrices

# TD
import pickle
        
with open('Matrices_EEG_ISO_All_TD.pkl', 'wb') as f:
    pickle.dump(EEG_ISO_All_TD, f)
with open('Matrices_EEG_JTSM_All_TD.pkl', 'wb') as f:
    pickle.dump(EEG_JTSM_All_TD, f)
with open('Matrices_EEG_JTLA_All_TD.pkl', 'wb') as f:
    pickle.dump(EEG_JTLA_All_TD, f)
with open('Matrices_EEG_RDM_All_TD.pkl', 'wb') as f:
    pickle.dump(EEG_RDM_All_TD, f)
    

with open('Matrices_EEG_ISO_Highpass_All_TD.pkl', 'wb') as f:
    pickle.dump(EEG_ISO_Highpass_All_TD, f)
with open('Matrices_EEG_JTSM_Highpass_All_TD.pkl', 'wb') as f:
    pickle.dump(EEG_JTSM_Highpass_All_TD, f)
with open('Matrices_EEG_JTLA_Highpass_All_TD.pkl', 'wb') as f:
    pickle.dump(EEG_JTLA_Highpass_All_TD, f)
with open('Matrices_EEG_RDM_Highpass_All_TD.pkl', 'wb') as f:
    pickle.dump(EEG_RDM_Highpass_All_TD, f)
    
    
with open('Matrices_EEG_ISO_Lowpass_All_TD.pkl', 'wb') as f:
    pickle.dump(EEG_ISO_Lowpass_All_TD, f)
with open('Matrices_EEG_JTSM_Lowpass_All_TD.pkl', 'wb') as f:
    pickle.dump(EEG_JTSM_Lowpass_All_TD, f)
with open('Matrices_EEG_JTLA_Lowpass_All_TD.pkl', 'wb') as f:
    pickle.dump(EEG_JTLA_Lowpass_All_TD, f)
with open('Matrices_EEG_RDM_Lowpass_All_TD.pkl', 'wb') as f:
    pickle.dump(EEG_RDM_Lowpass_All_TD, f)
    
    
with open('Matrices_Stim_ISO_All_TD.pkl', 'wb') as f:
    pickle.dump(Stim_ISO_All_TD, f)
with open('Matrices_Stim_JTSM_All_TD.pkl', 'wb') as f:
    pickle.dump(Stim_JTSM_All_TD, f)
with open('Matrices_Stim_JTLA_All_TD.pkl', 'wb') as f:
    pickle.dump(Stim_JTLA_All_TD, f)
with open('Matrices_Stim_RDM_All_TD.pkl', 'wb') as f:
    pickle.dump(Stim_RDM_All_TD, f)
    
# ASD

import pickle
        
with open('Matrices_EEG_ISO_All_ASD.pkl', 'wb') as f:
    pickle.dump(EEG_ISO_All_ASD, f)
with open('Matrices_EEG_JTSM_All_ASD.pkl', 'wb') as f:
    pickle.dump(EEG_JTSM_All_ASD, f)
with open('Matrices_EEG_JTLA_All_ASD.pkl', 'wb') as f:
    pickle.dump(EEG_JTLA_All_ASD, f)
with open('Matrices_EEG_RDM_All_ASD.pkl', 'wb') as f:
    pickle.dump(EEG_RDM_All_ASD, f)
    

with open('Matrices_EEG_ISO_Highpass_All_ASD.pkl', 'wb') as f:
    pickle.dump(EEG_ISO_Highpass_All_ASD, f)
with open('Matrices_EEG_JTSM_Highpass_All_ASD.pkl', 'wb') as f:
    pickle.dump(EEG_JTSM_Highpass_All_ASD, f)
with open('Matrices_EEG_JTLA_Highpass_All_ASD.pkl', 'wb') as f:
    pickle.dump(EEG_JTLA_Highpass_All_ASD, f)
with open('Matrices_EEG_RDM_Highpass_All_ASD.pkl', 'wb') as f:
    pickle.dump(EEG_RDM_Highpass_All_ASD, f)
    
    
with open('Matrices_EEG_ISO_Lowpass_All_ASD.pkl', 'wb') as f:
    pickle.dump(EEG_ISO_Lowpass_All_ASD, f)
with open('Matrices_EEG_JTSM_Lowpass_All_ASD.pkl', 'wb') as f:
    pickle.dump(EEG_JTSM_Lowpass_All_ASD, f)
with open('Matrices_EEG_JTLA_Lowpass_All_ASD.pkl', 'wb') as f:
    pickle.dump(EEG_JTLA_Lowpass_All_ASD, f)
with open('Matrices_EEG_RDM_Lowpass_All_ASD.pkl', 'wb') as f:
    pickle.dump(EEG_RDM_Lowpass_All_ASD, f)
    
    
with open('Matrices_Stim_ISO_All_ASD.pkl', 'wb') as f:
    pickle.dump(Stim_ISO_All_ASD, f)
with open('Matrices_Stim_JTSM_All_ASD.pkl', 'wb') as f:
    pickle.dump(Stim_JTSM_All_ASD, f)
with open('Matrices_Stim_JTLA_All_ASD.pkl', 'wb') as f:
    pickle.dump(Stim_JTLA_All_ASD, f)
with open('Matrices_Stim_RDM_All_ASD.pkl', 'wb') as f:
    pickle.dump(Stim_RDM_All_ASD, f)

#%%

# Remove bads electrodes for ASD = AF7, PO7, AF8, F7, FC5

for i in range (0,len(Response),1):
    Response[i] = np.delete(Response[i],34,1) # AF8
    Response[i] = np.delete(Response[i],24,1) # PO7
    Response[i] = np.delete(Response[i],8,1) # FC5
    Response[i] = np.delete(Response[i],6,1) # F7
    Response[i] = np.delete(Response[i],1,1) # AF7
    
    
for i in range (0,len(Response),1):
    Response[i][:,1] = np.zeros((4480)) # AF8
    Response[i][:,6] = np.zeros((4480)) # PO7
    Response[i][:,8] = np.zeros((4480)) # FC5
    Response[i][:,24] = np.zeros((4480)) # F7
    Response[i][:,34] = np.zeros((4480)) # AF7
    
    Response[i][:,34] = np.zeros((4480)) # AF7
    Response[i][:,34] = np.zeros((4480)) # AF7
    Response[i][:,34] = np.zeros((4480)) # AF7
    Response[i][:,34] = np.zeros((4480)) # AF7
    
    
    
    



#%% Load matrices for TRF

import pickle
import os
import numpy as np
from matplotlib import pyplot as plt

with open('Matrices_EEG_RDM_Lowpass_All_TD.pkl', 'rb') as f:
     Response = pickle.load(f)

with open('Matrices_Stim_RDM_All_TD.pkl', 'rb') as f:
     Stim = pickle.load(f)
     
# # For ASD - EEG Lowpass - ISO

# del Response[46]

# del Stim[44]

# for i in range (0,len(Response),1):
#     Response[i][:,1] = np.zeros((4480)) # AF8


# For backward model with all electrodes

from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split

fs = 128 # Sampling rate

# regularization = [1.0e-4, 0.01, 100, 1e4]

# regularization = [1, 100]

regularization = 0.01

bwd_trf = TRF(metric=pearsonr, direction=-1)  # use pearsons correlation, use direction=-1 for backward model
tmin, tmax = 0.0, 2.5 # range of time lag

x_train, x_test, y_train, y_test = train_test_split(Stim, Response) # Split the data 75 - 25 (train - test)

bwd_trf.train(x_train, y_train, fs, tmin, tmax, regularization, k=5) # Regularization selection with crossvalidation inside trainset

prediction,r_corr_single = bwd_trf.predict(x_test,y_test) # Predicted vector of stimulus

#%% Save train and test set for comparaison with matlab mTRF toolbox

y_train_2 = y_train[0:40]
y_test_2 = y_test[0:8]
x_train_2 = x_train[0:40]
x_test_2 = x_test[0:8]

#regularization = [1, 100]

regularization = 0.01

bwd_trf = TRF(metric=pearsonr, direction=-1)  # use pearsons correlation, use direction=-1 for backward model
tmin, tmax = -1.0, 2.0 # range of time lag

bwd_trf.train(x_train_2, y_train_2, fs, tmin, tmax, regularization, k=5) # Regularization selection with crossvalidation inside trainset

prediction,r_corr_single = bwd_trf.predict(x_test_2,y_test_2) # Predicted vector of stimulus

file_path_mat = 'C://Users/tvanneau/Dropbox (EinsteinMed)/PC/Documents/MATLAB/mTRF Shlomit data/'

import scipy.io
scipy.io.savemat(file_path_mat+'EEG_train.mat', mdict={'EEG_train': y_train_2})
scipy.io.savemat(file_path_mat+'EEG_test.mat', mdict={'EEG_test': y_test_2})
scipy.io.savemat(file_path_mat+'Stim_train.mat', mdict={'Stim_train': x_train_2})
scipy.io.savemat(file_path_mat+'Stim_test.mat', mdict={'Stim_test': x_test_2})


#%%

import pickle
import os
import numpy as np
from matplotlib import pyplot as plt

with open('Matrices_EEG_ISO_Lowpass_All_TD.pkl', 'rb') as f:
     Response = pickle.load(f)

with open('Matrices_Stim_ISO_All_TD.pkl', 'rb') as f:
     Stim = pickle.load(f)

from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split

r_corr_array = np.zeros((1,10)) # Prediction accuracy for each rotation
reg_par_array = np.zeros((1,10)) # Selected regularization value for each rotation
weights_list = [] # Model weights
Stim_pred = [] # Will contain list of predicted value for each rotation 
Stim_real = [] # Will contain corresponding real stimulation vector

# Cut the matrice into a train and a test set 
for iii in range (0,10,1):
    x_train, x_test, y_train, y_test = train_test_split(Stim, Response) # Split the data 75 - 25 (train - test)
    
    bwd_trf.train(x_train, y_train, fs, tmin, tmax, regularization, k=10) # Regularization selection with crossvalidation inside trainset
    
    prediction,r_corr_single = bwd_trf.predict(x_test,y_test) # Predicted vector of stimulus
    
    r_corr_array[0,iii] = r_corr_single
    reg_par_array[0,iii] = bwd_trf.regularization
    weights_list.append(bwd_trf.weights)
    Stim_pred.append(prediction)
    Stim_real.append(x_test)
    
    
#%%    

from matplotlib import pyplot as plt
from scipy.signal import find_peaks

ISI_pred = []
for p in range (0,len(prediction),1):
    Peaks = find_peaks(prediction[p].flatten(), height= ( np.mean(prediction[p]) + 1.5*np.std(prediction[p])) )
    Peaks_idx = Peaks[0]
    
    Peaks_time = Peaks_idx/fs
    
    ISI = np.zeros((len(Peaks_time)-1,1))
    for i in range (0,len(Peaks_time)-1,1):
        ISI[i] = Peaks_time[i+1] - Peaks_time[i]
    
    ISI_pred.append(ISI)
    
ISI = np.vstack(ISI_pred)

ISI = ISI[ISI < (np.std(ISI))] 
#ISI = ISI[ISI < 2*(np.std(ISI))] 

bins=np.arange(0.5, 0.8, 0.02)
plt.hist(ISI, bins=bins, edgecolor='w')
plt.xticks(bins);

plt.title(label= 'Median =  ' + "%0.2f" % np.median(ISI) + '  Std =' "%0.2f" % np.std(ISI))

np.save('ISI_Backward_RDM_TD_Reg001_Tlags_2-5sec', ISI)

#%%

# from mne.channels import make_standard_montage

# # use standard montage for the EEG system used for recording the response
# montage = make_standard_montage('biosemi64')

# evokeds = bwd_trf.to_mne_evoked(montage)

# evokeds[0].data = np.flip(evokeds[0].data,1)

# evokeds[0].plot_joint([-1.5, -1.0, -0.345], topomap_args={"scalings": 1}, ts_args={"units": "a.u.", "scalings": dict(eeg=1)})

#%%

from mne.channels import make_standard_montage

# use standard montage for the EEG system used for recording the response
montage = make_standard_montage('biosemi64')

evokeds = bwd_trf.to_mne_evoked(montage)

evokeds[0].plot_joint([-0.483, -0.283, -0.11], topomap_args={"scalings": 1}, ts_args={"units": "a.u.", "scalings": dict(eeg=1)})


#%%

evokeds[0].save('Backward_TRF_JTSM_EEG_Lowpass_Fliped_ASD-ave.fif')

#%%

test = np.mean(y_test, axis=0)
ave_test = []
d=0
w = 321
for i in range(0,12,1):
    ave_test.append(test[d:w,:])
    d=w
    w = w + 321
    
test2 = np.mean(ave_test, axis=0)
plt.plot(test2)

