# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:39:20 2024

@author: tvanneau
"""

#%% Import and concatenate individuals matrices


#%% Train and test forward model for ISO condition for TD group 
import pickle
import mne
import numpy as np
import os
from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split
from mne.channels import make_standard_montage

fs = 128 # Sampling rate

regularization = [1.0e-4, 0.01, 1, 100, 1e4]

tmin, tmax = 0.0, 2.5 # range of time lag

lag = np.arange(64,320)

Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]


r_corr_all = []
weights_all = []

for sb in range(0,len(Subjects_name),1):
    
    file_path_2 = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/'
    
    with open(file_path_2 + Subjects_name[sb], 'rb') as f:
        [EEG_ISO, EEG_JTSM, EEG_JTLA, EEG_RDM, 
                     EEG_ISO_lowpass, EEG_JTSM_lowpass, EEG_JTLA_lowpass, EEG_RDM_lowpass,
                     EEG_ISO_highpass, EEG_JTSM_highpass, EEG_JTLA_highpass, EEG_RDM_highpass,
                     Stim_ISO, Stim_JTSM, Stim_JTLA, Stim_RDM] = pickle.load(f)
    
    # Occipital subset
    Nb_elect = 9
    Response_pre = EEG_ISO_lowpass
    Response_cluster = []
    for i in range (0,len(Response_pre),1):
        x = np.zeros((len(Response_pre[0]),Nb_elect))
        x[:,0] = Response_pre[i][:,25] # PO3
        x[:,1] = Response_pre[i][:,26] # O1
        x[:,2] = Response_pre[i][:,27] # Iz
        x[:,3] = Response_pre[i][:,28] # Oz
        x[:,4] = Response_pre[i][:,29] # POz
        x[:,5] = Response_pre[i][:,62] # PO4
        x[:,6] = Response_pre[i][:,63] # O2   
        x[:,7] = Response_pre[i][:,24] # PO7
        x[:,8] = Response_pre[i][:,61] # PO8
        
        Response_cluster.append(x)
    
    # ISO
    Stim = Stim_ISO
    Response = Response_cluster
    fwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model
    
    r_corr_array = np.zeros((1,10))
    weights_list = []
    
    for iii in range(0,10,1):
        x_train, x_test, y_train, y_test = train_test_split(Stim, Response) # Split the data 75 - 25 (train - test)
        fwd_trf.train(x_train, y_train, fs, tmin, tmax, regularization, k=5) # Regularization selection with crossvalidation inside trainset
        
        prediction,r_corr_single = fwd_trf.predict(x_test,y_test, lag) # Predicted vector of stimulus
    
        r_corr_array[0,iii] = r_corr_single # Contain the r corr value for the 10 outer loop
        weights_list.append(fwd_trf.weights) # Contain the weights for the 10 models builds during the outer loop
        
    r_corr_all.append(r_corr_array)
    weights_all.append(weights_list)
    

with open('Predicted_ISO_Stim_Forward_Lowpass_TD_Indiv_Subject.pkl', 'wb') as f:
   pickle.dump([weights_all,r_corr_all], f)
        
#%% Train and test forward model for JTSM condition for TD group 
import pickle
import mne
import numpy as np
import os
from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split
from mne.channels import make_standard_montage

fs = 128 # Sampling rate

regularization = [1.0e-4, 0.01, 1, 100, 1e4]

tmin, tmax = 0.0, 2.5 # range of time lag

lag = np.arange(64,320)

Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

r_corr_all = []
weights_all = []

for sb in range(0,len(Subjects_name),1):
    
    file_path_2 = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/'
    
    with open(file_path_2 + Subjects_name[sb], 'rb') as f:
        [EEG_ISO, EEG_JTSM, EEG_JTLA, EEG_RDM, 
                     EEG_ISO_lowpass, EEG_JTSM_lowpass, EEG_JTLA_lowpass, EEG_RDM_lowpass,
                     EEG_ISO_highpass, EEG_JTSM_highpass, EEG_JTLA_highpass, EEG_RDM_highpass,
                     Stim_ISO, Stim_JTSM, Stim_JTLA, Stim_RDM] = pickle.load(f)
    
    # Occipital subset
    Nb_elect = 9
    Response_pre = EEG_JTSM_lowpass
    Response_cluster = []
    for i in range (0,len(Response_pre),1):
        x = np.zeros((len(Response_pre[0]),Nb_elect))
        x[:,0] = Response_pre[i][:,25] # PO3
        x[:,1] = Response_pre[i][:,26] # O1
        x[:,2] = Response_pre[i][:,27] # Iz
        x[:,3] = Response_pre[i][:,28] # Oz
        x[:,4] = Response_pre[i][:,29] # POz
        x[:,5] = Response_pre[i][:,62] # PO4
        x[:,6] = Response_pre[i][:,63] # O2   
        x[:,7] = Response_pre[i][:,24] # PO7
        x[:,8] = Response_pre[i][:,61] # PO8
        
        Response_cluster.append(x)
    
    # JTSM
    Stim = Stim_JTSM
    Response = Response_cluster
    fwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model

    r_corr_array = np.zeros((1,10))
    weights_list = []
    
    for iii in range(0,10,1):
        x_train, x_test, y_train, y_test = train_test_split(Stim, Response) # Split the data 75 - 25 (train - test)
        fwd_trf.train(x_train, y_train, fs, tmin, tmax, regularization, k=5) # Regularization selection with crossvalidation inside trainset
        
        prediction,r_corr_single = fwd_trf.predict(x_test,y_test, lag) # Predicted vector of stimulus
    
        r_corr_array[0,iii] = r_corr_single # Contain the r corr value for the 10 outer loop
        weights_list.append(fwd_trf.weights) # Contain the weights for the 10 models builds during the outer loop
        
    r_corr_all.append(r_corr_array)
    weights_all.append(weights_list)
    

with open('Predicted_JTSM_Stim_Forward_Lowpass_TD_Indiv_Subject.pkl', 'wb') as f:
   pickle.dump([weights_all,r_corr_all], f)
        
        
#%% Train and test forward model for JTLA condition for TD group 
import pickle
import mne
import numpy as np
import os
from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split
from mne.channels import make_standard_montage

fs = 128 # Sampling rate

regularization = [1.0e-4, 0.01, 1, 100, 1e4]

tmin, tmax = 0.0, 2.5 # range of time lag

lag = np.arange(64,320)

Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

r_corr_all = []
weights_all = []

for sb in range(0,len(Subjects_name),1):
    
    file_path_2 = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/'
    
    with open(file_path_2 + Subjects_name[sb], 'rb') as f:
        [EEG_ISO, EEG_JTSM, EEG_JTLA, EEG_RDM, 
                     EEG_ISO_lowpass, EEG_JTSM_lowpass, EEG_JTLA_lowpass, EEG_RDM_lowpass,
                     EEG_ISO_highpass, EEG_JTSM_highpass, EEG_JTLA_highpass, EEG_RDM_highpass,
                     Stim_ISO, Stim_JTSM, Stim_JTLA, Stim_RDM] = pickle.load(f)
    
    # Occipital subset
    Nb_elect = 9
    Response_pre = EEG_JTLA_lowpass
    Response_cluster = []
    for i in range (0,len(Response_pre),1):
        x = np.zeros((len(Response_pre[0]),Nb_elect))
        x[:,0] = Response_pre[i][:,25] # PO3
        x[:,1] = Response_pre[i][:,26] # O1
        x[:,2] = Response_pre[i][:,27] # Iz
        x[:,3] = Response_pre[i][:,28] # Oz
        x[:,4] = Response_pre[i][:,29] # POz
        x[:,5] = Response_pre[i][:,62] # PO4
        x[:,6] = Response_pre[i][:,63] # O2   
        x[:,7] = Response_pre[i][:,24] # PO7
        x[:,8] = Response_pre[i][:,61] # PO8
        
        Response_cluster.append(x)
    
    # JTLA
    Stim = Stim_JTLA
    Response = Response_cluster
    fwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model

    r_corr_array = np.zeros((1,10))
    weights_list = []
    
    for iii in range(0,10,1):
        x_train, x_test, y_train, y_test = train_test_split(Stim, Response) # Split the data 75 - 25 (train - test)
        fwd_trf.train(x_train, y_train, fs, tmin, tmax, regularization, k=5) # Regularization selection with crossvalidation inside trainset
        
        prediction,r_corr_single = fwd_trf.predict(x_test,y_test, lag) # Predicted vector of stimulus
    
        r_corr_array[0,iii] = r_corr_single # Contain the r corr value for the 10 outer loop
        weights_list.append(fwd_trf.weights) # Contain the weights for the 10 models builds during the outer loop
        
    r_corr_all.append(r_corr_array)
    weights_all.append(weights_list)
    

with open('Predicted_JTLA_Stim_Forward_Lowpass_TD_Indiv_Subject.pkl', 'wb') as f:
   pickle.dump([weights_all,r_corr_all], f)
        
#%% Train and test forward model for RDM condition for TD group 
import pickle
import mne
import numpy as np
import os
from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split
from mne.channels import make_standard_montage

fs = 128 # Sampling rate

regularization = [1.0e-4, 0.01, 1, 100, 1e4]

tmin, tmax = 0.0, 2.5 # range of time lag

lag = np.arange(64,320)

Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

r_corr_all = []
weights_all = []

for sb in range(0,len(Subjects_name),1):
    
    file_path_2 = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/'
    
    with open(file_path_2 + Subjects_name[sb], 'rb') as f:
        [EEG_ISO, EEG_JTSM, EEG_JTLA, EEG_RDM, 
                     EEG_ISO_lowpass, EEG_JTSM_lowpass, EEG_JTLA_lowpass, EEG_RDM_lowpass,
                     EEG_ISO_highpass, EEG_JTSM_highpass, EEG_JTLA_highpass, EEG_RDM_highpass,
                     Stim_ISO, Stim_JTSM, Stim_JTLA, Stim_RDM] = pickle.load(f)
    
    # Occipital subset
    Nb_elect = 9
    Response_pre = EEG_RDM_lowpass
    Response_cluster = []
    for i in range (0,len(Response_pre),1):
        x = np.zeros((len(Response_pre[0]),Nb_elect))
        x[:,0] = Response_pre[i][:,25] # PO3
        x[:,1] = Response_pre[i][:,26] # O1
        x[:,2] = Response_pre[i][:,27] # Iz
        x[:,3] = Response_pre[i][:,28] # Oz
        x[:,4] = Response_pre[i][:,29] # POz
        x[:,5] = Response_pre[i][:,62] # PO4
        x[:,6] = Response_pre[i][:,63] # O2   
        x[:,7] = Response_pre[i][:,24] # PO7
        x[:,8] = Response_pre[i][:,61] # PO8
        
        Response_cluster.append(x)
    
    # RDM
    Stim = Stim_RDM
    Response = Response_cluster
    fwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model

    r_corr_array = np.zeros((1,10))
    weights_list = []
    
    for iii in range(0,10,1):
        x_train, x_test, y_train, y_test = train_test_split(Stim, Response) # Split the data 75 - 25 (train - test)
        fwd_trf.train(x_train, y_train, fs, tmin, tmax, regularization, k=5) # Regularization selection with crossvalidation inside trainset
        
        prediction,r_corr_single = fwd_trf.predict(x_test,y_test, lag) # Predicted vector of stimulus
    
        r_corr_array[0,iii] = r_corr_single # Contain the r corr value for the 10 outer loop
        weights_list.append(fwd_trf.weights) # Contain the weights for the 10 models builds during the outer loop
        
    r_corr_all.append(r_corr_array)
    weights_all.append(weights_list)
    

with open('Predicted_RDM_Stim_Forward_Lowpass_TD_Indiv_Subject.pkl', 'wb') as f:
   pickle.dump([weights_all,r_corr_all], f)
        
        
#%% Train and test forward model for ISO condition for ASD group 
import pickle
import mne
import numpy as np
import os
from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split
from mne.channels import make_standard_montage

fs = 128 # Sampling rate

regularization = [1.0e-4, 0.01, 1, 100, 1e4]

tmin, tmax = 0.0, 2.5 # range of time lag

lag = np.arange(64,320)

Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/ASD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

r_corr_all = []
weights_all = []

for sb in range(0,len(Subjects_name),1):
    
    file_path_2 = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/ASD/Autoreject_preprocessed files/Matrices/'
    
    with open(file_path_2 + Subjects_name[sb], 'rb') as f:
        [EEG_ISO, EEG_JTSM, EEG_JTLA, EEG_RDM, 
                     EEG_ISO_lowpass, EEG_JTSM_lowpass, EEG_JTLA_lowpass, EEG_RDM_lowpass,
                     EEG_ISO_highpass, EEG_JTSM_highpass, EEG_JTLA_highpass, EEG_RDM_highpass,
                     Stim_ISO, Stim_JTSM, Stim_JTLA, Stim_RDM] = pickle.load(f)
    
    # Occipital subset
    Nb_elect = 9
    Response_pre = EEG_ISO_lowpass
    Response_cluster = []
    for i in range (0,len(Response_pre),1):
        x = np.zeros((len(Response_pre[0]),Nb_elect))
        x[:,0] = Response_pre[i][:,25] # PO3
        x[:,1] = Response_pre[i][:,26] # O1
        x[:,2] = Response_pre[i][:,27] # Iz
        x[:,3] = Response_pre[i][:,28] # Oz
        x[:,4] = Response_pre[i][:,29] # POz
        x[:,5] = Response_pre[i][:,62] # PO4
        x[:,6] = Response_pre[i][:,63] # O2   
        x[:,7] = Response_pre[i][:,24] # PO7
        x[:,8] = Response_pre[i][:,61] # PO8
        
        Response_cluster.append(x)
    
    # ISO
    Stim = Stim_ISO
    Response = Response_cluster
    fwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model

    r_corr_array = np.zeros((1,10))
    weights_list = []
    
    for iii in range(0,10,1):
        x_train, x_test, y_train, y_test = train_test_split(Stim, Response) # Split the data 75 - 25 (train - test)
        fwd_trf.train(x_train, y_train, fs, tmin, tmax, regularization, k=5) # Regularization selection with crossvalidation inside trainset
        
        prediction,r_corr_single = fwd_trf.predict(x_test,y_test, lag) # Predicted vector of stimulus
    
        r_corr_array[0,iii] = r_corr_single # Contain the r corr value for the 10 outer loop
        weights_list.append(fwd_trf.weights) # Contain the weights for the 10 models builds during the outer loop
        
    r_corr_all.append(r_corr_array)
    weights_all.append(weights_list)
    

with open('Predicted_ISO_Stim_Forward_Lowpass_ASD_Indiv_Subject.pkl', 'wb') as f:
   pickle.dump([weights_all,r_corr_all], f)
        
        
#%% Train and test forward model for JTSM condition for ASD group 
import pickle
import mne
import numpy as np
import os
from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split
from mne.channels import make_standard_montage

fs = 128 # Sampling rate

regularization = [1.0e-4, 0.01, 1, 100, 1e4]

tmin, tmax = 0.0, 2.5 # range of time lag

lag = np.arange(64,320)

Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/ASD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

r_corr_all = []
weights_all = []

for sb in range(0,len(Subjects_name),1):
    
    file_path_2 = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/ASD/Autoreject_preprocessed files/Matrices/'
    
    with open(file_path_2 + Subjects_name[sb], 'rb') as f:
        [EEG_ISO, EEG_JTSM, EEG_JTLA, EEG_RDM, 
                     EEG_ISO_lowpass, EEG_JTSM_lowpass, EEG_JTLA_lowpass, EEG_RDM_lowpass,
                     EEG_ISO_highpass, EEG_JTSM_highpass, EEG_JTLA_highpass, EEG_RDM_highpass,
                     Stim_ISO, Stim_JTSM, Stim_JTLA, Stim_RDM] = pickle.load(f)
    
    # Occipital subset
    Nb_elect = 9
    Response_pre = EEG_JTSM_lowpass
    Response_cluster = []
    for i in range (0,len(Response_pre),1):
        x = np.zeros((len(Response_pre[0]),Nb_elect))
        x[:,0] = Response_pre[i][:,25] # PO3
        x[:,1] = Response_pre[i][:,26] # O1
        x[:,2] = Response_pre[i][:,27] # Iz
        x[:,3] = Response_pre[i][:,28] # Oz
        x[:,4] = Response_pre[i][:,29] # POz
        x[:,5] = Response_pre[i][:,62] # PO4
        x[:,6] = Response_pre[i][:,63] # O2   
        x[:,7] = Response_pre[i][:,24] # PO7
        x[:,8] = Response_pre[i][:,61] # PO8
        
        Response_cluster.append(x)
    
    # ISO
    Stim = Stim_JTSM
    Response = Response_cluster
    fwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model

    r_corr_array = np.zeros((1,10))
    weights_list = []
    
    for iii in range(0,10,1):
        x_train, x_test, y_train, y_test = train_test_split(Stim, Response) # Split the data 75 - 25 (train - test)
        fwd_trf.train(x_train, y_train, fs, tmin, tmax, regularization, k=5) # Regularization selection with crossvalidation inside trainset
        
        prediction,r_corr_single = fwd_trf.predict(x_test,y_test, lag) # Predicted vector of stimulus
    
        r_corr_array[0,iii] = r_corr_single # Contain the r corr value for the 10 outer loop
        weights_list.append(fwd_trf.weights) # Contain the weights for the 10 models builds during the outer loop
        
    r_corr_all.append(r_corr_array)
    weights_all.append(weights_list)
    

with open('Predicted_JTSM_Stim_Forward_Lowpass_ASD_Indiv_Subject.pkl', 'wb') as f:
   pickle.dump([weights_all,r_corr_all], f)
        
        
#%% Train and test forward model for JTLA condition for ASD group 
import pickle
import mne
import numpy as np
import os
from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split
from mne.channels import make_standard_montage

fs = 128 # Sampling rate

regularization = [1.0e-4, 0.01, 1, 100, 1e4]

tmin, tmax = 0.0, 2.5 # range of time lag

lag = np.arange(64,320)

Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/ASD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

r_corr_all = []
weights_all = []

for sb in range(0,len(Subjects_name),1):
    
    file_path_2 = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/ASD/Autoreject_preprocessed files/Matrices/'
    
    with open(file_path_2 + Subjects_name[sb], 'rb') as f:
        [EEG_ISO, EEG_JTSM, EEG_JTLA, EEG_RDM, 
                     EEG_ISO_lowpass, EEG_JTSM_lowpass, EEG_JTLA_lowpass, EEG_RDM_lowpass,
                     EEG_ISO_highpass, EEG_JTSM_highpass, EEG_JTLA_highpass, EEG_RDM_highpass,
                     Stim_ISO, Stim_JTSM, Stim_JTLA, Stim_RDM] = pickle.load(f)
    
    # Occipital subset
    Nb_elect = 9
    Response_pre = EEG_JTLA_lowpass
    Response_cluster = []
    for i in range (0,len(Response_pre),1):
        x = np.zeros((len(Response_pre[0]),Nb_elect))
        x[:,0] = Response_pre[i][:,25] # PO3
        x[:,1] = Response_pre[i][:,26] # O1
        x[:,2] = Response_pre[i][:,27] # Iz
        x[:,3] = Response_pre[i][:,28] # Oz
        x[:,4] = Response_pre[i][:,29] # POz
        x[:,5] = Response_pre[i][:,62] # PO4
        x[:,6] = Response_pre[i][:,63] # O2   
        x[:,7] = Response_pre[i][:,24] # PO7
        x[:,8] = Response_pre[i][:,61] # PO8
        
        Response_cluster.append(x)
    
    # ISO
    Stim = Stim_JTLA
    Response = Response_cluster
    fwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model

    r_corr_array = np.zeros((1,10))
    weights_list = []
    
    for iii in range(0,10,1):
        x_train, x_test, y_train, y_test = train_test_split(Stim, Response) # Split the data 75 - 25 (train - test)
        fwd_trf.train(x_train, y_train, fs, tmin, tmax, regularization, k=5) # Regularization selection with crossvalidation inside trainset
        
        prediction,r_corr_single = fwd_trf.predict(x_test,y_test, lag) # Predicted vector of stimulus
    
        r_corr_array[0,iii] = r_corr_single # Contain the r corr value for the 10 outer loop
        weights_list.append(fwd_trf.weights) # Contain the weights for the 10 models builds during the outer loop
        
    r_corr_all.append(r_corr_array)
    weights_all.append(weights_list)
    

with open('Predicted_JTLA_Stim_Forward_Lowpass_ASD_Indiv_Subject.pkl', 'wb') as f:
   pickle.dump([weights_all,r_corr_all], f)
        
        
#%% Train and test forward model for RDM condition for ASD group 
import pickle
import mne
import numpy as np
import os
from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split
from mne.channels import make_standard_montage

fs = 128 # Sampling rate

regularization = [1.0e-4, 0.01, 1, 100, 1e4]

tmin, tmax = 0.0, 2.5 # range of time lag

lag = np.arange(64,320)

Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/ASD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

r_corr_all = []
weights_all = []

for sb in range(0,len(Subjects_name),1):
    
    file_path_2 = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/ASD/Autoreject_preprocessed files/Matrices/'
    
    with open(file_path_2 + Subjects_name[sb], 'rb') as f:
        [EEG_ISO, EEG_JTSM, EEG_JTLA, EEG_RDM, 
                     EEG_ISO_lowpass, EEG_JTSM_lowpass, EEG_JTLA_lowpass, EEG_RDM_lowpass,
                     EEG_ISO_highpass, EEG_JTSM_highpass, EEG_JTLA_highpass, EEG_RDM_highpass,
                     Stim_ISO, Stim_JTSM, Stim_JTLA, Stim_RDM] = pickle.load(f)
    
    # Occipital subset
    Nb_elect = 9
    Response_pre = EEG_RDM_lowpass
    Response_cluster = []
    for i in range (0,len(Response_pre),1):
        x = np.zeros((len(Response_pre[0]),Nb_elect))
        x[:,0] = Response_pre[i][:,25] # PO3
        x[:,1] = Response_pre[i][:,26] # O1
        x[:,2] = Response_pre[i][:,27] # Iz
        x[:,3] = Response_pre[i][:,28] # Oz
        x[:,4] = Response_pre[i][:,29] # POz
        x[:,5] = Response_pre[i][:,62] # PO4
        x[:,6] = Response_pre[i][:,63] # O2   
        x[:,7] = Response_pre[i][:,24] # PO7
        x[:,8] = Response_pre[i][:,61] # PO8
        
        Response_cluster.append(x)
    
    # ISO
    Stim = Stim_RDM
    Response = Response_cluster
    fwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model

    r_corr_array = np.zeros((1,10))
    weights_list = []
    
    for iii in range(0,10,1):
        x_train, x_test, y_train, y_test = train_test_split(Stim, Response) # Split the data 75 - 25 (train - test)
        fwd_trf.train(x_train, y_train, fs, tmin, tmax, regularization, k=5) # Regularization selection with crossvalidation inside trainset
        
        prediction,r_corr_single = fwd_trf.predict(x_test,y_test, lag) # Predicted vector of stimulus
    
        r_corr_array[0,iii] = r_corr_single # Contain the r corr value for the 10 outer loop
        weights_list.append(fwd_trf.weights) # Contain the weights for the 10 models builds during the outer loop
        
    r_corr_all.append(r_corr_array)
    weights_all.append(weights_list)
    

with open('Predicted_RDM_Stim_Forward_Lowpass_ASD_Indiv_Subject.pkl', 'wb') as f:
   pickle.dump([weights_all,r_corr_all], f)
#%%

# Individuals forward models - Using ISO models to predict jitters conditions 

# For TD
import os
import numpy as np
from matplotlib import pyplot as plt
import pickle
from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split

Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

r_corr_JTSM = []
r_corr_JTLA = []
r_corr_RDM = []

# EEG_JTSM_TD = []
# EEG_JTLA_TD = []

for sb in range(0,len(Subjects_name),1):
    
    file_path_2 = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/'
    
    with open(file_path_2 + Subjects_name[sb], 'rb') as f:
        [EEG_ISO, EEG_JTSM, EEG_JTLA, EEG_RDM, 
                     EEG_ISO_lowpass, EEG_JTSM_lowpass, EEG_JTLA_lowpass, EEG_RDM_lowpass,
                     EEG_ISO_highpass, EEG_JTSM_highpass, EEG_JTLA_highpass, EEG_RDM_highpass,
                     Stim_ISO, Stim_JTSM, Stim_JTLA, Stim_RDM] = pickle.load(f)
    
    # EEG_JTSM_TD.append(EEG_JTSM)
    # EEG_JTLA_TD.append(EEG_JTLA)

    file_path_m = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/Forward models indiv with lag/TD/'
    # Load ISO model
    bwd_trf = TRF(metric=pearsonr, direction=-1)  # use pearsons correlation, use direction=-1 for backward model
    bwd_trf.load(file_path_m + 'Forward_TRF_ISO_EEG_Lowpass_TD_Subject_' + Subjects_name[sb])
    
    # JTSM
    Stim = Stim_JTSM
    
    # Occipital subset
    Nb_elect = 9
    Response_pre = EEG_JTSM_lowpass
    Response_cluster = []
    for i in range (0,len(Response_pre),1):
        x = np.zeros((len(Response_pre[0]),Nb_elect))
        x[:,0] = Response_pre[i][:,25] # PO3
        x[:,1] = Response_pre[i][:,26] # O1
        x[:,2] = Response_pre[i][:,27] # Iz
        x[:,3] = Response_pre[i][:,28] # Oz
        x[:,4] = Response_pre[i][:,29] # POz
        x[:,5] = Response_pre[i][:,62] # PO4
        x[:,6] = Response_pre[i][:,63] # O2   
        x[:,7] = Response_pre[i][:,24] # PO7
        x[:,8] = Response_pre[i][:,61] # PO8
        
        Response_cluster.append(x)
    
    Response = Response_cluster
    lag = np.arange(64,320)
        
    prediction,r_corr_single = bwd_trf.predict(Stim,Response,lag) # Predicted vector of stimulus
    
    r_corr_JTSM.append(r_corr_single)
    
    
    # JTLA
    Stim = Stim_JTLA
    
    # Occipital subset
    Nb_elect = 9
    Response_pre = EEG_JTLA_lowpass
    Response_cluster = []
    for i in range (0,len(Response_pre),1):
        x = np.zeros((len(Response_pre[0]),Nb_elect))
        x[:,0] = Response_pre[i][:,25] # PO3
        x[:,1] = Response_pre[i][:,26] # O1
        x[:,2] = Response_pre[i][:,27] # Iz
        x[:,3] = Response_pre[i][:,28] # Oz
        x[:,4] = Response_pre[i][:,29] # POz
        x[:,5] = Response_pre[i][:,62] # PO4
        x[:,6] = Response_pre[i][:,63] # O2   
        x[:,7] = Response_pre[i][:,24] # PO7
        x[:,8] = Response_pre[i][:,61] # PO8
        
        Response_cluster.append(x)
    
    Response = Response_cluster
    lag = np.arange(64,320)
        
    prediction,r_corr_single = bwd_trf.predict(Stim,Response,lag) # Predicted vector of stimulus
    
    r_corr_JTLA.append(r_corr_single)
    
    
    # RDM
    Stim = Stim_RDM
    
    # Occipital subset
    Nb_elect = 9
    Response_pre = EEG_RDM_lowpass
    Response_cluster = []
    for i in range (0,len(Response_pre),1):
        x = np.zeros((len(Response_pre[0]),Nb_elect))
        x[:,0] = Response_pre[i][:,25] # PO3
        x[:,1] = Response_pre[i][:,26] # O1
        x[:,2] = Response_pre[i][:,27] # Iz
        x[:,3] = Response_pre[i][:,28] # Oz
        x[:,4] = Response_pre[i][:,29] # POz
        x[:,5] = Response_pre[i][:,62] # PO4
        x[:,6] = Response_pre[i][:,63] # O2   
        x[:,7] = Response_pre[i][:,24] # PO7
        x[:,8] = Response_pre[i][:,61] # PO8
        
        Response_cluster.append(x)
    
    Response = Response_cluster
    lag = np.arange(64,320)
        
    prediction,r_corr_single = bwd_trf.predict(Stim,Response,lag) # Predicted vector of stimulus
    
    r_corr_RDM.append(r_corr_single)

#%% Saved predicted files

with open('Predicted_Indiv_Forward_Models_From_ISO_TD.pkl', 'wb') as f:
    pickle.dump([r_corr_JTSM,r_corr_JTLA,r_corr_RDM], f)
    
#%% Individuals forward models - Using ISO models to predict jitters conditions 

# For ASD
import os
import numpy as np
from matplotlib import pyplot as plt
import pickle
from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split

Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/ASD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

r_corr_JTSM = []
r_corr_JTLA = []
r_corr_RDM = []

# EEG_JTSM_TD = []
# EEG_JTLA_TD = []

for sb in range(0,len(Subjects_name),1):
    
    file_path_2 = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/ASD/Autoreject_preprocessed files/Matrices/'
    
    with open(file_path_2 + Subjects_name[sb], 'rb') as f:
        [EEG_ISO, EEG_JTSM, EEG_JTLA, EEG_RDM, 
                     EEG_ISO_lowpass, EEG_JTSM_lowpass, EEG_JTLA_lowpass, EEG_RDM_lowpass,
                     EEG_ISO_highpass, EEG_JTSM_highpass, EEG_JTLA_highpass, EEG_RDM_highpass,
                     Stim_ISO, Stim_JTSM, Stim_JTLA, Stim_RDM] = pickle.load(f)
    
    # EEG_JTSM_TD.append(EEG_JTSM)
    # EEG_JTLA_TD.append(EEG_JTLA)

    file_path_m = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/Forward models indiv with lag/ASD/'
    # Load ISO model
    bwd_trf = TRF(metric=pearsonr, direction=-1)  # use pearsons correlation, use direction=-1 for backward model
    bwd_trf.load(file_path_m + 'Forward_TRF_ISO_EEG_Lowpass_ASD_Subject_' + Subjects_name[sb])
    
    # JTSM
    Stim = Stim_JTSM
    # Occipital subset
    Nb_elect = 9
    Response_pre = EEG_JTSM_lowpass
    Response_cluster = []
    for i in range (0,len(Response_pre),1):
        x = np.zeros((len(Response_pre[0]),Nb_elect))
        x[:,0] = Response_pre[i][:,25] # PO3
        x[:,1] = Response_pre[i][:,26] # O1
        x[:,2] = Response_pre[i][:,27] # Iz
        x[:,3] = Response_pre[i][:,28] # Oz
        x[:,4] = Response_pre[i][:,29] # POz
        x[:,5] = Response_pre[i][:,62] # PO4
        x[:,6] = Response_pre[i][:,63] # O2   
        x[:,7] = Response_pre[i][:,24] # PO7
        x[:,8] = Response_pre[i][:,61] # PO8
        
        Response_cluster.append(x)
    Response = Response_cluster
    lag = np.arange(64,320)
        
    prediction,r_corr_single = bwd_trf.predict(Stim,Response,lag) # Predicted vector of stimulus
    
    r_corr_JTSM.append(r_corr_single)
    
    
    # JTLA
    Stim = Stim_JTLA
    # Occipital subset
    Nb_elect = 9
    Response_pre = EEG_JTLA_lowpass
    Response_cluster = []
    for i in range (0,len(Response_pre),1):
        x = np.zeros((len(Response_pre[0]),Nb_elect))
        x[:,0] = Response_pre[i][:,25] # PO3
        x[:,1] = Response_pre[i][:,26] # O1
        x[:,2] = Response_pre[i][:,27] # Iz
        x[:,3] = Response_pre[i][:,28] # Oz
        x[:,4] = Response_pre[i][:,29] # POz
        x[:,5] = Response_pre[i][:,62] # PO4
        x[:,6] = Response_pre[i][:,63] # O2   
        x[:,7] = Response_pre[i][:,24] # PO7
        x[:,8] = Response_pre[i][:,61] # PO8
        
        Response_cluster.append(x)
    
    Response = Response_cluster
    lag = np.arange(64,320)
        
    prediction,r_corr_single = bwd_trf.predict(Stim,Response,lag) # Predicted vector of stimulus
    
    r_corr_JTLA.append(r_corr_single)
    
    
    # RDM
    Stim = Stim_RDM
    # Occipital subset
    Nb_elect = 9
    Response_pre = EEG_RDM_lowpass
    Response_cluster = []
    for i in range (0,len(Response_pre),1):
        x = np.zeros((len(Response_pre[0]),Nb_elect))
        x[:,0] = Response_pre[i][:,25] # PO3
        x[:,1] = Response_pre[i][:,26] # O1
        x[:,2] = Response_pre[i][:,27] # Iz
        x[:,3] = Response_pre[i][:,28] # Oz
        x[:,4] = Response_pre[i][:,29] # POz
        x[:,5] = Response_pre[i][:,62] # PO4
        x[:,6] = Response_pre[i][:,63] # O2   
        x[:,7] = Response_pre[i][:,24] # PO7
        x[:,8] = Response_pre[i][:,61] # PO8
        
        Response_cluster.append(x)
    
    Response = Response_cluster
    lag = np.arange(64,320)
        
    prediction,r_corr_single = bwd_trf.predict(Stim,Response,lag) # Predicted vector of stimulus
    
    r_corr_RDM.append(r_corr_single)

#%% Saved predicted files

with open('Predicted_Indiv_Forward_Models_From_ISO_ASD.pkl', 'wb') as f:
    pickle.dump([r_corr_JTSM,r_corr_JTLA,r_corr_RDM], f)
    
    
#%% Analysis

import scipy
import mne
import os
import numpy as np
from matplotlib import pyplot as plt
import pickle

# TD 
with open('Predicted_Indiv_Forward_Models_From_ISO_TD.pkl', 'rb') as f:
 [r_corr_JTSM_TD,r_corr_JTLA_TD,r_corr_RDM_TD] = pickle.load(f)
 
 # TD 
 with open('Predicted_Indiv_Forward_Models_From_ISO_ASD.pkl', 'rb') as f:
  [r_corr_JTSM_ASD,r_corr_JTLA_ASD,r_corr_RDM_ASD] = pickle.load(f)
  
  
# Boxplot
#For jamovi
r_corr_jamovi = np.concatenate((ISO_TD.T, ISO_ASD.T, JTSM_TD.T, JTSM_ASD.T, JTLA_TD.T, JTLA_ASD.T, RDM_TD.T, RDM_ASD.T))

ticks = ['Small Jitter', 'Large Jitter', 'Random' ]

Data_B2 = [ r_corr_JTSM_TD , r_corr_JTLA_TD , r_corr_RDM_TD ]
Data_B3 = [ r_corr_JTSM_ASD , r_corr_JTLA_ASD , r_corr_RDM_ASD]

Data_B2_plot = plt.boxplot(Data_B2,
                               positions=np.array(
    np.arange(len(Data_B2)))*2.0-0.35,
                               widths=0.6)

Data_B3_plot = plt.boxplot(Data_B3,
                               positions=np.array(
    np.arange(len(Data_B3)))*2.0+0.35,
                               widths=0.6)



def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
         
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend()
 
 
# setting colors for each groups
define_box_properties(Data_B2_plot, '#2C7BB6', 'TD')
define_box_properties(Data_B3_plot, '#D7191C', 'ASD')


 
# set the x label values
plt.xticks(np.arange(0, len(ticks)*2, 2), ticks)

#%% Load ISO prediction

import pickle
import mne
import numpy as np
import os
from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split
from mne.channels import make_standard_montage

Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

r_corr_ISO_TD = []
r_corr_JTSM_TD = []
r_corr_JTLA_TD = []
r_corr_RDM_TD = []

for sb in range(0,len(Subjects_name),1):
    
    with open('Predicted_ISO_Stim_Forward_Lowpass_TD_Subject_' + Subjects_name[sb] +'.pkl', 'rb') as f:
        [prediction,r_corr_single,regularization] = pickle.load(f)
        
    r_corr_ISO_TD.append(r_corr_single)
    
    with open('Predicted_JTSM_Stim_Forward_Lowpass_TD_Subject_' + Subjects_name[sb] +'.pkl', 'rb') as f:
        [prediction,r_corr_single,regularization] = pickle.load(f)
        
    r_corr_JTSM_TD.append(r_corr_single)
    
    with open('Predicted_JTLA_Stim_Forward_Lowpass_TD_Subject_' + Subjects_name[sb] +'.pkl', 'rb') as f:
        [prediction,r_corr_single,regularization] = pickle.load(f)
        
    r_corr_JTLA_TD.append(r_corr_single)
    
    with open('Predicted_RDM_Stim_Forward_Lowpass_TD_Subject_' + Subjects_name[sb] +'.pkl', 'rb') as f:
        [prediction,r_corr_single,regularization] = pickle.load(f)
        
    r_corr_RDM_TD.append(r_corr_single)

#%%

r_corr_ISO_ASD = []

r_corr_ISO_ASD = []

Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/ASD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

r_corr_ISO_TD = []
r_corr_JTSM_ASD = []
r_corr_JTLA_ASD = []
r_corr_RDM_ASD = []

for sb in range(0,len(Subjects_name),1):
    
    with open('Predicted_ISO_Stim_Forward_Lowpass_ASD_Subject_' + Subjects_name[sb] +'.pkl', 'rb') as f:
        [prediction,r_corr_single,regularization] = pickle.load(f)
        
    r_corr_ISO_ASD.append(r_corr_single)
    
    with open('Predicted_JTSM_Stim_Forward_Lowpass_ASD_Subject_' + Subjects_name[sb] +'.pkl', 'rb') as f:
        [prediction,r_corr_single,regularization] = pickle.load(f)
        
    r_corr_JTSM_ASD.append(r_corr_single)
    
    with open('Predicted_JTLA_Stim_Forward_Lowpass_ASD_Subject_' + Subjects_name[sb] +'.pkl', 'rb') as f:
        [prediction,r_corr_single,regularization] = pickle.load(f)
        
    r_corr_JTLA_ASD.append(r_corr_single)
    
    with open('Predicted_RDM_Stim_Forward_Lowpass_ASD_Subject_' + Subjects_name[sb] +'.pkl', 'rb') as f:
        [prediction,r_corr_single,regularization] = pickle.load(f)
        
    r_corr_RDM_ASD.append(r_corr_single)
    
#%%
import scipy
import mne
import os
import numpy as np
from matplotlib import pyplot as plt
import pickle

# Boxplot
#For jamovi
r_corr_jamovi = np.concatenate((ISO_TD.T, ISO_ASD.T, JTSM_TD.T, JTSM_ASD.T, JTLA_TD.T, JTLA_ASD.T, RDM_TD.T, RDM_ASD.T))

ticks = ['ISO','Small Jitter', 'Large Jitter', 'Random' ]

Data_B2 = [ r_corr_ISO_TD  , r_corr_JTSM_TD , r_corr_JTLA_TD , r_corr_RDM_TD]
Data_B3 = [ r_corr_ISO_ASD ,  r_corr_JTSM_ASD , r_corr_JTLA_ASD , r_corr_RDM_ASD ]

Data_B2_plot = plt.boxplot(Data_B2,
                               positions=np.array(
    np.arange(len(Data_B2)))*2.0-0.35,
                               widths=0.6)

Data_B3_plot = plt.boxplot(Data_B3,
                               positions=np.array(
    np.arange(len(Data_B3)))*2.0+0.35,
                               widths=0.6)



def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
         
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend()
 
 
# setting colors for each groups
define_box_properties(Data_B2_plot, '#2C7BB6', 'TD')
define_box_properties(Data_B3_plot, '#D7191C', 'ASD')


 
# set the x label values
plt.xticks(np.arange(0, len(ticks)*2, 2), ticks)


#%% Plot indiv forward weights

import pickle
import mne
import numpy as np
import os
from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split
from mne.channels import make_standard_montage
import scipy
import mne
import os
import numpy as np
from matplotlib import pyplot as plt
import pickle

# ISO
Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

fig, ax = plt.subplots()

for sb in range(0,len(Subjects_name),1):
    
    file_path_m = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/Forward models indiv with lag/TD/'
    # Load ISO model
    bwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model
    bwd_trf.load(file_path_m + 'Forward_TRF_ISO_EEG_Lowpass_TD_Subject_' + Subjects_name[sb])
    
    
    a = np.zeros((1,321,1))
    a[:,:,0] = np.mean(bwd_trf.weights, axis=2)
    bwd_trf.weights = a

    bwd_trf.plot(axes=ax)
    ax.set_ylim(-0.0007,0.0007)
    
#%% JTSM
Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

fig, ax = plt.subplots()

for sb in range(0,len(Subjects_name),1):
    
    file_path_m = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/Forward models indiv with lag/TD/'
    # Load ISO model
    bwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model
    bwd_trf.load(file_path_m + 'Forward_TRF_JTSM_EEG_Lowpass_TD_Subject_' + Subjects_name[sb])
    
    
    a = np.zeros((1,321,1))
    a[:,:,0] = np.mean(bwd_trf.weights, axis=2)
    bwd_trf.weights = a

    bwd_trf.plot(axes=ax)
    ax.set_ylim(-0.0007,0.0007)
    
    
#%% JTLA
Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

fig, ax = plt.subplots()

for sb in range(0,len(Subjects_name),1):
    
    file_path_m = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/Forward models indiv with lag/TD/'
    # Load ISO model
    bwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model
    bwd_trf.load(file_path_m + 'Forward_TRF_JTLA_EEG_Lowpass_TD_Subject_' + Subjects_name[sb])
    
    
    a = np.zeros((1,321,1))
    a[:,:,0] = np.mean(bwd_trf.weights, axis=2)
    bwd_trf.weights = a

    bwd_trf.plot(axes=ax)
    ax.set_ylim(-0.0007,0.0007)
    
#%% RDM
Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

fig, ax = plt.subplots()

for sb in range(0,len(Subjects_name),1):
    
    file_path_m = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/Forward models indiv with lag/TD/'
    # Load ISO model
    bwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model
    bwd_trf.load(file_path_m + 'Forward_TRF_RDM_EEG_Lowpass_TD_Subject_' + Subjects_name[sb])
    
    
    a = np.zeros((1,321,1))
    a[:,:,0] = np.mean(bwd_trf.weights, axis=2)
    bwd_trf.weights = a

    bwd_trf.plot(axes=ax)
    ax.set_ylim(-0.0007,0.0007)
    
    
#%% Plot indiv forward weights

import pickle
import mne
import numpy as np
import os
from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split
from mne.channels import make_standard_montage
import scipy
import mne
import os
import numpy as np
from matplotlib import pyplot as plt
import pickle

# ISO
Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/ASD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

fig, ax = plt.subplots()
test = []
for sb in range(0,len(Subjects_name),1):
    
    file_path_m = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/Forward models indiv with lag/ASD/'
    # Load ISO model
    bwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model
    bwd_trf.load(file_path_m + 'Forward_TRF_ISO_EEG_Lowpass_ASD_Subject_' + Subjects_name[sb])
    
    
    a = np.zeros((1,321,1))
    a[:,:,0] = np.nanmean(bwd_trf.weights, axis=2)
    bwd_trf.weights = a

    bwd_trf.plot(axes=ax)
    ax.set_ylim(-0.0007,0.0007)
    test.append(a)
    
#%% JTSM
Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

fig, ax = plt.subplots()

for sb in range(0,len(Subjects_name),1):
    
    file_path_m = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/Forward models indiv with lag/TD/'
    # Load ISO model
    bwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model
    bwd_trf.load(file_path_m + 'Forward_TRF_JTSM_EEG_Lowpass_TD_Subject_' + Subjects_name[sb])
    
    
    a = np.zeros((1,321,1))
    a[:,:,0] = np.mean(bwd_trf.weights, axis=2)
    bwd_trf.weights = a

    bwd_trf.plot(axes=ax)
    ax.set_ylim(-0.0007,0.0007)
    
    
#%% JTLA
Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

fig, ax = plt.subplots()

for sb in range(0,len(Subjects_name),1):
    
    file_path_m = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/Forward models indiv with lag/TD/'
    # Load ISO model
    bwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model
    bwd_trf.load(file_path_m + 'Forward_TRF_JTLA_EEG_Lowpass_TD_Subject_' + Subjects_name[sb])
    
    
    a = np.zeros((1,321,1))
    a[:,:,0] = np.mean(bwd_trf.weights, axis=2)
    bwd_trf.weights = a

    bwd_trf.plot(axes=ax)
    ax.set_ylim(-0.0007,0.0007)
    
#%% RDM
Subjects_name = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/TD/Autoreject_preprocessed files/Matrices/')
Subjects_name = Subjects_name[16:]

fig, ax = plt.subplots()

for sb in range(0,len(Subjects_name),1):
    
    file_path_m = 'C://Users/tvanneau/Dropbox (EinsteinMed)/Model with Theo/Forward models indiv with lag/TD/'
    # Load ISO model
    bwd_trf = TRF(metric=pearsonr)  # use pearsons correlation, use direction=-1 for backward model
    bwd_trf.load(file_path_m + 'Forward_TRF_RDM_EEG_Lowpass_TD_Subject_' + Subjects_name[sb])
    
    
    a = np.zeros((1,321,1))
    a[:,:,0] = np.mean(bwd_trf.weights, axis=2)
    bwd_trf.weights = a

    bwd_trf.plot(axes=ax)
    ax.set_ylim(-0.0007,0.0007)
    
#%% Plot indiv forward weights

import pickle
import mne
import numpy as np
import os
from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse
from mtrf.stats import crossval
from mtrf.stats import nested_crossval
from sklearn.model_selection import train_test_split
from mne.channels import make_standard_montage
import scipy
import mne
import os
import numpy as np
from matplotlib import pyplot as plt
import pickle
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

mpl.rcParams['ps.fonttype'] = 42

# TD
# ISO
with open('Predicted_ISO_Stim_Forward_Lowpass_TD_Indiv_Subject.pkl', 'rb') as f:
    [weights_all_ISO_TD,r_corr_all_ISO_TD] = pickle.load(f)

# JTSM
with open('Predicted_JTSM_Stim_Forward_Lowpass_TD_Indiv_Subject.pkl', 'rb') as f:
    [weights_all_JTSM_TD,r_corr_all_JTSM_TD] = pickle.load(f)
    
# JTLA
with open('Predicted_JTLA_Stim_Forward_Lowpass_TD_Indiv_Subject.pkl', 'rb') as f:
    [weights_all_JTLA_TD,r_corr_all_JTLA_TD] = pickle.load(f)
    
# RDM
with open('Predicted_RDM_Stim_Forward_Lowpass_TD_Indiv_Subject.pkl', 'rb') as f:
    [weights_all_RDM_TD,r_corr_all_RDM_TD] = pickle.load(f)
    
    
# ASD
# ISO
with open('Predicted_ISO_Stim_Forward_Lowpass_ASD_Indiv_Subject.pkl', 'rb') as f:
    [weights_all_ISO_ASD,r_corr_all_ISO_ASD] = pickle.load(f)

# JTSM
with open('Predicted_JTSM_Stim_Forward_Lowpass_ASD_Indiv_Subject.pkl', 'rb') as f:
    [weights_all_JTSM_ASD,r_corr_all_JTSM_ASD] = pickle.load(f)
    
# JTLA
with open('Predicted_JTLA_Stim_Forward_Lowpass_ASD_Indiv_Subject.pkl', 'rb') as f:
    [weights_all_JTLA_ASD,r_corr_all_JTLA_ASD] = pickle.load(f)
    
# RDM
with open('Predicted_RDM_Stim_Forward_Lowpass_ASD_Indiv_Subject.pkl', 'rb') as f:
    [weights_all_RDM_ASD,r_corr_all_RDM_ASD] = pickle.load(f)


#%% Plot model weights

fig, ax = plt.subplots()

for i in range (0,len(weights_all_ISO_TD),1):
    
    a = np.mean(weights_all_ISO_TD[i], axis=0)
    
    ax.plot(np.mean(a, axis=2).T)
    
    
ax.set_ylim([-0.00065, 0.00045])
    
fig.savefig('TFR_ISO_TD.svg', bbox_inches='tight', format='svg')

fig, ax = plt.subplots()



for i in range (0,len(weights_all_JTSM_TD),1):
    
    a = np.mean(weights_all_JTSM_TD[i], axis=0)
    
    ax.plot(np.mean(a, axis=2).T)
    
    
ax.set_ylim([-0.00065, 0.00045])
    
fig.savefig('TFR_JTSM_TD.svg', bbox_inches='tight', format='svg')




fig, ax = plt.subplots()

for i in range (0,len(weights_all_JTLA_TD),1):
    
    a = np.mean(weights_all_JTLA_TD[i], axis=0)
    
    ax.plot(np.mean(a, axis=2).T)
    
    
ax.set_ylim([-0.00065, 0.00045])
    
fig.savefig('TFR_JTLA_TD.svg', bbox_inches='tight', format='svg')




fig, ax = plt.subplots()

for i in range (0,len(weights_all_RDM_TD),1):
    
    a = np.mean(weights_all_RDM_TD[i], axis=0)
    
    ax.plot(np.mean(a, axis=2).T)
    
    
ax.set_ylim([-0.00065, 0.00045])
    
fig.savefig('TFR_RDM_TD.svg', bbox_inches='tight', format='svg')

#%% Plot model weights

fig, ax = plt.subplots()

for i in range (0,len(weights_all_ISO_ASD),1):
    
    a = np.mean(weights_all_ISO_ASD[i], axis=0)
    
    ax.plot(np.mean(a, axis=2).T)
    
    
ax.set_ylim([-0.00065, 0.00045])
    
fig.savefig('TFR_ISO_ASD.svg', bbox_inches='tight', format='svg')

fig, ax = plt.subplots()



for i in range (0,len(weights_all_JTSM_ASD),1):
    
    a = np.mean(weights_all_JTSM_ASD[i], axis=0)
    
    ax.plot(np.mean(a, axis=2).T)
    
    
ax.set_ylim([-0.00065, 0.00045])
    
fig.savefig('TFR_JTSM_ASD.svg', bbox_inches='tight', format='svg')




fig, ax = plt.subplots()

for i in range (0,len(weights_all_JTLA_ASD),1):
    
    a = np.mean(weights_all_JTLA_ASD[i], axis=0)
    
    ax.plot(np.mean(a, axis=2).T)
    
    
ax.set_ylim([-0.00065, 0.00045])
    
fig.savefig('TFR_JTLA_ASD.svg', bbox_inches='tight', format='svg')




fig, ax = plt.subplots()

for i in range (0,len(weights_all_RDM_ASD),1):
    
    a = np.mean(weights_all_RDM_ASD[i], axis=0)
    
    ax.plot(np.mean(a, axis=2).T)
    
    
ax.set_ylim([-0.00065, 0.00045])
    
fig.savefig('TFR_RDM_ASD.svg', bbox_inches='tight', format='svg')

#%%

# Boxplot
#For jamovi
r_corr_jamovi = np.concatenate((ISO_TD.T, ISO_ASD.T, JTSM_TD.T, JTSM_ASD.T, JTLA_TD.T, JTLA_ASD.T, RDM_TD.T, RDM_ASD.T))

ticks = ['ISO','Small Jitter', 'Large Jitter', 'Random' ]

Data_B2 = [ np.mean(np.mean(r_corr_all_ISO_TD, axis=1), axis=1)  , np.mean(np.mean(r_corr_all_JTSM_TD, axis=1), axis=1) , 
           np.mean(np.mean(r_corr_all_JTLA_TD, axis=1), axis=1) , np.mean(np.mean(r_corr_all_RDM_TD, axis=1), axis=1)]

Data_B3 = [ np.mean(np.mean(r_corr_all_ISO_ASD, axis=1), axis=1) ,  np.mean(np.mean(r_corr_all_JTSM_ASD, axis=1), axis=1) , 
           np.mean(np.mean(r_corr_all_JTLA_ASD, axis=1), axis=1) , np.mean(np.mean(r_corr_all_RDM_ASD, axis=1), axis=1) ]

Data_B2_plot = plt.boxplot(Data_B2,
                               positions=np.array(
    np.arange(len(Data_B2)))*2.0-0.35,
                               widths=0.6)

Data_B3_plot = plt.boxplot(Data_B3,
                               positions=np.array(
    np.arange(len(Data_B3)))*2.0+0.35,
                               widths=0.6)



def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
         
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend()
 
 
# setting colors for each groups
define_box_properties(Data_B2_plot, '#2C7BB6', 'TD')
define_box_properties(Data_B3_plot, '#D7191C', 'ASD')


 
# set the x label values
plt.xticks(np.arange(0, len(ticks)*2, 2), ticks)

#%%

scipy.io.savemat('TD.mat', mdict={'TD_r_corr': Data_B2})
scipy.io.savemat('ASD.mat', mdict={'ASD_r_corr': Data_B3})