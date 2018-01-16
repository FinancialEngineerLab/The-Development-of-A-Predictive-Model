
# coding: utf-8

# # Building Algorithms for bond price movement ONLY MP4

# In[1]:

import io, os, sys, types
import numpy as np
import pandas as pd
import sklearn as sk
sys.path.append('../Python Scripts/APIs/')
import api_algorithm as algo
MP4 = "Only"


# ## Gathering and Organising Data

# In[23]:

data_us = algo.importData("US",MP4,max_corr=1)
data_uk = algo.importData("UK",MP4,max_corr=1)
data_jpn = algo.importData("JPN",MP4,max_corr=1)
data_aus = algo.importData("AUS",MP4,max_corr=1)
data_cnd = algo.importData("CND",MP4,max_corr=1)
data_grm = algo.importData("GRM",MP4,max_corr=1)


# In[25]:

##Enforcing neccesary lag on certain data points
data_us = algo.featureLag(data_us)
data_uk = algo.featureLag(data_uk)
data_jpn = algo.featureLag(data_jpn)
data_aus = algo.featureLag(data_aus)
data_cnd = algo.featureLag(data_cnd)
data_grm = algo.featureLag(data_grm)


# In[26]:

data_us = algo.businessCycleSplitter([(data_us,"US")])[0]
list_all_country_data = algo.businessCycleSplitter([(data_uk,"UK"), (data_jpn,"JPN"), (data_aus,"AUS"), (data_cnd,"CND"), (data_grm,"GRM")])


# In[27]:

test_country_data = algo.classAndFeature([data_us])
all_country_data =  algo.classAndFeature([data_uk, data_jpn, data_aus, data_cnd, data_grm])


# ## Training Algos

# In[6]:

all_country_data_with_algos  = algo.testingAlgoTypes(all_country_data,verbose=1,MP4=MP4)


# In[7]:

all_country_data_with_trained_algos = algo.fineTuneModel(all_country_data_with_algos)


# Voting Classifier using aglortihms within each country within business cycle

# In[9]:

##forms a voting ensemble out of the top3 algorithms for each ensemble
a_c_d_w_t_a_and_acc_scores = algo.votingEnsembleTest(all_country_data_with_trained_algos,test_country_data.get('US'))


# Voting classifiers using voting classifiers across countries for a given business business cycle

# In[10]:

algo.votingEnsembleTest2ndLayer(a_c_d_w_t_a_and_acc_scores,test_country_data.get('US'),2)


# ### For the Purposes of without API_research 
