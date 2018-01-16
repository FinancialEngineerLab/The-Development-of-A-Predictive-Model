
# coding: utf-8

# # Data Preparation

# In[1]:

import io, os, sys, types
import numpy as np
import pandas as pd
sys.path.append('../Python Scripts/APIs/')
import api_data_retrieval as api


# ## Collecting Data and Creating AdeSavGol variables

# In[2]:

for country in ["UK","US","GRM","CND","AUS","JPN"]:
    api.retreiveData(country,['I2','I1','GM2','GM1','FF1','MP1','MP4','MP2','AA1'])


# In[3]:

windowlength, polyorder = 5, 4

api.AdewoyinSavgolFilter(['I2','I1','GM2','GM1','FF1','MP1','MP4','MP2'],'dir_pred_US',save_xlsx=True,
                         _auto_filter_date = True, windowlength = windowlength, polyorder = polyorder)
api.AdewoyinSavgolFilter(['I2','I1','GM2','GM1','FF1','MP1','MP4','MP2'],'dir_pred_UK',save_xlsx=True,
                         _auto_filter_date = True, windowlength = windowlength, polyorder = polyorder)
api.AdewoyinSavgolFilter(['I2','I1','GM2','GM1','FF1','MP1','MP4','MP2'],'dir_pred_GRM',save_xlsx=True,
                         _auto_filter_date = True, windowlength = windowlength, polyorder = polyorder)
api.AdewoyinSavgolFilter(['I2','I1','GM2','GM1','FF1','MP1','MP4','MP2'],'dir_pred_JPN',save_xlsx=True,
                         _auto_filter_date = True, windowlength = windowlength, polyorder = polyorder)
api.AdewoyinSavgolFilter(['I2','I1','GM2','GM1','FF1','MP1','MP4','MP2'],'dir_pred_CND',save_xlsx=True,
                         _auto_filter_date = True, windowlength = windowlength, polyorder = polyorder)
api.AdewoyinSavgolFilter(['I2','I1','GM2','GM1','FF1','MP1','MP4','MP2'],'dir_pred_AUS',save_xlsx=True,
                         _auto_filter_date = True, windowlength = windowlength, polyorder = polyorder)
#Gather data into one "COMBINED" sheet

api.createCombinedSheet('dir_pred_US',['I2','I1','GM2','GM1','FF1','MP1','MP4','MP2','BC3','AA1'],two_types=True)
api.createCombinedSheet('dir_pred_UK',['I2','I1','GM2','GM1','FF1','MP1','MP4','MP2','BC3','AA1'],two_types=True)
api.createCombinedSheet('dir_pred_GRM',['I2','I1','GM2','GM1','FF1','MP1','MP4','MP2','BC3','AA1'],two_types=True)
api.createCombinedSheet('dir_pred_JPN',['I2','I1','GM2','GM1','FF1','MP1','MP4','MP2','BC3','AA1'],two_types=True)
api.createCombinedSheet('dir_pred_CND',['I2','I1','GM2','GM1','FF1','MP1','MP4','MP2','BC3','AA1'],two_types=True)
api.createCombinedSheet('dir_pred_AUS',['I2','I1','GM2','GM1','FF1','MP1','MP4','MP2','BC3','AA1'],two_types=True)

