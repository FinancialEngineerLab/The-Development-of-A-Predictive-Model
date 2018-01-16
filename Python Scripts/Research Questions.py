
# coding: utf-8

# In[1]:

get_ipython().system('jupyter nbconvert --to script *.ipynb')
import io, os, sys, types
import numpy as np
try:
    del research
except NameError:
    print("instance of research api does not exist yet")
sys.path.append('../Python Scripts/APIs/')
import api_research as research


# # Reseach Questions

# ## 1) Can I build a good trading model

# In[ ]:

#MP4 = True performs the analysis with the MP4L and MP4C features included. MP4=False does the opposite. MP4="Only" includes only MP4L and MP4C as features
research.finalModelScore(True)


# ## 2)To what extent do my statistics contribute to the 5-Year MoM Bond

# In[ ]:

research.featureImportance(MP4=True) 


# I will compare the importance of different features within each country-business cycle.
# I will evaluate feature importance using the scores from Gradient Boosted Classifiers.

# ## 3) Evaluating my assumption that behaviour differs by business cycle

# In[ ]:

##uses MP4 = True
research.crossBusinessCycleTest()


# ## 4) Which country shows the most similarities to the USA and why analysis on sub business cycle accuracy as well

# In[ ]:

##uses MP4 =True
research.countryPredComparison()


# ### Correlation between C and L variables for a given country

# In[ ]:

research.adeSavGolcorrelation("UK")

