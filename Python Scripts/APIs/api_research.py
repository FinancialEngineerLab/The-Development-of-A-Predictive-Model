
# coding: utf-8

# In[18]:

get_ipython().system('jupyter nbconvert --to script *.ipynb')


# In[17]:

import io, os, sys, types
import numpy as np
import sklearn as sk
import pandas as pd
import scipy as sp
import canova as cnv

import plotly as py
py.offline.init_notebook_mode()
import plotly.graph_objs as go

##using R libraries for MIC calculation
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects.vectors as vc
base = rpackages.importr('base')
utils = rpackages.importr('utils')
stats = rpackages.importr('stats')
devtools = rpackages.importr('devtools')
minerva = rpackages.importr('minerva')
sys.path.append('../Python Scripts/')
bpp, bpp_x_MP4, bpp_o_MP4 = None, None, None


# # FINAL ACCURACY SCORE FOR ALGORITHM

# In[ ]:

##True means include MP4C and MP4 as features. False the opposite. "Only" means all variables besides MP4L and MP4C are removed
def finalModelScore(MP4=True):
    if MP4 ==True:
        import bond_price_pred as bpp #normal analysis with all features
        bpp.algo.votingEnsembleTest2ndLayer(bpp.a_c_d_w_t_a_and_acc_scores,bpp.test_country_data.get('US'),2)
    if MP4==False:
        import bond_price_pred_excl_MP4 as bpp_x_MP4 #analysis with previous movement of 5Y bond removed
        bpp_x_MP4.algo.votingEnsembleTest2ndLayer(bpp_x_MP4.a_c_d_w_t_a_and_acc_scores,bpp_x_MP4.test_country_data.get('US'),2)
    if MP4=="Only":
        import bond_price_pred_only_MP4 as bpp_o_MP4 #analysis with only the 5Y bond movement included
        bpp_o_MP4.algo.votingEnsembleTest2ndLayer(bpp_o_MP4.a_c_d_w_t_a_and_acc_scores,bpp_o_MP4.test_country_data.get('US'),2)


# # Research Questions

# ## Determining Feature Importance

# In[ ]:

def featureImportance(MP4=True): #whether or not past performance of bond price is included as a predictor
    if MP4==True:
        _data = bpp.all_country_data_with_trained_algos
    elif MP4 ==False:
        _data = bpp_x_MP4.all_country_data_with_trained_algos
    
    all_feature_importance = featureImportance_Calculator(_data)
    featureImportance_Chart(all_feature_importance, _data['UK'][1]["X"].columns.get_values().tolist(),MP4)

##calculates three feature importance datasets for each country at once
def featureImportance_Calculator(all_country_data_with_trained_algos):
    all_feature_importance = {}
    for country in all_country_data_with_trained_algos.keys():
        bc_dict ={}
        for _bc in np.arange(1,4):
            _stats ={}

            try: #get instance of gradient boosted classifier to rank 
                algo =[_algo for _algo in all_country_data_with_trained_algos[country][_bc]["trained algos"] if isinstance(_algo,sk.ensemble.GradientBoostingClassifier)][0]
            except IndexError:
                X = all_country_data_with_trained_algos[country][_bc]["X"]
                Y = all_country_data_with_trained_algos[country][_bc]["Y"]
                algo = sk.ensemble.GradientBoostingClassifier()
                algo.fit(X,Y)
            
            importances = algo.feature_importances_
            stds = np.std([tree[0].feature_importances_ for tree in algo.estimators_],axis=0)
            _stats['importances'] = importances
            _stats['stds'] = stds
            bc_dict[_bc] = _stats
        all_feature_importance[country]=bc_dict

    return all_feature_importance
## returns new dictionary of the form
#L1Keys=country   L1Vals=Dictionary
#L2Keys= BCs       L2vals= Dictionary
#L3Keys = importances,std, L3values =

def featureImportance_Chart(all_feature_importance,features,MP4=True):

    for country in all_feature_importance.keys():
        for _bc in np.arange(1,4):
            
            filename = "{}{}".format(country,_bc) 
            if MP4==True:
                filepath = "../Reserach/Feature Importance/"
            elif MP4 == False:
                filepath = "../Reserach/Feature Importance/excl. MP4/"
            elif MP4 == "Only":
                filepath = "../Reserach/Feature Importance/only MP4/"
            
            
            importances = all_feature_importance[country][_bc].get("importances")
            stds = all_feature_importance[country][_bc].get("stds")
            
            indices = np.argsort(importances)[::-1]
            
            ##making a chart
            _title = "{}{} Feature Importances".format(country,_bc)
            trace = go.Bar(x=np.asarray(features)[indices], y=importances[indices], marker=dict(color='red'),
                           error_y=dict(visible=True, arrayminus=stds[indices]), opacity=0.5)
            layout = go.Layout(title=_title)
            fig = go.Figure(data=[trace], layout=layout)
            py.offline.iplot(fig, image='png',filename=filename ) 
            #py.image.save_as(fig,filename=filepath+filename+".png")
            
            #saving excel file with feature importance levels
            importance_df = pd.DataFrame({"Importance":importances[indices], "Standard dev.":stds[indices]},index=np.asarray(features)[indices])
            importance_df.to_excel("{}{}.xlsx".format(filepath,filename),engine="openpyxl")


# ## Testing initial assumption that investors responses differ by my defined business cycles

# In[ ]:

def crossBusinessCycleTest():
    data_us = bpp.test_country_data["US"]
    filepath = "../Reserach/Comparing VCs performance on alternative MCs"
    for country in bpp.a_c_d_w_t_a_and_acc_scores.keys():
        
        _cols = ["Train {}BC1".format(country),"Train {}BC2".format(country),"Train {}BC3".format(country)]
        _ind = index= ["US1","US2","US3"]
        df_results = pd.DataFrame(columns=_cols, index=_ind)
        
        for _bc in np.arange(1,4):
            vclf = bpp.a_c_d_w_t_a_and_acc_scores[country][_bc].get("votingclassifier")
            y_estimates =[]
            for _bc_Y in np.arange(1,4):
                Y = data_us[_bc_Y].get("Y")
                X = data_us[_bc_Y].get("X")
                y_estimates = y_estimates + [np.mean(vclf.predict(X)==Y)] #forms a list of the accuracies resulting from testing one trained algorithm on all US business cycles
            
            df_results.iloc[:,_bc-1] = y_estimates
        df_results.to_excel("{}/{}.xlsx".format(filepath,country),engine="openpyxl")
                


# ## Which countries are best at predicting US Data 

# In[ ]:

def countryPredComparison():
    filepath = "../Reserach/Comparison of Countries' VC when tested on US Data"
    data = bpp.a_c_d_w_t_a_and_acc_scores
    scores_df = pd.DataFrame(index=["US1","US2","US3","US-All"], columns=[country for country in data.keys()])
    
    for country in data.keys():
        accs = [data[country][1]['accuracy'],data[country][2]['accuracy'],data[country][3]['accuracy'],data[country]['accuracy']]
        scores_df[country]= accs
    scores_df.to_excel("{}/Compmarison.xlsx".format(filepath),engine="openpyxl")


# # Other

# ## Correllation between the pairs of L and C, AVG statistics for a given Country

# In[ ]:

def adeSavGolCorrelation(country):
    data = bpp.algo.importData(country)
    algos = ['I2','I1','GM2','GM1','FF1','MP1','MP4','MP2']
    correlation_results = pd.DataFrame(columns=algos, index=['Pearson','MIC'])
    
    for _algo in algos:
        change_and_level = data[[col_name for col_name in data.columns.values if col_name.startswith(_algo)]]
        linear_corr = sp.stats.pearsonr(change_and_level.iloc[:,0],change_and_level.iloc[:,1])
        
        nonlinear_corr = minerva.mine(vc.FloatVector(np.asarray(change_and_level.iloc[:,0])),vc.FloatVector(np.asarray(change_and_level.iloc[:,1])))
        
        correlation_results[_algo]= pd.Series([linear_corr[0], float(nonlinear_corr[0][0])],index=['Pearson','MIC'])
    
    
    print(correlation_results)
    correlation_results.to_excel("../Reserach/AdeSavGol Transform/L&C_corr.xlsx",engine="openpyxl")

