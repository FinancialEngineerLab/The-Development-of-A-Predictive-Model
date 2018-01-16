
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import math as mt
import copy
import scipy as sp

from sklearn import exceptions as skle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=skle.ConvergenceWarning)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
from sklearn import ensemble
from sklearn import neural_network
from sknn import mlp
from mlxtend.classifier import EnsembleVoteClassifier

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import pipeline

seed = 15
path_misc_files = "../Bond Price Predictive Model/Other/"

##imports below are for use of the R library Minerva (MINE) used in calculating MIC
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects.vectors as vc

##below imports allow the use of R in Python for the MIC correlation package written in R
base = rpackages.importr('base')
utils = rpackages.importr('utils')
stats = rpackages.importr('stats')
#utils.install_packages("devtools")
devtools = rpackages.importr('devtools')
#utils.install_packages("minerva")
minerva = rpackages.importr('minerva')



# ## Miscellaneous

# In[4]:

#imports data with or without MP4 features included
#in Half data enter C or L to select only one of C or L to include
##remove one of the C or L feature for each ASG pair of features if they have a correlation over 0.75
##Also prints the outputs of each correlation test to the Reserach Folder

def importData(country, MP4=True, c_or_l=None, max_corr=0.75,topvars=False):    
    path = "../Bond Price Predictive Model/"+country+"/Combined_hardcoded.xlsx"
    ##insert code to remove columns with 0 as target variable, i.e. those that went neither up or down
    data = pd.read_excel(path,header=0)
    data = data[(data['AA1'] != 0) ]
    data.loc[:,'AA1'] = [0 if x==-1 else x for x in data.loc[:,'AA1']] #-1 can not be used in voting Classifier due to it being negative
    
    data = adeSavGolFeatureCorrelation(data,max_corr,country)
    if topvars == True:
        data = data.loc[:,['Date','AA1','BC3', 'MP4C','MP4L','MP1C','I2L','FF1L','I2C','MP1L','MP2L']]
    data = MP4Status(data,MP4,country)
    data = removeCorL(data,c_or_l)
            
    return data

#remove, keep, or only have MP4
def MP4Status(data,MP4,country): 
    if MP4 == False:
        data = data[data.columns[~data.columns.str.contains("MP4")]]
        print("\n MP4 removed for {} analysis".format(country))
    elif MP4 == "Only":
        data = data.loc[:,['Date','AA1','BC3', 'MP4C','MP4L']]
        print("\n All removed except MP4 for {} analysis".format(country))
    elif MP4 == True:
        print("\n MP4 kept for {} analysis".format(country))
    return data

#remove all stats of C or L
def removeCorL(data,c_or_l):
    if c_or_l== "C":
        data = data[data.columns[~data.columns.str.endswith("L")]]
        data = data.drop('Month',1)
    elif c_or_l == "L":
        data = data[data.columns[~data.columns.str.endswith("C")]]
        data = data.drop('Month',1)
    return data
    
##Used for removing  the L statistic from L,C pairs which show high levels of corellation
def adeSavGolFeatureCorrelation(data,max_corr,country):
    print("Testing Correlation between corresponding L and C AVG statistics for each underlying datatype")
    under_data = ['I2','I1','GM2','GM1','FF1','MP1','MP4','MP2']
    correlation_results = pd.DataFrame(columns=under_data, index=['Pearson','MIC'])
    cols_to_remove = []
    for _ud in under_data:
        change_and_level = data[[col_name for col_name in data.columns.values if col_name.startswith(_ud)]]
        linear_corr = sp.stats.pearsonr(change_and_level.iloc[:,0],change_and_level.iloc[:,1])[0]
        nonlinear_corr = minerva.mine(vc.FloatVector(np.asarray(change_and_level.iloc[:,0])),vc.FloatVector(np.asarray(change_and_level.iloc[:,1])))[0][0]
        correlation_results[_ud]= pd.Series([linear_corr, float(nonlinear_corr)],index=['Pearson','MIC'])
        ##removing columns that contain a corellation number of 0.75
        
        if (linear_corr>max_corr) or (nonlinear_corr>max_corr):
            cols_to_remove = cols_to_remove + ["{}{}".format(_ud,"L")]
    data = data.loc[:,[_col for _col in list(data.columns.values) if _col not in cols_to_remove]]
    
    correlation_results.to_excel("../Reserach/AdeSavGol Transform/L&C correlation/{}.xlsx".format(country),engine="openpyxl")
    print(correlation_results)
    return data


# In[5]:

##takes in a country's dataframe. USE with unedited dataframes received from Import
def featureLag(country_data, types=['C','L']):
    ##inital one month lag to move features to predict next month based on this month data
    country_data[[feature for feature in country_data.columns.values if feature not in ['Date','Month']]] = country_data[[feature for feature in country_data.columns.values if feature not in ['Date','Month']]].shift()
    
    days_lags={ 'FF1': 0, 'GM1': 15, 'GM2':31, 'I1':31, 'I2':0, 'MP1':0, 'MP2':0, 'MP4':0 } ##days into month until US statistic is released
    for col_name in days_lags.keys():
        _lag = mt.ceil(days_lags[col_name]/31)
        for _type in types:
            _full_col_name = "{}{}".format(col_name,_type)
            if _full_col_name in country_data.columns.values:
                country_data.loc[:,_full_col_name] = country_data.loc[:,_full_col_name].shift(_lag)
    country_data = country_data.iloc[2:,:]
    
    return country_data  


# # Training Algos

# ### split a country's dataset by business cycle

# In[6]:

#pass in dataframe for one country
def businessCycleSplitter(list_tuples_data_country):
    print("\n Splitting Each countries data into 3 sets defined by Mentality Cycle")
    list_all_country_data = []
    for data_country in list_tuples_data_country:
        split_data_dict = {}
        split_data_dict['country_name'] = data_country[1]

        for i in np.arange(1,4):
            bus_cycle_col_name = [col for col in data_country[0].columns.values if col[:3]=="BC3"][0]
            _temp_data = data_country[0][(data_country[0][bus_cycle_col_name] == i)]
            _temp_data.reset_index(inplace=True, drop=True)
            split_data_dict.update({i:_temp_data})
        list_all_country_data = list_all_country_data + [split_data_dict]
        
        print("{} split completed".format(data_country[1]))
        
    return list_all_country_data
#keys for returned dictionary are 1,2 and 3        
#returns a dictionary of form
#L1Key = 1,2,3,Name L1Value = X[where BC==1], X[whereBC==2], country_abbreviation 


# ### divide each split_dataset into class and feature

# In[7]:

#takes in list of dictionaries from businessCycleSplitter
def classAndFeature(list_of_dictionaries):
    print("\n Splitting Each countries data into feature and target variable")
    all_country_data = {}
    for _country_dictionary in list_of_dictionaries:
        country_data = {}
        for _bus_cycle in list(_country_dictionary.keys() - ['country_name']): #_bus_cycle = 1, 2 or 3 - selecting different data split from above method
            country_data_split = {} 
            _dataset = _country_dictionary[_bus_cycle]
            Y = _dataset.loc[:,'AA1']
            X = _dataset.loc[:,[col for col in _dataset.columns.values if col not in ['AA1','BC3','Date']]]
            country_data_split.update({"Y":Y})
            country_data_split.update({"X":X})
            country_data.update({_bus_cycle:country_data_split})
            
        all_country_data.update({_country_dictionary.get('country_name'): country_data} )
        print("{} split completed".format(_country_dictionary['country_name'])) 
    return all_country_data
##Returns Dictionary of the following form
#L1Key = Country Name, L1val=Dictionary
#L2Key = Business cycle, L2val = Dictionary
#L3Key = Y or X, L3val = Targets, features


# ### finding best algos for each split dataset

# In[8]:

##For each country, for each data subset split by business cycle this finds the best N algorithim types from a range of algos
##takes in a dictionary from ClassandFeatures.
def testingAlgoTypes(_all_country_data,MP4,verbose=0):
    print("\n \n \n Testing various untrained classification algorithms on each country's seperate sub datasets ")
    all_country_data_with_algos = copy.deepcopy(_all_country_data)
    ##parameters for NeuralNet
    nn_layers = [mlp.Layer('Sigmoid',units=7, name="Layer1"),mlp.Layer("Softmax",)]
    nn_params = {'layers':nn_layers,'learning_momentum':0.9,'n_stable':10 ,'f_stable':0.01,'learning_rate':0.001,
                 'learning_rule':'adadelta','random_state':seed,'n_iter':8,'batch_size':100,'warning':None,
                 'verbose':None,'debug':False}
    
        
    max_iter_params = {'max_iter':1000}
    
    classifiers = [LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto'),
                   linear_model.RidgeClassifier(random_state=seed),
                   linear_model.LogisticRegression(solver='saga',penalty='l2',class_weight='balanced',random_state=seed),
                   neighbors.KNeighborsClassifier(n_neighbors=9,weights='distance',leaf_size=20),
                   svm.LinearSVC(class_weight='balanced',random_state=seed,dual=False),
                   ensemble.RandomForestClassifier(n_estimators=200,min_samples_split=5,min_samples_leaf=3,max_depth=3,random_state=seed), 
                   ensemble.GradientBoostingClassifier(random_state=seed, n_estimators=200, min_samples_split=5,max_features='sqrt'),
                   mlp.Classifier(**nn_params),
                   linear_model.PassiveAggressiveClassifier(max_iter=1000,random_state=seed,class_weight="balanced"),
                   linear_model.SGDClassifier(max_iter=1000,random_state=seed,class_weight='balanced', penalty='l2')]
    
    headers = ['LDA','RC','LogR','KNN','SVM','RF','GBC', 'NN','PAC','SGD']
    
    for country in all_country_data_with_algos.keys():
        df_cv_results = pd.DataFrame(columns=headers)
        for _bus_cycle in all_country_data_with_algos[country].keys() : #iterating through the different business cycles
            means_vars_for_clf = []
            result_all_clf = []
            Y_target =  all_country_data_with_algos[country][_bus_cycle].get("Y")
            X_features = all_country_data_with_algos[country][_bus_cycle].get("X")
            for _clf in classifiers:
                ##Creating Pipelines
                #standardizer = ('standardize',preprocessing.StandardScaler())
                algo = ('clf',_clf)
                steps=[]
                #steps.append(standardizer)   
                steps.append(algo)
                pipeline_clf = pipeline.Pipeline(steps)
                kfold = model_selection.KFold(n_splits =2, random_state=seed,shuffle=True)
                result_clf = model_selection.cross_val_score(pipeline_clf, np.array(X_features), Y_target.values.ravel(), cv=kfold,n_jobs=1)
                result_all_clf = result_all_clf + [result_clf.mean()] ##used to find top 3 methods
                
                means_vars_for_clf = means_vars_for_clf + ["{0:.3g}".format(result_clf.mean())] ##used for excel sheet
            df_cv_results.loc["{}-{}".format(country, _bus_cycle),:] = means_vars_for_clf
            
                ##gathering names of top three algos to be inserted into all_country_data dictionary
            top3 = sorted(result_all_clf,reverse=True)[:3]
            indexes_of_top_3 = [result_all_clf.index(x) for x in top3]
            top_3_algos_by_mean = [headers[x] for x in indexes_of_top_3] ##stored as 3 letter abbreviation of algo
            all_country_data_with_algos[country][_bus_cycle].update({"algos":top_3_algos_by_mean})
        
        if MP4 == True:
            df_cv_results.to_excel('../Reserach/Classifier Cross Validation Scores For All Countries/All/'+country+'.xlsx',index=False)
        if MP4 == "Only":
                df_cv_results.to_excel('../Reserach/Classifier Cross Validation Scores For All Countries/Only/'+country+'.xlsx',index=False)
        if MP4 == False:
                df_cv_results.to_excel('../Reserach/Classifier Cross Validation Scores For All Countries/Excl/'+country+'.xlsx',index=False)

        if verbose > 0:
            print(df_cv_results)
            print("\n")
    saveTopThreeAlgos(all_country_data_with_algos)
    
    return all_country_data_with_algos
##Returns a dict of form
#L1Keys = country names, L1Values = dictionary.
#L2Keys = x in [1,2,3], L2Values = Dictionary
#L3Keys = X, Y,algos L3Values = features, targets,List of algos'3 letter abbreviation

def saveTopThreeAlgos(all_country_data_with_algos):
    df = pd.DataFrame(columns=['First', 'Second', 'Third'])
    for country in all_country_data_with_algos.keys():
        for BC in all_country_data_with_algos[country].keys():
            algo_list = all_country_data_with_algos[country][BC].get('algos')
            df.loc["{}{}".format(country,BC)]=[algo_list[0],algo_list[1],algo_list[2]]
    df.to_excel("../Reserach/Top 3 Classifiers for Each Country MentalityCycle/Classifiers.xlsx")
    


# ### tuning algorithm

# In[9]:

##Fine Tuning parameters to for a specific model
#Takes a dictionary output in testingAlgoTypes

def fineTuneModel(_all_country_data_with_algos):
    print("\n \n Fine Tuning Parameters for the top 3 predictive algorithms for each country for each sub dataset split by Mentality/Business Cycle ")
    all_country_data_with_algos = copy.deepcopy(_all_country_data_with_algos)
    algos_dict = {"LDA":LinearDiscriminantAnalysis(), 
                  "RC":linear_model.RidgeClassifier(), 
                  "LogR":linear_model.LogisticRegression(),
                  "KNN":neighbors.KNeighborsClassifier(),
                  "SVM":svm.LinearSVC(),
                  "RF":ensemble.RandomForestClassifier(verbose=0),
                  "GBC":ensemble.GradientBoostingClassifier(verbose=0),
                  "NN": mlp.Classifier(layers= [mlp.Layer('Rectifier',units=7),mlp.Layer("Softmax",)]),
                 "PAC": linear_model.PassiveAggressiveClassifier(),
                 "SGD":linear_model.SGDClassifier()}
   
    cv_folds = 3
    n_jobs_count = np.arange(1,2)
    results = {}
    
    for country in all_country_data_with_algos.keys():
        for _bus_cycle in all_country_data_with_algos[country]:
            X = all_country_data_with_algos[country][_bus_cycle].get("X")
            Y = all_country_data_with_algos[country][_bus_cycle].get("Y")
            all_country_data_with_algos[country][_bus_cycle].update({"trained algos":[]})
            
            for _algo in all_country_data_with_algos[country][_bus_cycle].get("algos"):
            #Possible parameters for each var Parameters
            
                _parameters = {}
                
                if _algo == "LDA":
                    lda_n_components = np.arange(2,8,1)
                    shrinkage = ['auto']
                    
                    lda_solver = ['lsqr', 'eigen']
                    _parameters.update({'n_components':lda_n_components,'solver':lda_solver, 'shrinkage':shrinkage}) 
                    
                if _algo == "RC":
                    rc_class_weight = ['balanced']
                    rc_solver = ['saga','sparse_cg','svd']
                    alpha=np.arange(0.5,4.5,0.5)
                    _parameters.update( {'class_weight': rc_class_weight, 'solver':rc_solver,'alpha':alpha})
                    
                if _algo == "LogR":
                    lr_penalty = ['l1','l2']
                    lr_class_weight = ['balanced']
                    lr_solver = ['liblinear']
                    _parameters.update({'penalty':lr_penalty, 'class_weight':lr_class_weight, 'solver':lr_solver, 'random_state':[seed]})
                    
                if _algo =="KNN":
                    knn_neighbors = np.arange(2,13,1)
                    knn_weights= ['uniform','distance']
                    knn_leaf_size = np.arange(10,30,2)
                    _parameters.update({'n_neighbors': knn_neighbors, 'weights': knn_weights, 'leaf_size': knn_leaf_size})

                if _algo == "SVM":
                    ##put change of kernel in after
                    svm_weights = ['balanced']
                    dual = [False]
                    
                    _parameters.update({'class_weight':svm_weights, 'dual':dual,'random_state':[seed] })
                    
                if _algo == "RF":
                    rf_max_depth = np.arange(1,5,1)
                    n_estimators = np.asarray([200])
                    min_samples_leaf = np.arange(3,6,1)
                    min_samples_split= np.arange(3,5,1)
                    max_features = ["sqrt"]
                    _parameters.update({'max_depth':rf_max_depth, 'n_estimators':n_estimators,'min_samples_leaf':min_samples_leaf,
                                       'min_samples_split':min_samples_split,'max_features':max_features, 'random_state':[seed]})
                    
                if _algo == "GBC":
                    gb_loss =['deviance']
                    gb_max_depth = np.arange(1,5,1)
                    n_estimators = np.asarray([200])
                    min_samples_leaf = np.arange(3,6,1)
                    min_samples_leaf = np.arange(3,6,1)
                    min_samples_split= np.arange(3,6,1)
                    max_features = ["sqrt"]
                    _parameters.update({'loss':gb_loss, 'max_depth':gb_max_depth, 'min_samples_leaf':min_samples_leaf,
                                       'n_estimators':n_estimators, 'min_samples_leaf':min_samples_leaf,
                                        'max_features':max_features, 'random_state':[seed] })
                    
                if _algo == "NN":
                    layer_1 = [mlp.Layer(type="Sigmoid",units=7,name="layer1"),mlp.Layer(type="Softmax",name="layer2")]
                    #mlp.Layer('Rectifier',units=5)
                    nn_layers = [layer_1]
                    nn_regularize = ['L1']
                    learning_rate= [0.01]
                    n_iter = [1000]
                    weight_decay = [0.01]
                    learning_rule=['adadelta']
                    momentum=[0.90]
                    n_stable=np.arange(150,151,2)
                    f_stable=[0.001]
                    dropout_rate=np.asarray([0,0.25,0.5])
                    random_state=[seed]
                    nn_params = {'layers':nn_layers, 'regularize':nn_regularize,'learning_rate':learning_rate,
                                 'n_iter':n_iter,'learning_rule':learning_rule,'n_iter':n_iter,'weight_decay':weight_decay, 
                                'learning_momentum':momentum,'n_stable':n_stable, 'random_state':random_state} #hidden layer size should be average of input layer and output layer
                    _parameters.update(nn_params)             
                
                if _algo== "PAC":
                    class_weight=['balanced']
                    max_iter = np.arange(1000,10001,1)
                    _parameters.update({'class_weight':class_weight,'max_iter':max_iter,'random_state':[seed]})

                if _algo== "SGD":
                    loss = ['squared_hinge','hinge']
                    class_weight=['balanced']
                    penalty = ['l2','l1','elasticnet']
                    _parameters.update({'loss':loss, 'class_weight':class_weight, 'max_iter':[1000],'penalty':penalty,
                                        'random_state':[seed] })
                    
                _grid = model_selection.GridSearchCV(algos_dict.get(_algo), param_grid=_parameters, cv=cv_folds,n_jobs=1)
               
   
                _grid.fit(np.array(X), Y.as_matrix().flatten())
                
                trained_algo = _grid.best_estimator_
                all_country_data_with_algos[country][_bus_cycle]["trained algos"].append(trained_algo)

    return all_country_data_with_algos
##Returns a dict of form
#L1Keys = country names, L1Values = dictionary.
#L2Keys = x in [1,2,3], L2Values = Dictionary
#L3Keys = "X", "Y","algos","trained algos" L3Values = features, targets,List of algos' abbreviated,trained instance of algorithm


# # Testing trained Algorithms

# ### Voting Ensemble Test: 
#     ####1) Testing by division on country and BC split
#     ####2) Accumalating results of BC splits per country to get per country accuracy test

# In[10]:

##takes output from above method. 2nd argument is the data for test country (US)
def votingEnsembleTest(all_country_data_with_algos, test_country_data_US):
    print(" \n For each training set country for each sub dataset (split by Mentality Cycle): the top n trained algorithms form a Voting Classifiers. This Voting Classifiers is then tested on its corresponding US sub data set. An aggregate scocre for each trainging set country is calculated through an Aggregation of its 3 Voting Classifiers' performances")
    _all_country_data_with_trained_algos = copy.deepcopy(all_country_data_with_algos)
    
    for country in _all_country_data_with_trained_algos.keys():
        country_level_total_hits = 0
        for BC in _all_country_data_with_trained_algos[country].keys():
            classifiers = copy.deepcopy(_all_country_data_with_trained_algos[country][BC].get('trained algos'))
            
            clf_weights = np.asarray([1,1,1],dtype=int)
            
            Y = test_country_data_US[BC].get("Y")
            X = test_country_data_US[BC].get("X")
            
            vclf = EnsembleVoteClassifier(clfs=classifiers ,weights=clf_weights ,refit=False, voting='hard') # voting='soft'            
            
            vclf.fit(X,Y)
            y_estimate = vclf.predict(np.array(X))
            print("Voting Classifier trained on {} Mentality Cycle {} has accuracy: {}".format(country,BC ,np.mean(Y==pd.Series(y_estimate))))
            
            ##saving Country-BC split accuracy and instance of Voting Classifier score to all_country... dictionary
            _all_country_data_with_trained_algos[country][BC]['accuracy'] = np.mean(Y==y_estimate)
            _all_country_data_with_trained_algos[country][BC]['votingclassifier'] = vclf           
            country_level_total_hits = country_level_total_hits + np.sum(Y==y_estimate)
        
        record_count = test_country_data_US[1]["Y"].shape[0] + test_country_data_US[2]["Y"].shape[0] + test_country_data_US[3]["Y"].shape[0]
        _all_country_data_with_trained_algos[country]['accuracy'] = (country_level_total_hits / record_count)
        print("Aggregated Classifier trained on {} has accuracy: {} \n".format(country,_all_country_data_with_trained_algos[country]['accuracy']))
    
    return _all_country_data_with_trained_algos
##Returns a dict of form
#L1Keys = country names, L1Values = dictionary.
#L2Keys = x in [1,2,3], "accuracy" L2Values = Dictionary,Combined Accuracy across BCs per country
#L3Keys = "X", "Y","algos","trained algos","accuracy","votingclassifier" L3Values = features, targets,List of algos' abbreviated,trained instance of algorithm,accuracy on US Data


# ### 2nd Layer Ensemble Test Combining: Ensemble of BC Ensembles by top country_BC

# In[11]:

###Using the accuracy scores for individual Country-BC level ensembles, I will select the following ensemble
##Voting Classifier using top 2(maybe 3) ensembles
def votingEnsembleTest2ndLayer(a_c_d_w_t_a_and_acc_scores,test_country_data,ensemble_size=3):
    print("For each mentality cycle, the top 3 or 2 Voting Classifiers across countries are combined to form a 2nd Level Voting Classifier")
    a_c_d_w_t_a_and_acc_scores2 = copy.deepcopy(a_c_d_w_t_a_and_acc_scores)
    top_ensembles_dict = votingEnsembleTest2ndLayer_baseVClassifierSelection(a_c_d_w_t_a_and_acc_scores2,ensemble_size)
    ##null before this point
    votingEnsembleTest2ndLayer_Test(top_ensembles_dict,test_country_data)
    
    

def votingEnsembleTest2ndLayer_baseVClassifierSelection(a_c_d_w_t_a_and_acc_scores2,ensemble_size):
    top_ensembles_dict = {}
    for BC in list(a_c_d_w_t_a_and_acc_scores2['UK'].keys() - ['accuracy']):
        top_accuracies = []
        for country in a_c_d_w_t_a_and_acc_scores2.keys():
            accuracy = a_c_d_w_t_a_and_acc_scores2[country][BC].get('accuracy')
            vclf = a_c_d_w_t_a_and_acc_scores2[country][BC].get('votingclassifier')
            
            if len(top_accuracies)<ensemble_size :  ##top2accuracies is a list of lists of form [[country,vclf,accuracy]]
                top_accuracies = top_accuracies + [[country,vclf,accuracy]]
            
            else:
                acc_list = [acc for acc in [sub_list[2] for sub_list in top_accuracies]]
                if accuracy > min(acc_list):
                    index_of_min = acc_list.index(min(acc_list)) # find index of lowest accuracy
                    top_accuracies[index_of_min] = [country,vclf,accuracy]
        
        top_ensembles_dict[BC] = top_accuracies
    return top_ensembles_dict
        ##top ensembles is a dictionary of the form:
        #L1 keys: 1,2,3 L1Values: array of arrays of form [[country,vclf,accuracy],]

def votingEnsembleTest2ndLayer_Test(top_ensembles_dict,test_country_data):
    hit_count=0
    for BC in top_ensembles_dict.keys():
        classifiers= [_vclf for _vclf in [sub_list[1] for sub_list in top_ensembles_dict[BC]]]
        _weights = np.asarray([1]*len(classifiers))
        vclf_layer2 = EnsembleVoteClassifier(clfs=classifiers, weights=_weights,refit=False)
        Y = test_country_data[BC]["Y"]
        X = test_country_data[BC]["X"]
        vclf_layer2.fit(X,Y)
        y_estimate = vclf_layer2.predict(X)
        print("Mentality Cycle {} 2nd Layer Voting Classifier Ensemble has accuracy: {}".format(BC ,np.mean(Y==y_estimate)))
        hit_count = hit_count + np.sum(Y==y_estimate) ##calc overall performance of top 3 classifiers for each region
    
    total_obvs = test_country_data[1]["Y"].shape[0]+test_country_data[2]["Y"].shape[0]+test_country_data[3]["Y"].shape[0]
    overall_hit_rate = hit_count/total_obvs
    print("Aggregated accuracy of 2nd Layer Voting Classifiers is: {}".format(overall_hit_rate))
        

