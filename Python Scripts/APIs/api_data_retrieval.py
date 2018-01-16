
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from scipy import signal
from sklearn import neighbors
import math

import itertools, os
import datetime as dt
import dateutil.relativedelta as reldelta
import requests as rq
import json
import xml.etree.ElementTree as ET

import plotly as py
from plotly.offline import download_plotlyjs
py.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import quandl as qd
qd.ApiConfig.api_key = "-o5sCYdh6rHfBtU_RVto"
from fredapi import Fred
fred = Fred(api_key='11566066f130f6f1d3af60679239e53e')
import googlefinance.client as gf
import pandasdmx as sdmx

import win32com
import win32com.client as cl # note due to use of win32 this will not work on apple mac computers
from win32com.client import makepy
import sys

# sys.argv = ["makepy", r"Excel.Application.16"]
# cl.makepy.main()
#add jaso
dir_bus_US = '../Business Cycles/US/Data/'
dir_pred_US = '../Bond Price Predictive Model/US/'
dir_pred_UK = '../Bond Price Predictive Model/UK/'
dir_pred_JPN = '../Bond Price Predictive Model/JPN/'
dir_pred_AUS = '../Bond Price Predictive Model/AUS/'
dir_pred_CND = '../Bond Price Predictive Model/CND/'
dir_pred_GRM = '../Bond Price Predictive Model/GRM/'
    
dirs_excel = {'dir_bus_US':dir_bus_US,'dir_pred_US':dir_pred_US,'dir_pred_UK':dir_pred_UK, 'dir_pred_JPN': dir_pred_JPN, 'dir_pred_AUS':dir_pred_AUS, 'dir_pred_CND':dir_pred_CND, 'dir_pred_GRM':dir_pred_GRM}


# In[2]:

##filters a dataset, between two given dates. Alternatively, if auto = True is passed as a parameter the end data is automaticaly set to 10 months prior to todays date
def dateFilter(data, auto = False ,start_date = "1975-11-30" , end_date = (dt.datetime.today()+reldelta.relativedelta(day=31,months=-10))):
    print("Filtering data to the range of {} to {}".format(start_date, end_date))
    if auto:
        not_nan_indicators_on_col_C = [False if math.isnan(x) else True for x in data.loc[:,'C']]
        start_date = data[(not_nan_indicators_on_col_C)]['Date'].iloc[0]
                    #uses column 3 as main statistic to find out when last value is
    else:
        start_date =  dt.datetime.strptime(start_date, '%Y-%m-%d')
                       ##taking out values of the form 00:00:00. These are rows with no statistic value
    data = data[[True if type(x) in [dt.datetime, pd.Timestamp ] else False for x in data['Date']]]
    data = data[ (data['Date'] > start_date) ]
    data = data[(data['Date'] < end_date)]
    data.reset_index(inplace=True,drop=True)
    return data


# In[3]:

#note used
#Given a dataset and names of specific coloumns/features, pairwise multiplication is performed on all combinations of two coloumns to produce a extended set of features
def xyXxColoumns(data,cols,replace=False):
    i = 0
    temp_xy_cols= pd.DataFrame() 
    cols_to_mult = data.loc[:,cols]
    for n in cols:
        m = list(cols).index(n)
        while m < len(cols):
            temp_xy_cols=pd.concat([temp_xy_cols,cols_to_mult.loc[:,n].mul(cols_to_mult.iloc[:,m],axis='index')],axis='columns', join='outer')
            m = m+1
    temp_xy_cols.columns = ['{0}-{1}'.format(cols[y],cols[z]) for y in list(range(0,len(cols))) for z in list(range(0,len(cols))) if z>=y ]
    
    if replace is True:
        data = pd.concat([data,temp_xy_cols],axis='columns',join='inner')
        data.reset_index(inplace=True, drop=True)
        return data
    return(data, temp_xy_cols)


# In[4]:

#Given a dataset and specified coloumns, col, and a number of shift(lag) n, for each coloumn/feature it creates n more coloumns that are shifts of the specified coloumn
#with shift varying from 1 to n
def lags(data, cols, shifts, replace=False):
    i=0
    df_lags= pd.DataFrame()
    
    while i < shifts:
        t = (data.loc[:,cols]).shift(i+1)
        t_cols=list(t.columns.values)
        t.columns = ['{0}-lags-{1}'.format(y,z+1) for z in list(range(0,3)) for y in t_cols if ((t_cols.index(y)>=20*z) & (t_cols.index(y)<20*(z+1)) ) ]
        df_lags= pd.concat([df_lags, t],axis='columns', join='outer')
        i = i+1

    data = pd.concat([data,df_lags] ,axis=1,join='inner')
    data.reset_index(inplace=True, drop=True)

    return data.iloc[1:,:]



# In[5]:

#for a given country copys the relevant transformed time-series data from each individual excel sheet
#into one "Combined sheet" containing all data for that country
##inner joins by "YYYY-MM"

def createCombinedSheet(dir_model_country, features,two_types=False ,re_turn = False):
    
    df_combined = pd.read_excel(dirs_excel[dir_model_country]+'Combined.xlsx',sheet_name='Data',header=0,datr_parser=True)
   
    for feature in features:
        df_data = pd.read_excel(dirs_excel[dir_model_country]+feature+'_hardcoded.xlsx',sheet_name='Data',header=0,datr_parser=True)
        if two_types:
            date_col_name = 'Date'
            if feature in ['AA1','BC3']:
                df_date_var = df_data.loc[:,[date_col_name,feature]]
            else: 
                df_date_var = df_data.loc[:,[date_col_name,feature+"C", feature+"L"]]
        else:
            date_col_name = 'Month-Year'
            df_date_var = df_data.loc[:,['Month-Year',feature]]
        
        df_combined = pd.merge(df_combined,df_date_var,on=date_col_name,how='inner',sort=False) 
    
    writer = pd.ExcelWriter(dirs_excel[dir_model_country]+'Combined_hardcoded.xlsx')
    df_combined.to_excel(writer,'Data',index=False)
    
    if re_turn == True : 
        return df_combined.loc[:,features]
    


# In[6]:

##Interpolation on a given coloumn of data provided as an excel sheet
#This method is for data which represents 'levels' over 3 month intervals and as such linear inerpolation is used
def interpolationLevels(cols,dir_model_country,code ,re_turn = False):
    df_interpolated = pd.read_excel(dirs_excel[dir_model_country]+code+'.xlsx',sheetname='Data',datr_parser=True)
    for col in cols:
        temp = df_interpolated.loc[:col].interpolate(method='linear')
        df_interpolated.loc[:,col] = temp
    
    writer = pd.ExcelWriter(dirs_excel[dir_model_country]+'Combined_hardcoded.xlsx')
    df_interpolated.to_excel(writer,'Data',index=False)
    
    if re_turn == True:
        return df_interpolated.loc[:,[cols]]
    


# In[7]:

##Interpolation on a given coloumn of data provided as an excel sheet
#This method is for data which represents 'Changes' across 3 month intervals and as such forward interpolation using 'zero method' 
def InterpolationLevelChanges(cols,combined_excel_file_path, re_turn = False):
    df_uninterpolated = pd.read_excel(combined_excel_file_path+'.xlsx',sheetname='Data',header=0,datr_parser=True)
    df_interpolated = df_uninterpolated
    
    for col in cols:
        temp = df_uninterpolated.loc[:,col]
        temp = temp[::,-1].interpolate(method='zero', limit_direction='forward')[::-1].div(3)
        df_interpolated.loc[:,col]= temp
    
    writer = pd.ExcelWriter(combined_excel_file_path+'_hardcoded.xlsx')
    df_interpolated.to_excel(writer,'Data',index=False)
    
    if re_turn == True:
        return df_interpolated.loc[:,[cols]]
    


# In[1]:

##Given the codes for underlying data, this downloads the data from its source, cleans the data and saves it in an appropriate location
def retreiveData(country,features,Update=False,start_date="1966-11-30"): 
    for feature in features:
        data = retreiveDataDownload(country,feature,start_date)
        retreiveDataToExcel(data,country,feature)
        print("Underlying Data for {} for {} has been downloaded".format(feature,country))
        #add update=True logic to only download last entry

def retreiveDataDownload(country,feature,_start_date):
    c_f = "{}{}".format(country,feature)
    #Quandl Codes
    qd_stats_US = ['USMP2']
    qd_codes_US = {'USMP2':['MULTPL/SP500_REAL_PRICE_MONTH']}
    qd_stats_UK = ['UKMP4','UKAA1']
    qd_codes_UK = {'UKMP4':['BOE/IUMWRLN','BOE/IUMSNPY'],'UKMP1':['BOE/IUMWRLN','BOE/IUMSNPY'],'UKAA1':['BOE/IUMWRLN','BOE/IUMSNPY']}
    qd_stats_JPN = ['JPNI2', 'JPNGM1'] # JPNMP1 has a fault on quandls end temporarily removed
    qd_codes_JPN = {'JPNI2':['MOFJ/INTEREST_RATE_JAPAN_9Y'],'JPNGM1':['RATEINF/CPI_JPN'],'JPNMP1':['MOFJ/INTEREST_RATE_JAPAN_9Y','MOFJ/INTEREST_RATE_JAPAN_5Y']}
    qd_stats_GRM = ['GRMMP2']
    qd_codes_GRM = {'GRMMP2' :['BUNDESBANK/BBK01_WU3141'] }
    qd_stats_AUS =[]
    qd_codes_AUS = {}
    qd_stats_CND =[]
    qd_codes_CND ={}
    qd_stats= qd_stats_US + qd_stats_UK + qd_stats_JPN + qd_stats_GRM + qd_stats_AUS + qd_stats_CND
    qd_codes = {}
    qd_codes.update(qd_codes_US), qd_codes.update(qd_codes_UK), qd_codes.update(qd_codes_JPN), qd_codes.update(qd_codes_GRM), qd_codes.update(qd_codes_AUS), qd_codes.update(qd_codes_CND)
    
    ##FRED codes
    fd_stats_US = ['USAA1','USI2','USI1','USGM2','USGM1','USFF1','USMP1','USMP4'] 
    fd_codes_US = {'USAA1':['GS5'], 'USI2':['GS10','BAA'], 'USI1':['GDPC1','NAEXKP04USQ661S'], 'USGM2':['NAEXKP04USQ661S'], 'USGM1':['CPIAUCSL'],'USFF1':['NNUSBIS'],'USMP1':['GS10','GS5'],'USMP4':['GS5']}
    fd_stats_UK =['UKI2','UKI1','UKGM2','UKGM1','UKFF1','UKMP1','UKMP2']
    fd_codes_UK = {'UKI2':['IRLTLT01GBM156N','BAA'],'UKI1':['NAEXKP01GBQ652S','NAEXKP04GBQ652S'],'UKGM2':['CLVMNACSCAB1GQUK'],'UKGM1':['GBRCPIALLMINMEI'],'UKFF1':['NNGBBIS'],'UKMP1':['IRLTLT01GBM156N'],'UKMP2':['SPASTT01GBM661N']}
    fd_stats_GRM = ['GRMI1','GRMI2' ,'GRMGM2','GRMGM1','GRMFF1']
    fd_codes_GRM = { 'GRMI1':['DEUGFCFQDSMEI','NAEXKP01DEQ661S'], 'GRMI2': ['IRLTLT01DEM156N','BAA'],'GRMGM2':['NAEXKP01DEQ661S'], 'GRMGM1':['DEUCPIALLMINMEI'],'GRMFF1':['RNDEBIS'], 'GRMMP1':['IRLTLT01DEM156N']}
    fd_stats_JPN = ['JPNMP2','JPNI2']
    fd_codes_JPN ={'JPNMP2':['SPASTT01JPM661N'], 'JPNI2':['BAA'] }
    fd_stats_AUS = ['AUSI2','AUSI1', 'AUSGM2', 'AUSFF1', 'AUSMP1', 'AUSMP2']
    fd_codes_AUS ={'AUSI2':['IRLTLT01AUM156N','BAA'],'AUSI1': ['AUSGDPRQDSMEI','NAEXKP04AUQ189S'],'AUSGM2':['AUSGDPRQDSMEI'],'AUSFF1':['NNAUBIS'],'AUSMP1':['IRLTLT01AUM156N'],'AUSMP2':['SPASTT01AUM661N']}
    fd_stats_CND = ['CNDI2','CNDI1','CNDGM2','CNDGM1','CNDFF1','CNDMP1','CNDMP2']
    fd_codes_CND ={ 'CNDI2' :['IRLTLT01CAM156N','BAA'],'CNDI1':['NAEXKP01CAQ189S','NAEXKP04CAQ189S'], 'CNDGM2':['NAEXKP01CAQ189S'],'CNDGM1':['CPALCY01CAM661N'] ,'JPNFF1':['NNJPBIS'],'CNDFF1':['NNCABIS'],'CNDMP1':['IRLTLT01CAM156N'],'CNDMP2':['SPASTT01CAM661N']}
    fred_stats= fd_stats_US + fd_stats_UK+ fd_stats_GRM + fd_stats_JPN + fd_stats_AUS + fd_stats_CND
    fred_codes={}
    fred_codes.update(fd_codes_US), fred_codes.update(fd_codes_UK), fred_codes.update(fd_codes_GRM), fred_codes.update(fd_codes_JPN), fred_codes.update(fd_codes_AUS), fred_codes.update(fd_codes_CND)
    
    ##Google Finance, OECD & IMF & LocalData

    #oecd_stats =['GRMMP4']
#     s_date = "{}-{}".format(_start_date[:4],_start_date[5:7])
#     e_date =  "{}-{}".format(dt.date.today().year,dt.date.today().month)
    #oecd_codes = {'GRMMP4':["http://stats.oecd.org/restsdmx/sdmx.ashx/GetData/MEI_FIN/IR3TIB+CCUS.DEU.M/all?startTime={0}&endTime={1}".format(s_date,e_date)]}
    imf_stats = []
                     #enter IMF parameters in the following order: ['Freq', 'Country', 'Code']
    imf_codes = {}
    
    local_data= ['GRMPMP1','GRMAA1','CNDMP1', 'CNDMP4','CNDAA1','AUSMP1','AUSMP4','AUSAA1','JPNI1','JPNGM2','JPNAA1']
    local_stats = {'GRMMP1':['https://www.bundesbank.de/cae/servlet/StatisticDownload?tsId=BBK01.WZ3404&its_csvFormat=en&its_fileFormat=csv&mode=its'],
                    'GRMMP4' : ['https://www.bundesbank.de/cae/servlet/StatisticDownload?tsId=BBK01.WZ3404&its_csvFormat=en&its_fileFormat=csv&mode=its'],
                    'GRMAA1' : ['https://www.bundesbank.de/cae/servlet/StatisticDownload?tsId=BBK01.WZ3404&its_csvFormat=en&its_fileFormat=csv&mode=its'],
                   'AUSMP1':['https://www.rba.gov.au/statistics/tables/xls/f02hist.xls?v=2017-11-20-14-18-03'],
                   'AUSMP4':['http://www.rba.gov.au/statistics/tables/xls-hist/f02dhist.xls'],
                   'AUSAA1':['http://www.rba.gov.au/statistics/tables/xls-hist/f02dhist.xls'],
                   'JPNI1':['http://www.esri.cao.go.jp/en/sna/data/sokuhou/files/2017/qe173/gdemenuea.html'],
                   'JPNGM2':['http://www.esri.cao.go.jp/en/sna/data/sokuhou/files/2017/qe173/gdemenuea.html'],
                   'JPNMP1':[dirs_excel['dir_pred_JPN']],
                   "JPNMP4":[dirs_excel['dir_pred_JPN']],
                   "JPNMP1" : ["C:/Users/Rilwa/Desktop/Algorthmic Trading Platform/Bond Price Predictive Model/JPN"],
                   "CNDMP1" : ['http://www5.statcan.gc.ca/access_acces/alternative_alternatif?l=eng&keng=7.059&kfra=7.059&teng=Download%20file%20from%20CANSIM&tfra=Fichier%20extrait%20de%20CANSIM&loc=http://www5.statcan.gc.ca/cansim/results/cansim6627822687716512013.csv'],
                   "CNDMP4" : ['http://www5.statcan.gc.ca/access_acces/alternative_alternatif?l=eng&keng=7.059&kfra=7.059&teng=Download%20file%20from%20CANSIM&tfra=Fichier%20extrait%20de%20CANSIM&loc=http://www5.statcan.gc.ca/cansim/results/cansim6627822687716512013.csv'],
                   "CNDAA1" : ['http://www5.statcan.gc.ca/access_acces/alternative_alternatif?l=eng&keng=7.059&kfra=7.059&teng=Download%20file%20from%20CANSIM&tfra=Fichier%20extrait%20de%20CANSIM&loc=http://www5.statcan.gc.ca/cansim/results/cansim6627822687716512013.csv'],
                    'AUSGM1':['http://www.abs.gov.au/AUSSTATS/abs@.nsf/DetailsPage/6401.0Sep%202017?OpenDocument#Time'],
                      'JPNAA1':[dirs_excel['dir_pred_JPN']]}
                
    #define period and interval size
    
    day1 = dt.datetime.strptime(_start_date, '%Y-%m-%d' )
    day2 = dt.date.today() - reldelta.relativedelta(day=31,months=-1)
    date_range = pd.date_range(day1, periods= ( 12*(day2.year - day1.year) + day2.month - day1.month) ,freq='M')
    data_full =pd.DataFrame(index=date_range)
    
    if c_f in fred_stats:
        data = pd.DataFrame(index=date_range)
        for code in fred_codes[c_f]:
            temp_data = pd.DataFrame(fred.get_series_first_release(series_id=code))
                        
            temp_data.index.name =None
            temp_data.index = [time.date() for time in temp_data.index]
            temp_data.index = [ (_t + dt.timedelta(-1)) if (_t.day ==1) else _t for _t in temp_data.index]
            data = pd.merge(data,temp_data,how='left',left_index=True, right_index=True,)
        data_full = pd.merge(data_full, data,how='left',left_index=True, right_index=True,sort=True)
        
    if c_f in qd_stats:
        data = pd.DataFrame(index= date_range)
        for code in qd_codes[c_f]:
            temp_data = qd.get(code,returns='pandas',start_date=_start_date)
            temp_data.index = [time.date() for time in temp_data.index]
            temp_data.index = [ (_t - dt.timedelta(days=1)) if _t.day==1 else _t for _t in temp_data.index]
            data = pd.merge(data,temp_data,how='left',left_index=True, right_index=True)
        data_full = pd.merge(data_full, data,how='left',left_index=True, right_index=True,sort=True)
    
    if c_f in imf_stats:
        data = pd.DataFrame(index=date_range)
        for code in imf_codes[c_f]:
            root_link= "http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/"
            s_date, e_date = dt.datetime.strptime(_start_date,'%Y-%m-%d').year, dt.date.today().year
            query = '{0}.{1}.{2}.?startPeriod={3}&endPeriod={4}'.format(code[0],code[1],code[2],s_date,e_date)
        
            data_xml = rq.get(root_link+query).json()
            temp_data = pd.DataFrame(data_xml['CompactData']['DataSet']['Series'])
            
            temp_data.index = [dt.datetime.strptime("{}{}".format(d[:4],d[-1:][0]*3),'%Y%m').date + pd.offsets.MonthEnd() for d in temp_data['@TIME_PERIOD']]
            del temp_data['@TIME_PERIOD']
            data = pd.merge(data,temp_data,how='left',left_index=True, right_index=True)
        data_full = pd.merge(data_full, data, how='left',left_index=True, right_index=True)
    
    if c_f in local_data:
        print("Some or all Data for {} is saved locally".format(c_f))
        for code in local_stats[c_f]:
            print("The original link/s is {0} . ".format(code))
            
    data_full = data_full[~data_full.index.duplicated(keep='first')]    
    return data_full

def retreiveDataToExcel(_data,_country,_feature):   
    data = _data.reset_index().applymap(lambda x: str(x))
    row_no = _data.shape[0] +1
    col_no = _data.shape[1] + 1
    data = data.as_matrix().tolist()
    
    xl_instance = cl.gencache.EnsureDispatch('Excel.Application.16')
    xl_instance.Visible = True
    xl_path = "../Bond Price Predictive Model/{0}/{1}.xlsx".format(_country,_feature)
       
    xl_wb = openWorkbook(xl_instance, os.path.abspath(xl_path))
    wl_ws = xl_wb.Worksheets('Source')

    wl_ws.Range(wl_ws.Cells(2,1),wl_ws.Cells(row_no,col_no)).Value = data
    xl_wb.Close(True)
    
def openWorkbook(_xl_instance, _xl_path):
    try:        
        xlwb = _xl_instance.Workbooks(_xl_path)            
    except Exception as e:
        try:
            xlwb = _xl_instance.Workbooks.Open(_xl_path)
        except Exception as e:
            print(e)
            xlwb = None                    
    return xlwb
        


# In[3]:


##Adewoyin-Savitzky-Golay Filters Smoothing
##sends it from excel file to excel file
##intended to work on raw files, to be used before sending to combined

#Method
##UnSavgolfiltered values are transformed by subtracting the min value from all values
##train savgol filter on data : window size chosen to be a factor 2 bigger than polyorder
## Finds unit value of Positivity/negativity in SAVGOl filter betwen t1 and t0.
##This value (+1/-1) is then multiplied by the corresponding unSavgolfiltered values
##Allows one to distinguish between high and rising and high and dipping.

##out_col should be the general code for variable
def AdewoyinSavgolFilter(out_col,dir_model_country,windowlength=13, polyorder=4,types=['C','L'], save_xlsx= False, _auto_filter_date = False, re_turn_graph = False):
    
    for _out_col in out_col:
        
        path = dirs_excel[dir_model_country]+_out_col
        data = pd.read_excel(path+'.xlsx', sheet_name= 'Data')
        
        data = dateFilter(data,auto=_auto_filter_date )
              
        for _type in types:
            #Creating copy  of original data ex outliers
                #Since there is approx 600 values in my data and they are stable, I will assume there are at most 12 outliers since data
            _contamination =  (40/data[_type].shape[0])
            outlier_clf = neighbors.LocalOutlierFactor(n_neighbors=20,contamination=_contamination,n_jobs=1)
            data[_type+" inliers"] = outlier_clf.fit_predict( data.loc[:,_type].as_matrix().reshape(-1,1) )

            inlier_max = data[(data[_type+" inliers"]==1)][_type].max()
            inlier_min = data[(data[_type+" inliers"]==1)][_type].min()
            absolute_inlier_max = np.maximum(np.absolute(inlier_max), np.absolute(inlier_min))

            data[_type+" ex. outliers"] = [(np.sign(x)*absolute_inlier_max) if np.absolute(x)>absolute_inlier_max else x for x in data[_type]]

            ##making savgol filter and savgol period on period change
            savgol_signal = pd.DataFrame(signal.savgol_filter(data.loc[:,_type], window_length=windowlength, polyorder=polyorder))
            data["{}_Savgol_Filtered".format(_type)] = savgol_signal
            signal_change_per_period = savgol_signal.diff()
            signal_change_per_period_sign = [1 if val>0 else -1 if val<0 else 0 for val in signal_change_per_period.iloc[:,0]]
            
            #AdeSavGol output
            data[_type+" above 0"] = data[_type+" ex. outliers"] - (data[_type+" ex. outliers"].min())   
            data[_out_col+_type] = data[_type+" above 0"] * (signal_change_per_period_sign)

        if save_xlsx == True:
            writer = pd.ExcelWriter(path+'_hardcoded.xlsx')
            data.to_excel(writer,'Data',index=False)

        if re_turn_graph == True:
            #x= np.arange(len(savgol_signal)-1)
            trace0= go.Scatter(x= np.asarray(data['Date']), y=data[_type].as_matrix().flatten(), name="Original Data")
            trace1 = go.Scatter(x= np.asarray(data['Date']),y=savgol_signal.as_matrix().flatten(), name ="Savgol Filter")
            trace2 = go.Scatter(x= np.asarray(data['Date']),y=np.asarray(signal_change_per_period_sign).flatten(), name ="Savgol Filter Change")
            
            trace3= go.Scatter(x= np.asarray(data['Date']), y=data[_type+" ex. outliers"].as_matrix().flatten(), name = "L^ex-out")
            trace4= go.Scatter(x= np.asarray(data['Date']), y=data[_type+" above 0"].as_matrix().flatten(), name = "L^S2")
            trace5 = go.Scatter(x= np.asarray(data['Date']), y=data[_out_col+_type].as_matrix().flatten(),name='L_asg' )
            
            trace_set = [trace0,trace1,trace2,trace3,trace4,trace5]
            layout = go.Layout(title = _out_col+_type)
            fig = go.Figure(data=trace_set, layout=layout)

            py.offline.iplot(fig, image='png',filename="{}-{}".format(_out_col,_type))



# In[11]:


#     if c_f in oecd_stats:
#         data = pd.DataFrame(index=date_range)
#         for code in oecd_codes[c_f]:
#             response = rq.get(code)
            
#             temp_data = ET.fromstring(response.text[3:])
#             namespaces = {'main':"http://www.SDMX.org/resources/SDMXML/schemas/v2_0/generic"}                      
#             xml_dataset = temp_data.find('main:DataSet',namespaces)
#             xml_series = xml_dataset.find('main:Series',namespaces)
#             obs_list =[]
#             for obs in xml_series.findall('main:Obs',namespaces):
#                  obs_list = obs_list + [obs.find('main:Time',namespaces).text, obs.find('main:ObsValue',namespaces).get('value')] 
#             obs_array = np.asarray(obs_list).reshape(-1,2)
            
#             temp_data=pd.DataFrame( data=obs_array[:,1], columns =["Value"], index=obs_array[:,0])
                     
#             temp_data.index = [dt.datetime.strptime(_d,"%Y-%m").date() + pd.offsets.MonthEnd() for _d in temp_data.index] # was not needed
#             #temp_data.index = [date + pd.offsets.MonthEnd() for date in temp_data.index]
            
#             data = pd.merge(data,temp_data,how='left',left_index=True, right_index=True)
#         data_full = pd.merge(data_full,data,how='left',left_index=True, right_index=True)

