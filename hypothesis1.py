import numpy as np
import pandas as pd
import sklearn
import seaborn as sn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew 
%matplotlib inline

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

data= pd.read_excel('hypothesis1.xlsx', sheetname='Sheet1')

print("Column headings:")
print(data.columns)

str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in data.iteritems():
    if type(colvalue[2]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = data.columns.difference(str_list) 
# Create Dataframe containing only numerical features
data_num = data[num_list]
f, ax = plt.subplots(figsize=(14, 11))
plt.title('Pearson Correlation of Video Game Numerical Features')
# Draw the heatmap using seaborn
sns.heatmap(data_num.astype(float).corr(),linewidths=0.25,vmax=1.0, 
            square=True, cmap="cubehelix_r", linecolor='k', annot=True)
			
fig, ax = plt.subplots()
ax.scatter(x = data['sanitation_2015_total'], y = data['Under5_Mortality_Rate_Male'])
plt.ylabel('Under5_Mortality_Rate_Male', fontsize=13)
plt.xlabel('Prim_school_net_enrol_ratio', fontsize=13)
plt.show()


X=data[['Prim_school_net_enrol_ratio','water_2015_total','polio3','sanitation_2015_total','Life_expectancy_females','Adult_literacy_rate_females',
'EnrolmentRatio_PrimaryGER','EnrolmentRatio_SecondaryGER','Antenatalcare_AtleastOneVisit','Antenatalcare_Atleast4Visit'
,'DeliverCare_Skilled_birth_attendant','youth_litereacy_male','MobilePhone','InternetUser','Lower_Sec_School_Gross_Enroll_Ratio','Upper_Sec_School_Gross_Enrol_Ratio']]
Y=data[['Under5_Mortality_Rate_Male']]

from sklearn.linear_model import LinearRegression

lreg = LinearRegression()
lreg.fit(X, Y)
print(lreg.score(X, Y))

import statsmodels.api as sm
from scipy.stats.mstats import zscore

print(sm.OLS(zscore(Y), zscore(X)).fit().summary())

import statsmodels.api as sm
from scipy.stats.mstats import zscore
sanitation=zscore(data[['sanitation_2015_total']])
internet=zscore(data[['InternetUser']])
water=zscore(data[['water_2015_total']])
literacy=zscore(data[['youth_litereacy_male']])
healthcare=zscore(data[['Antenatalcare_AtleastOneVisit']])


Five_Factor_Index=[]

sum_list=sanitation+internet+water+literacy+healthcare

Five_Factor_Index=0.2*sum_list

print(Five_Factor_Index)

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaled_index = scaler.fit_transform(Five_Factor_Index)

print(scaled_index)

keys=data[['Countries']]
print(keys)

final_df={'Countries':keys,'Rank':scaled_index}

print(final_df)

