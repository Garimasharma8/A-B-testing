#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 17:07:01 2022

@author: garimasharma
This is a A/B testing - two sample t test to check the significance of the data
"""
#%% Import libraries 

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss


#%% load the data 

ab_data = pd.read_csv('ab_data.csv')
print(ab_data[:10]) # print first 10 rows of the data

#%% chekc the information of the data 

ab_data.info()

#%% # To make sure all the control group are seeing the old page and viceversa

print(pd.crosstab(ab_data['group'],ab_data['landing_page']))

#%% Check for multiple entries by same user in the dataset 

session_counts = ab_data['user_id'].value_counts(ascending=False)
multi_users = session_counts[session_counts>1].count()

print(f"There are {multi_users} users appearing multiple times in the data")

#%% Remove the multiple enteries from the data 

users_to_drop = session_counts[session_counts > 1].index
ab_data = ab_data[~ab_data['user_id'].isin(users_to_drop)]
print(f'The updated dataset now has {ab_data.shape[0]} entries')

#%% sampling
required_n = 4720
control_sample = ab_data[ab_data['group'] == 'control'].sample(n=required_n, random_state=22)
treatment_sample = ab_data[ab_data['group'] == 'treatment'].sample(n=required_n, random_state=22)

ab_test = pd.concat([control_sample, treatment_sample], axis=0)
ab_test.reset_index(drop=True, inplace=True)
ab_test

#%% 

conversion_rates = ab_test.groupby('group')['converted']

std_p = lambda x: np.std(x, ddof=0)              # Std. deviation of the proportion
se_p = lambda x: ss.sem(x, ddof=0)            # Std. error of the proportion (std / sqrt(n))

conversion_rates = conversion_rates.agg([np.mean, std_p, se_p])
conversion_rates.columns = ['conversion_rate', 'std_deviation', 'std_error']


conversion_rates.style.format('{:.3f}')
#%% 
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))

sns.barplot(x=ab_test['group'], y=ab_test['converted'], ci=False)

plt.ylim(0, 0.17)
plt.title('Conversion rate by group', pad=20)
plt.xlabel('Group', labelpad=15)
plt.ylabel('Converted (proportion)', labelpad=15);

#%% 

from statsmodels.stats.proportion import proportions_ztest, proportion_confint
control_results = ab_test[ab_test['group'] == 'control']['converted']
treatment_results = ab_test[ab_test['group'] == 'treatment']['converted']
n_con = control_results.count()
n_treat = treatment_results.count()
successes = [control_results.sum(), treatment_results.sum()]
nobs = [n_con, n_treat]

z_stat, pval = proportions_ztest(successes, nobs=nobs)
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs=nobs, alpha=0.05)

print(f'z statistic: {z_stat:.2f}')
print(f'p-value: {pval:.3f}')
print(f'ci 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
print(f'ci 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')