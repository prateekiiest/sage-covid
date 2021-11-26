#performs 10-fold CV using lightGBM with 44 numeric features (measurement, condition, age) with with hyper-parameter optimization using RandomizedSearchCV

import datetime as dt
import numpy as np
from datetime import date
import pandas as pd
import sklearn
from sklearn.model_selection import cross_val_predict
from joblib import dump
#from joblib import load
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import make_scorer
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
#from scipy.stats import uniform
#from scipy.stats import loguniform
import lightgbm as lgb
import math
from sklearn.model_selection import train_test_split

def auprc(y_true, y_pred):

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        return auc(recall, precision)

print(datetime.now())
print("Load measurement.csv")

measurement = pd.read_csv('../data/release_07-06-2020/training/measurement.csv',usecols = ['measurement_concept_id','value_as_number','person_id', 'measurement_date', 'measurement_time'])

measurement_feature_concept_ids = ['3020891', '3027018', '3012888', '3004249', '3023314','3013650','3004327','3016502', '3010156', '3023091', '3024929', '3000963']
condition_feature_concept_ids = ['254761', '259153', '378253', '437663', '28060','4305080', '4223659']

measurement = measurement.dropna(subset = ['measurement_concept_id'])
measurement = measurement.astype({"measurement_concept_id": int})
measurement = measurement.astype({"measurement_concept_id": str})
measurement['value_as_number'] = measurement['value_as_number'].fillna(-10)
measurement['measurement_date'] = measurement['measurement_date'].fillna('1900-01-01')
measurement['measurement_time'] = measurement['measurement_time'].fillna('1900-01-01')

#print(measurement['person_id'][0])
print(measurement.shape)

print(datetime.now())

print("Load condition.csv")
condition = pd.read_csv("../data/release_07-06-2020/training/condition_occurrence.csv",usecols = ['condition_concept_id','person_id', 'condition_start_date', 'condition_end_date'])

condition = condition.dropna(subset = ['condition_concept_id'])
condition = condition.astype({"condition_concept_id": int})
condition = condition.astype({"condition_concept_id": str})
condition['condition_end_date'] = condition['condition_end_date'].fillna('2100-01-01')
condition['condition_start_date'] = condition['condition_start_date'].fillna('1900-01-01')

print(datetime.now())

print("Load person.csv")

person = pd.read_csv('../data/release_07-06-2020/training/person.csv')
today = date.today().year

'''generate the feature set'''
print(datetime.now())
print("Define feature arrays")

person = person.drop_duplicates(subset = ['person_id'])
person_index = dict(zip(person.person_id, range(len(person.person_id))))
measurement_feature_index = dict(zip(measurement_feature_concept_ids, range(len(measurement_feature_concept_ids))))
condition_feature_index = dict(zip(condition_feature_concept_ids, range(len(condition_feature_concept_ids))))
n_persons = person.shape[0]
n_measurement_types = len(measurement_feature_concept_ids)

X_min = np.zeros((len(person_index), n_measurement_types))
X_max = np.zeros((len(person_index), n_measurement_types))
X_ave = np.zeros((len(person_index), n_measurement_types))
X_min[:] = -10
X_max[:] = -10
X_ave[:] = -10

#X_measurements = np.zeros((len(person_index), 3*n_measurement_types))
#X_measurements[:] = -10

n_condition_feature_groups = len(condition_feature_concept_ids)
X_condition = np.zeros((len(person_index), n_condition_feature_groups))
#X_conditions[:] = -10

X_age = np.zeros((len(person_index), 1))
#X_ages[:] = -10

print(datetime.now())
person_ids_after_covid_set = set()

print("measurements")

for i in measurement_feature_concept_ids:

        index_f = measurement_feature_index[i]

        #subm = measurement[measurement['measurement_concept_id'] == i]
        subm = measurement.query('measurement_concept_id==@i')
        #print(subm.equals(subm_2))
        subm_after_covid = subm.query('measurement_date > "2019-11-17"')
        subm_before_covid = subm.query('measurement_date <= "2019-11-17"')

        subm_after_covid_min = subm_after_covid.groupby('person_id')['value_as_number'].min()
        subm_after_covid_max = subm_after_covid.groupby('person_id')['value_as_number'].max()
        subm_after_covid_ave = subm_after_covid.groupby('person_id')['value_as_number'].mean()
        
        subm_before_covid_min = subm_before_covid.groupby('person_id')['value_as_number'].min()
        subm_before_covid_max = subm_before_covid.groupby('person_id')['value_as_number'].max()
        subm_before_covid_ave = subm_before_covid.groupby('person_id')['value_as_number'].mean()

        for person_id in subm_after_covid_min.keys():

                index_p = person_index[person_id]

                X_min[index_p][index_f] = subm_after_covid_min[person_id]
                X_max[index_p][index_f] = subm_after_covid_max[person_id]
                X_ave[index_p][index_f] = subm_after_covid_ave[person_id]
                person_ids_after_covid_set.add(person_id)

        person_ids_before_covid_set = set(subm_before_covid_min.keys())
        person_ids_before_covid_but_not_after_covid_set = person_ids_before_covid_set.difference(person_ids_after_covid_set)

        for person_id in person_ids_before_covid_but_not_after_covid_set:
        
                index_p = person_index[person_id]

                X_min[index_p][index_f] = subm_before_covid_min[person_id]
                X_max[index_p][index_f] = subm_before_covid_max[person_id]
                X_ave[index_p][index_f] = subm_before_covid_ave[person_id]

        #print(datetime.now())

print(datetime.now())
print("conditions")

for i in condition_feature_concept_ids:

        index_f = condition_feature_index[i]
        subm = condition.query('condition_concept_id==@i')
        subm_after_covid = subm.query('condition_end_date > "2019-11-17"')
        #subm_before_covid = subm.query('condition_end_date <= 2019-11-17')

        #condition_person_ids = subm_after_covid.groupby('person_id')['person_id']
        
        for person_id in subm_after_covid.drop_duplicates(subset = ['person_id'])['person_id']:

                index_p = person_index[person_id]
                X_condition[index_p][index_f] = 1.0

print(datetime.now())

print("ages")

for i in range(n_persons):

        #print(i)
        person_id = person['person_id'][i]
        year_of_birth = person['year_of_birth'][i]
        age = today - year_of_birth
        index_p = person_index[person_id]
        index_f = 0
        X_age[index_p, index_f] = age

print(datetime.now())

print("concatenations")
scaler_measurement_filename = "scaler_measurement.save"
scaler_age_filename = "scaler_age.save"

#Need to load the saved scaler objects later using 
#from joblib import load
#scaler_measurement = load("scaler_measurement.save")
#scaler_age = load("scaler_age.save")

X = np.concatenate((X_min, X_max), axis=1)
X_measurement = np.concatenate((X, X_ave), axis=1)
#scaler_measurement = StandardScaler()
#X_measurement_scaled = scaler_measurement.fit_transform(X_measurement)
#dump(scaler_measurement, scaler_measurement_filename)
#X = np.concatenate((X_measurement_scaled, X_condition), axis=1)
X = np.concatenate((X_measurement, X_condition), axis=1)

#scaler_age = StandardScaler()
#X_age_scaled = scaler_age.fit_transform(X_age)
#dump(scaler_age, scaler_age_filename)
#X = np.concatenate((X, X_age_scaled), axis=1)
X = np.concatenate((X, X_age), axis=1)

print("X.shape")
print(X.shape)

print(datetime.now())

print("true labels")

gs = pd.read_csv('../data/release_07-06-2020/training/goldstandard.csv')
person_status = person.merge(gs, how = 'left', on = ['person_id'])
person_status.drop_duplicates(subset=['person_id'], keep = 'first',inplace = True)
Y =  np.array(person_status[['status']]).ravel()
print("Y.shape")
print(Y.shape)
#clf = LogisticRegressionCV(cv = 10, penalty = 'l2', tol = 0.0001, fit_intercept = True, intercept_scaling = 1, class_weight = None, random_state = None,
#max_iter = 100, verbose = 0, n_jobs = None, scoring='roc_auc').fit(X,Y)
print(datetime.now())

X_remain, X_opt, Y_remain, Y_opt = train_test_split(X, Y, test_size=0.25, stratify=Y)

print("model training and cv")

auprc_score = make_scorer(auprc, greater_is_better=True)

#xgb = XGBClassifier(objective= 'binary:logistic', n_jobs=4, random_state=0)
lgb_classifier = lgb.LGBMClassifier(objective='binary')

parameters = { 'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
        'n_estimators': range(50, 200, 50),
        'max_depth': range(5,9,1),
        'num_leaves': [25, 31, 61, 81, 127, 197, 231, 275, 302],
        'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0]
}

clf = RandomizedSearchCV(lgb_classifier, parameters, random_state=0, n_iter=500, n_jobs=4, verbose=1, scoring=auprc_score, cv=5)

#X = X[:9000,:]
#Y = Y[:9000]

search = clf.fit(X_opt, Y_opt)
best_params = search.best_params_
learning_rate_opt = best_params['learning_rate']
n_estimators_opt = best_params['n_estimators']
max_depth_opt = best_params['max_depth']
num_leaves_opt = best_params['num_leaves']
colsample_bytree_opt = best_params['colsample_bytree']

print(datetime.now())
print("Model training with optimum hyper-parameters")

lgb_classifier = lgb.LGBMClassifier(objective='binary', n_jobs=4, random_state=0, learning_rate=learning_rate_opt, n_estimators=n_estimators_opt, max_depth=max_depth_opt, num_leaves=num_leaves_opt, colsample_bytree=colsample_bytree_opt).fit(X,Y)

#******************************************
#NEED TO REMOVE THESE LINES
#y_pred = cross_val_predict(svm, X, Y, cv=3, method='predict_proba')
#print(y_pred.shape)
#precision, recall, thresholds = precision_recall_curve(Y, y_pred[:,1])
#auprc = auc(recall, precision)
#print("AUPRC={:.3f}".format(auprc))
#******************************************

print(datetime.now())

dump(lgb_classifier, '../model/baseline.joblib')
print("Training stage finished")
#print(clf.score(X,Y))

