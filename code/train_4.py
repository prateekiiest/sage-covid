#performs 10-fold CV using logistic regression with 44 numeric features (measurement, condition, age) with undersampling applied

import datetime as dt
import numpy as np
from datetime import date
import pandas as pd
import sklearn
#from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from joblib import dump
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import sys
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler

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

X = np.concatenate((X_min, X_max), axis=1)
X = np.concatenate((X, X_ave), axis=1)
X = np.concatenate((X, X_condition), axis=1)
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

print("model training and cv")

us = RandomUnderSampler(1/1)
X_us, Y_us = us.fit_resample(X, Y)

clf = LogisticRegression(solver='saga').fit(X_us,Y_us)
#clf = RandomForestClassifier()

#******************************************
#NEED TO REMOVE THESE LINES
y_pred = cross_val_predict(clf, X_us, Y_us, cv=3, method='predict_proba')
print(y_pred.shape)
precision, recall, thresholds = precision_recall_curve(Y_us, y_pred[:,1])
auprc = auc(recall, precision)

print("AUPRC={:.3f}".format(auprc))

Y_us = np.reshape(Y_us, (Y_us.size, 1))
data_set = np.concatenate((X_us, Y_us), axis=1)
df = pd.DataFrame(data_set)
df.to_csv("../data/train_set_44_features_numeric_undersampled.csv")
#******************************************

print(datetime.now())

dump(clf, '../model/baseline.joblib')
print("Training stage finished")
#print(clf.score(X,Y))

