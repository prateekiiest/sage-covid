import numpy as np
from datetime import date
import pandas as pd
import sklearn
from joblib import load
import sys
from datetime import datetime
from sklearn import preprocessing
import math

measurement_concept_ids_filename = "../model/measurement_concept_ids.list"
best_n_features_selected_measurement_filename = "../model/best_n_features_selected_measurement"

print(datetime.now())
print("Load measurement.csv")

measurement = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/evaluation/measurement.csv',usecols = ['measurement_concept_id','value_as_number','person_id', 'measurement_date', 'measurement_time'])
measurement = measurement.dropna(subset = ['measurement_concept_id'])
measurement = measurement.astype({"measurement_concept_id": int})
measurement = measurement.astype({"measurement_concept_id": str})
measurement['value_as_number'] = measurement['value_as_number'].fillna(-10)
measurement['measurement_date'] = measurement['measurement_date'].fillna('1900-01-01')
measurement['measurement_time'] = measurement['measurement_time'].fillna('1900-01-01')
print(measurement.shape)

print(datetime.now())

measurement_feature_concept_ids = load(open(measurement_concept_ids_filename, "rb"))
best_n_features_selected_measurement = load(open(best_n_features_selected_measurement_filename, "rb"))

print("Load person.csv")

person = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/evaluation/person.csv')
today = date.today().year

'''generate the feature set'''
print(datetime.now())
print("Define feature arrays")

person = person.drop_duplicates(subset = ['person_id'])
person_indices = dict(zip(person.person_id, range(len(person.person_id))))
measurement_feature_indices = dict(zip(measurement_feature_concept_ids, range(len(measurement_feature_concept_ids))))

n_persons = person.shape[0]

print(n_persons)

print(datetime.now())

print("measurements")

le = preprocessing.LabelEncoder()

n_measurement_features = len(measurement_feature_concept_ids)

X_min = np.zeros((len(person_indices), n_measurement_features))
X_max = np.zeros((len(person_indices), n_measurement_features))
X_ave = np.zeros((len(person_indices), n_measurement_features))
X_min[:] = -10
X_max[:] = -10
X_ave[:] = -10

person_ids_after_covid_set = set()

measurement = measurement.query('measurement_date > "2020-01-01"')

for i in measurement_feature_concept_ids:

        index_f = measurement_feature_indices[i]

        #subm = measurement[measurement['measurement_concept_id'] == i]
        subm = measurement.query('measurement_concept_id in @i')

        if (i == '3003694'): #blood group and Rh group 

                subm = subm.query('value_as_number != -10')
                if (subm.empty):
                        continue

                subm = subm.drop_duplicates(subset = 'person_id')
                for person_id in subm['person_id']:
                        index_p = person_indices[person_id]
                        X_min[index_p][index_f] = subm.loc[subm['person_id']==person_id, 'value_as_number'].iloc[0]
                        X_max[index_p][index_f] = X_min[index_p][index_f]
                        X_ave[index_p][index_f] = X_min[index_p][index_f]
                continue

        subm_after_covid = subm.query('measurement_date > "2020-01-01"')

        subm_after_covid_min = subm_after_covid.groupby('person_id')['value_as_number'].min()
        subm_after_covid_max = subm_after_covid.groupby('person_id')['value_as_number'].max()
        subm_after_covid_ave = subm_after_covid.groupby('person_id')['value_as_number'].mean()

        for person_id in subm_after_covid_min.keys():

                index_p = person_indices[person_id]

                X_min[index_p][index_f] = subm_after_covid_min[person_id]
                X_max[index_p][index_f] = subm_after_covid_max[person_id]
                X_ave[index_p][index_f] = subm_after_covid_ave[person_id]
                person_ids_after_covid_set.add(person_id)

if ('3003694' in measurement_feature_indices):

        feature_index = measurement_feature_indices['3003694']
        le.fit(X_min[:, feature_index])
        X_min[:, feature_index] = le.transform(X_min[:, feature_index])
        le.fit(X_max[:, feature_index])
        X_max[:, feature_index] = le.transform(X_max[:, feature_index])
        le.fit(X_ave[:, feature_index])
        X_ave[:, feature_index] = le.transform(X_ave[:, feature_index])

X_min_selected = X_min[:, 0:best_n_features_selected_measurement]
X_max_selected = X_max[:, 0:best_n_features_selected_measurement]
X_ave_selected = X_ave[:, 0:best_n_features_selected_measurement]

X_measurement_selected = np.concatenate((X_min_selected, X_max_selected), axis=1)
X_measurement_selected = np.concatenate((X_measurement_selected, X_ave_selected), axis=1)
X_selected = X_measurement_selected

selectorfile = '../model/feature_selector_stage_2.sav'
selector = load(open(selectorfile, 'rb'))
X_selected = selector.transform(X_selected)

print("X_selected.shape")
print(X_selected.shape)

print(datetime.now())

person_id = person[['person_id']]
clf =  load('../model/baseline.joblib')
Y_pred = clf.predict_proba(X_selected)[:,1]
output = pd.DataFrame(Y_pred,columns = ['score'])
output_prob = pd.concat([person_id,output],axis = 1)
output_prob.columns = ["person_id", "score"]
output_prob.to_csv('../output/predictions.csv', index = False)
print("Inferring stage finished")

