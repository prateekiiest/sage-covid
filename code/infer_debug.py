import datetime as dt
import numpy as np
from datetime import date
import pandas as pd
import sklearn
from sklearn.model_selection import cross_val_predict
from joblib import load
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def read_data_frame(csv_filename, usecols_list, dropna_subset, concept_id_name, fillna_field, fillna_value, fillna_field_2='', fillna_value_2=0.0):

        df = pd.read_csv(csv_filename, usecols=usecols_list)
        df = df.dropna(subset = dropna_subset)
        df = df.astype({concept_id_name: int})
        df = df.astype({concept_id_name: str})
        df[fillna_field] = df[fillna_field].fillna(fillna_value)
        if (not fillna_field_2):
                df[fillna_field_2] = df[fillna_field_2].fillna(fillna_value_2)

        return df

def prepare_feature_matrix_measurement_slow(person_ids, person_indices, measurement, measurement_feature_indices, X_min, X_max, X_ave):

        for person_id in person_ids:

                index_p = person_indices[person_id]
                measurement_concept_ids = []

                subm_after_covid = measurement.query('measurement_date > "2020-01-01" and person_id == @person_id')
                subm_before_covid = measurement.query('measurement_date <= "2020-01-01" and person_id == @person_id')

                if (not subm_after_covid.empty):

                        subm_covid_min = subm_after_covid.groupby('measurement_concept_id')['value_as_number'].min()
                        subm_covid_max = subm_after_covid.groupby('measurement_concept_id')['value_as_number'].max()
                        subm_covid_ave = subm_after_covid.groupby('measurement_concept_id')['value_as_number'].mean()

                        measurement_concept_ids = subm_covid_min.index.tolist()

                elif (not subm_before_covid.empty):

                        subm_covid_min = subm_before_covid.groupby('measurement_concept_id')['value_as_number'].min()
                        subm_covid_max = subm_before_covid.groupby('measurement_concept_id')['value_as_number'].max()
                        subm_covid_ave = subm_before_covid.groupby('measurement_concept_id')['value_as_number'].mean()
                        measurement_concept_ids = subm_covid_min.index.tolist()

                for measurement_concept_id in measurement_concept_ids:
                        if (measurement_concept_id not in measurement_feature_indices):
                                continue
                        index_f = measurement_feature_indices[measurement_concept_id]
                        X_min[index_p][index_f] = subm_covid_min[measurement_concept_id]
                        X_max[index_p][index_f] = subm_covid_max[measurement_concept_id]
                        X_ave[index_p][index_f] = subm_covid_ave[measurement_concept_id]

        le = preprocessing.LabelEncoder()

        if ('3003694' in measurement_feature_concept_ids):

                index_f = measurement_feature_indices['3003694']
                le.fit(X_min[:, index_f])
                X_min[:, index_f] = le.transform(X_min[:, index_f])
                le.fit(X_max[:, index_f])
                X_max[:, index_f] = le.transform(X_max[:, index_f])
                le.fit(X_ave[:, index_f])
                X_ave[:, index_f] = le.transform(X_ave[:, index_f])

        return X_min, X_max, X_ave, le

def prepare_feature_matrix_measurement(measurement_feature_indices, person_indices, X_min, X_max, X_ave):

        measurement_feature_concept_ids = list(measurement_feature_indices.keys())
        le = preprocessing.LabelEncoder()
        person_ids_after_covid_set = set()

        for i in measurement_feature_concept_ids:

                index_f = measurement_feature_indices[i]

                #subm = measurement[measurement['measurement_concept_id'] == i]
                subm = measurement.query('measurement_concept_id in @i')

                if (i == '3003694'): #blood group and Rh group

                        subm = subm.query('value_as_number != 0.0')
                        if (subm.empty):
                                continue

                        subm = subm.drop_duplicates(subset = 'person_id')
                        for person_id in subm['person_id']:
                                index_p = person_indices[person_id]
                                X_min[index_p][index_f] = subm.loc[subm['person_id']==person_id, 'value_as_number'].iloc[0]
                                X_max[index_p][index_f] = X_min[index_p][index_f]
                                X_ave[index_p][index_f] = X_min[index_p][index_f]

                        continue

                #print(subm.equals(subm_2))
                subm_after_covid = subm.query('measurement_date > "2020-01-01"')
                subm_before_covid = subm.query('measurement_date <= "2020-01-01"')

                if (subm_after_covid.empty):
                        subm_after_covid = subm_before_covid
                if (subm_before_covid.empty):
                        continue

                subm_after_covid_min = subm_after_covid.groupby('person_id')['value_as_number'].min()
                subm_after_covid_max = subm_after_covid.groupby('person_id')['value_as_number'].max()
                subm_after_covid_ave = subm_after_covid.groupby('person_id')['value_as_number'].mean()

                subm_before_covid_min = subm_before_covid.groupby('person_id')['value_as_number'].min()
                subm_before_covid_max = subm_before_covid.groupby('person_id')['value_as_number'].max()
                subm_before_covid_ave = subm_before_covid.groupby('person_id')['value_as_number'].mean()

                for person_id in subm_after_covid_min.keys():

                        index_p = person_indices[person_id]

                        X_min[index_p][index_f] = subm_after_covid_min[person_id]
                        X_max[index_p][index_f] = subm_after_covid_max[person_id]
                        X_ave[index_p][index_f] = subm_after_covid_ave[person_id]
                        person_ids_after_covid_set.add(person_id)

                person_ids_before_covid_set = set(subm_before_covid_min.keys())
                person_ids_before_covid_but_not_after_covid_set = person_ids_before_covid_set.difference(person_ids_after_covid_set)

                for person_id in person_ids_before_covid_but_not_after_covid_set:

                        index_p = person_indices[person_id]

                        X_min[index_p][index_f] = subm_before_covid_min[person_id]
                        X_max[index_p][index_f] = subm_before_covid_max[person_id]
                        X_ave[index_p][index_f] = subm_before_covid_ave[person_id]


        if ('3003694' in measurement_feature_concept_ids):

                index_f = measurement_feature_indices['3003694']
                le.fit(X_min[:, index_f])
                X_min[:, index_f] = le.transform(X_min[:, index_f])
                le.fit(X_max[:, index_f])
                X_max[:, index_f] = le.transform(X_max[:, index_f])
                le.fit(X_ave[:, index_f])
                X_ave[:, index_f] = le.transform(X_ave[:, index_f])

        return X_min, X_max, X_ave, le

print(datetime.now())
print("true labels")

measurement_concept_ids_filename = "../../model/measurement_concept_ids.list"
best_n_features_selected_measurement_filename = "../../model/best_n_features_selected_measurement"
label_encoder_filename = "../../model/label_encoder"
normalizer_filename = "../../model/normalizer"

print(datetime.now())
print("Load measurement.csv")

csv_filename = '../../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/evaluation/measurement.csv'
usecols_list = ['measurement_concept_id','value_as_number','person_id', 'measurement_date']
dropna_subset = ['measurement_concept_id']
concept_id_name = 'measurement_concept_id'
fillna_field = 'value_as_number'
fillna_value = 0.0
fillna_field_2 = 'measurement_date'
fillna_value_2 = '1900-01-01'
measurement = read_data_frame(csv_filename, usecols_list, dropna_subset, concept_id_name, fillna_field, fillna_value, fillna_field_2, fillna_value_2)

print(measurement.shape)
print(datetime.now())

measurement_feature_concept_ids = load(open(measurement_concept_ids_filename, "rb"))
best_n_features_selected_measurement = load(open(best_n_features_selected_measurement_filename, "rb"))
le = load(open(label_encoder_filename, "rb"))
transformer = load(open(normalizer_filename, "rb"))

print("Load person.csv")

person = pd.read_csv('../../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/evaluation/person.csv')
person_ids = person['person_id'].to_numpy()

today = date.today().year

'''generate the feature set'''
print(datetime.now())
print("Define feature arrays")

person = person.drop_duplicates(subset = ['person_id'])
person_indices = dict(zip(person.person_id, range(len(person.person_id))))
measurement_feature_indices = dict(zip(measurement_feature_concept_ids, range(len(measurement_feature_concept_ids))))

n_persons = person.shape[0]
n_measurement_types = len(measurement_feature_concept_ids)

X_min = np.zeros((len(person_indices), n_measurement_types))
X_max = np.zeros((len(person_indices), n_measurement_types))
X_ave = np.zeros((len(person_indices), n_measurement_types))

print(datetime.now())

print("measurements")

X_min, X_max, X_ave, le = prepare_feature_matrix_measurement(measurement_feature_indices, person_indices, X_min, X_max, X_ave)

print(datetime.now())

print("concatenations")

X_min_selected = X_min[:, 0:best_n_features_selected_measurement]
X_max_selected = X_max[:, 0:best_n_features_selected_measurement]
X_ave_selected = X_ave[:, 0:best_n_features_selected_measurement]

X_selected_first_stage = np.concatenate((X_min_selected, X_max_selected), axis=1)
X_selected_first_stage = np.concatenate((X_selected_first_stage, X_ave_selected), axis=1)

X_selected = X_selected_first_stage

selectorfile = '../../model/feature_selector_stage_2.sav'
selector = load(open(selectorfile, 'rb'))
X_selected = selector.transform(X_selected)

X_selected = transformer.transform(X_selected)

print("X_selected.shape")
print(X_selected.shape)

print(datetime.now())

person_id = person[['person_id']]
clf =  load('../../model/baseline.joblib')
Y_pred = clf.predict_proba(X_selected)[:,1]
output = pd.DataFrame(Y_pred,columns = ['score'])
output_prob = pd.concat([person_id,output],axis = 1)
output_prob.columns = ["person_id", "score"]
output_prob.to_csv('../../output/predictions.csv', index = False)
print("Inferring stage finished")

