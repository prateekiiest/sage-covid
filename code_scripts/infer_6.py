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

def add_COVID_measurement_date():
    measurement = pd.read_csv("../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/evaluation/measurement.csv",usecols =['person_id','measurement_date','measurement_concept_id','value_as_concept_id'])
    measurement = measurement.loc[measurement['measurement_concept_id']==706163]
    measurement['value_as_concept_id'] = measurement['value_as_concept_id'].astype(int)
    measurement = measurement.loc[(measurement['value_as_concept_id']==45877985.0) | (measurement['value_as_concept_id']==45884084.0)]
    measurement = measurement.sort_values(['measurement_date'],ascending=False).groupby('person_id').head(1)
    covid_measurement = measurement[['person_id','measurement_date']]
    return covid_measurement

def prepare_feature_matrix_measurement(person_ids, person_indices, covid_measurement, measurement, measurement_feature_indices, X_min, X_max, X_ave):

        for person_id in person_ids:

                index_p = person_indices[person_id]
                measurement_concept_ids = []

                measurement_date_array = covid_measurement[covid_measurement['person_id'] == person_id][['measurement_date']].to_numpy()

                if (measurement_date_array.size != 0):
                        measurement_date = measurement_date_array[0][0]
                else:
                        continue

                subm_after_covid = measurement.query('measurement_date > @measurement_date and person_id == @person_id')
                subm_before_covid = measurement.query('measurement_date <= @measurement_date and person_id == @person_id')

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

def prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, concept_id_str, feature_indices, X):

        for person_id in person_ids:

                index_p = person_indices[person_id]

                measurement_date_array = covid_measurement[covid_measurement['person_id'] == person_id][['measurement_date']].to_numpy()

                if (measurement_date_array.size != 0):
                        measurement_date = measurement_date_array[0][0]
                else:
                        continue

                subm_after_covid = df.query(query_str)

                if (not subm_after_covid.empty):

                        subm_covid_count = subm_after_covid.groupby(concept_id_str)['person_id'].count()
                        concept_ids = subm_covid_count.index.tolist()

                        for concept_id in concept_ids:
                                if (concept_id not in feature_indices):
                                        continue
                                index_f = feature_indices[concept_id]
                                X[index_p][index_f] = subm_covid_count[concept_id]

        return X

print(datetime.now())

measurement_concept_ids_filename = "../../data/measurement_concept_ids.list"
condition_concept_ids_filename = "../../data/condition_concept_ids.list"
observation_concept_ids_filename = "../../data/observation_concept_ids.list"
device_exposure_concept_ids_filename = "../../data/device_exposure_concept_ids.list"
drug_exposure_concept_ids_filename = "../../data/drug_exposure_concept_ids.list"
procedure_concept_ids_filename = "../../data/procedure_concept_ids.list"
visit_concept_ids_filename = "../../data/visit_concept_ids.list"
best_n_features_selected_measurement_filename = "../../model/best_n_features_selected_measurement"
best_n_features_selected_measurement_rt_pcr_filename = "../../model/best_n_features_selected_measurement_rt_pcr"
best_n_features_selected_condition_filename = "../../model/best_n_features_selected_condition"
best_n_features_selected_observation_filename = "../../model/best_n_features_selected_observation"
best_n_features_selected_device_exposure_filename = "../../model/best_n_features_selected_device_exposure"
best_n_features_selected_drug_exposure_filename = "../../model/best_n_features_selected_drug_exposure"
best_n_features_selected_procedure_filename = "../../model/best_n_features_selected_procedure"
best_n_features_selected_visit_filename = "../../model/best_n_features_selected_visit"
one_hot_encoder_filename = "../../model/one_hot_encoder"
label_encoder_filename = "../../model/label_encoder"
normalizer_filename = "../../model/normalizer"

print(datetime.now())
print("Load measurement.csv")

measurement = pd.read_csv('../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/evaluation/measurement.csv',usecols = ['measurement_concept_id','value_as_number','person_id', 'measurement_date', 'measurement_time', 'value_as_concept_id'])
measurement = measurement.dropna(subset = ['measurement_concept_id'])
measurement = measurement.astype({"measurement_concept_id": int})
measurement = measurement.astype({"measurement_concept_id": str})
measurement['value_as_number'] = measurement['value_as_number'].fillna(0.0)
measurement['measurement_date'] = measurement['measurement_date'].fillna('1900-01-01')
measurement['measurement_time'] = measurement['measurement_time'].fillna('1900-01-01')
measurement_rt_pcr = measurement.loc[(measurement['value_as_concept_id']==45877985.0) | (measurement['value_as_concept_id']==45884084.0)]
print(measurement.shape)

print(datetime.now())

print("Load condition.csv")
condition = pd.read_csv("../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/evaluation/condition_occurrence.csv",usecols = ['condition_concept_id','person_id', 'condition_start_date', 'condition_end_date'])

condition = condition.dropna(subset = ['condition_concept_id'])
condition = condition.astype({"condition_concept_id": int})
condition = condition.astype({"condition_concept_id": str})
condition['condition_end_date'] = condition['condition_end_date'].fillna('2100-01-01')
condition['condition_start_date'] = condition['condition_start_date'].fillna('1900-01-01')

print(datetime.now())

print("Load observation.csv")
observation = pd.read_csv("../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/evaluation/observation.csv",usecols = ['observation_concept_id','person_id', 'observation_date'])

observation = observation.dropna(subset = ['observation_concept_id'])
observation = observation.astype({"observation_concept_id": int})
observation = observation.astype({"observation_concept_id": str})

print(datetime.now())

print("Load device_exposure.csv")
device_exposure = pd.read_csv("../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/evaluation/device_exposure.csv",usecols = ['device_concept_id','person_id', 'device_exposure_start_date', 'device_exposure_end_date'])

device_exposure = device_exposure.dropna(subset = ['device_concept_id'])
device_exposure = device_exposure.astype({"device_concept_id": int})
device_exposure = device_exposure.astype({"device_concept_id": str})
device_exposure['device_exposure_end_date'] = device_exposure['device_exposure_end_date'].fillna('2100-01-01')
device_exposure['device_exposure_start_date'] = device_exposure['device_exposure_start_date'].fillna('1900-01-01')

print(datetime.now())

print("Load drug_exposure.csv")
drug_exposure = pd.read_csv('../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/evaluation/drug_exposure.csv',usecols = ['drug_concept_id', 'person_id', 'drug_exposure_start_date', 'drug_exposure_end_date'])

drug_exposure = drug_exposure.dropna(subset = ['drug_concept_id'])
drug_exposure = drug_exposure.astype({"drug_concept_id": int})
drug_exposure = drug_exposure.astype({"drug_concept_id": str})
drug_exposure['drug_exposure_end_date'] = drug_exposure['drug_exposure_end_date'].fillna('2100-01-01')
drug_exposure['drug_exposure_start_date'] = drug_exposure['drug_exposure_start_date'].fillna('1900-01-01')

print(datetime.now())

print("Load procedure_occurrence.csv")

procedure = pd.read_csv('../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/evaluation/procedure_occurrence.csv',usecols = ['procedure_concept_id', 'person_id', 'procedure_date'])

procedure = procedure.dropna(subset = ['procedure_concept_id'])
procedure = procedure.astype({"procedure_concept_id": int})
procedure = procedure.astype({"procedure_concept_id": str})
procedure['procedure_date'] = procedure['procedure_date'].fillna('1900-01-01')

print(datetime.now())

print("Load visit_occurrence.csv")

visit = pd.read_csv('../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/evaluation/visit_occurrence.csv',usecols = ['visit_concept_id', 'person_id', 'visit_start_date', 'visit_end_date'])

visit = visit.dropna(subset = ['visit_concept_id'])
visit = visit.astype({"visit_concept_id": int})
visit = visit.astype({"visit_concept_id": str})
visit['visit_start_date'] = visit['visit_start_date'].fillna('1900-01-01')
visit['visit_end_date'] = visit['visit_end_date'].fillna('2100-01-01')

print(datetime.now())

measurement_feature_concept_ids = load(open(measurement_concept_ids_filename, "rb"))
condition_feature_concept_ids = load(open(condition_concept_ids_filename, "rb"))
observation_feature_concept_ids = load(open(observation_concept_ids_filename, "rb"))
device_exposure_feature_concept_ids = load(open(device_exposure_concept_ids_filename, "rb"))
drug_exposure_feature_concept_ids = load(open(drug_exposure_concept_ids_filename, "rb"))
procedure_feature_concept_ids = load(open(procedure_concept_ids_filename, "rb"))
visit_feature_concept_ids = load(open(visit_concept_ids_filename, "rb"))
best_n_features_selected_measurement = load(open(best_n_features_selected_measurement_filename, "rb"))
best_n_features_selected_measurement_rt_pcr = load(open(best_n_features_selected_measurement_rt_pcr_filename, "rb"))
best_n_features_selected_condition = load(open(best_n_features_selected_condition_filename, "rb"))
best_n_features_selected_observation = load(open(best_n_features_selected_observation_filename, "rb"))
best_n_features_selected_device_exposure = load(open(best_n_features_selected_device_exposure_filename, "rb"))
best_n_features_selected_drug_exposure = load(open(best_n_features_selected_drug_exposure_filename, "rb"))
best_n_features_selected_procedure = load(open(best_n_features_selected_procedure_filename, "rb"))
best_n_features_selected_visit = load(open(best_n_features_selected_visit_filename, "rb"))
one_hot_encoder = load(open(one_hot_encoder_filename, "rb"))
le = load(open(label_encoder_filename, "rb"))
transformer = load(open(normalizer_filename, "rb"))

print("Load person.csv")

person = pd.read_csv('../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/evaluation/person.csv')
today = date.today().year

'''generate the feature set'''
print(datetime.now())
print("Define feature arrays")

person = person.drop_duplicates(subset = ['person_id'])
person_indices = dict(zip(person.person_id, range(len(person.person_id))))
measurement_feature_indices = dict(zip(measurement_feature_concept_ids, range(len(measurement_feature_concept_ids))))
condition_feature_indices = dict(zip(condition_feature_concept_ids, range(len(condition_feature_concept_ids))))
observation_feature_indices = dict(zip(observation_feature_concept_ids, range(len(observation_feature_concept_ids))))
device_exposure_feature_indices = dict(zip(device_exposure_feature_concept_ids, range(len(device_exposure_feature_concept_ids))))
drug_exposure_feature_indices = dict(zip(drug_exposure_feature_concept_ids, range(len(drug_exposure_feature_concept_ids))))
procedure_feature_indices = dict(zip(procedure_feature_concept_ids, range(len(procedure_feature_concept_ids))))
visit_feature_indices = dict(zip(visit_feature_concept_ids, range(len(visit_feature_concept_ids))))

n_persons = person.shape[0]
n_measurement_types = len(measurement_feature_concept_ids)

X_measurement_rt_pcr = np.zeros((len(person_indices), n_measurement_types))

X_min = np.zeros((len(person_indices), n_measurement_types))
X_max = np.zeros((len(person_indices), n_measurement_types))
X_ave = np.zeros((len(person_indices), n_measurement_types))

n_condition_types = len(condition_feature_concept_ids)
X_condition = np.zeros((len(person_indices), n_condition_types))

n_observation_types = len(observation_feature_concept_ids)
X_observation = np.zeros((len(person_indices), n_observation_types))

n_device_exposure_types = len(device_exposure_feature_concept_ids)
X_device_exposure = np.zeros((len(person_indices), n_device_exposure_types))

n_drug_exposure_types = len(drug_exposure_feature_concept_ids)
X_drug_exposure = np.zeros((len(person_indices), n_drug_exposure_types))

n_procedure_types = len(procedure_feature_concept_ids)
X_procedure = np.zeros((len(person_indices), n_procedure_types))

n_visit_types = len(visit_feature_concept_ids)
X_visit = np.zeros((len(person_indices), n_visit_types))

X_age = np.zeros((len(person_indices), 1))
X_gender_race_ethnicity_raw = np.zeros((len(person_indices), 3))

covid_measurement = add_COVID_measurement_date()

person_ids = person['person_id'].to_numpy()

print(datetime.now())
print("measurements")

for i in measurement_feature_concept_ids:

        index_f = measurement_feature_indices[i]

        if (measurement_rt_pcr.empty):
                continue

        #subm = measurement[measurement['measurement_concept_id'] == i]
        subm = measurement_rt_pcr.query('measurement_concept_id in @i')

        if (subm.empty):
                continue

        for person_id in subm['person_id']:

                index_p = person_indices[person_id]
                X_measurement_rt_pcr[index_p][index_f] += 1.0

X_min, X_max, X_ave, le = prepare_feature_matrix_measurement(person_ids, person_indices, covid_measurement, measurement, measurement_feature_indices, X_min, X_max, X_ave)

print(datetime.now())
print("conditions")

df = condition
#query_str = 'condition_end_date > @measurement_date and person_id == @person_id'
query_str = 'person_id == @person_id'
concept_id_str = 'condition_concept_id'
feature_indices = condition_feature_indices
X = X_condition
X_condition = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, concept_id_str, feature_indices, X)

print(datetime.now())
print("observations")

df = observation
query_str = 'person_id == @person_id'
concept_id_str = 'observation_concept_id'
feature_indices = observation_feature_indices
X = X_observation
X_observation = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, concept_id_str, feature_indices, X)

print(datetime.now())
print("device exposures")

df = device_exposure
#query_str = 'device_exposure_end_date > @measurement_date and person_id == @person_id'
query_str = 'person_id == @person_id'
concept_id_str = 'device_concept_id'
feature_indices = device_exposure_feature_indices
X = X_device_exposure
X_device_exposure = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, concept_id_str, feature_indices, X)

print(datetime.now())
print("drug exposures")

df = drug_exposure
#query_str = 'drug_exposure_end_date > @measurement_date and person_id == @person_id'
query_str = 'person_id == @person_id'
concept_id_str = 'drug_concept_id'
feature_indices = drug_exposure_feature_indices
X = X_drug_exposure
X_drug_exposure = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, concept_id_str, feature_indices, X)

print(datetime.now())
print("procedures")

df = procedure
#query_str = 'procedure_date > @measurement_date and person_id == @person_id'
query_str = 'person_id == @person_id'
concept_id_str = 'procedure_concept_id'
feature_indices = procedure_feature_indices
X = X_procedure
X_procedure = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, concept_id_str, feature_indices, X)

print(datetime.now())
print("visits")

df = visit
#query_str = 'visit_end_date > @measurement_date and person_id == @person_id'
query_str = 'person_id == @person_id'
concept_id_str = 'visit_concept_id'
feature_indices = visit_feature_indices
X = X_visit
X_visit = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, concept_id_str, feature_indices, X)

print(datetime.now())
print("ages")

person_ids = person['person_id'].to_numpy()
year_of_births = person['year_of_birth'].to_numpy()

for i in range(n_persons):

        #print(i)
        person_id = person_ids[i]
        year_of_birth = year_of_births[i]
        age = today - year_of_birth
        index_p = person_indices[person_id]
        index_f = 0
        X_age[index_p, index_f] = age

print(datetime.now())
print("gender, race, ethnicity")

genders = person['gender_concept_id'].to_numpy()
races = person['race_concept_id'].to_numpy()
ethnicities = person['ethnicity_concept_id'].to_numpy()

for i in range(n_persons):

        person_id = person_ids[i]
        index_p = person_indices[person_id]

        X_gender_race_ethnicity_raw[index_p][0] = genders[i]
        X_gender_race_ethnicity_raw[index_p][1] = races[i]
        X_gender_race_ethnicity_raw[index_p][2] = ethnicities[i]

X_gender_race_ethnicity = one_hot_encoder.transform(X_gender_race_ethnicity_raw).toarray()

print("concatenations")

X_min_selected = X_min[:, 0:best_n_features_selected_measurement]
X_max_selected = X_max[:, 0:best_n_features_selected_measurement]
X_ave_selected = X_ave[:, 0:best_n_features_selected_measurement]

X_measurement_selected = np.concatenate((X_min_selected, X_max_selected), axis=1)
X_measurement_selected = np.concatenate((X_measurement_selected, X_ave_selected), axis=1)
X_measurement_rt_pcr_selected = X_measurement_rt_pcr[:, 0:best_n_features_selected_measurement_rt_pcr]
X_condition_selected = X_condition[:, 0:best_n_features_selected_condition]
X_observation_selected = X_observation[:, 0:best_n_features_selected_observation]
X_device_exposure_selected = X_device_exposure[:, 0:best_n_features_selected_device_exposure]
X_drug_exposure_selected = X_drug_exposure[:, 0:best_n_features_selected_drug_exposure]
X_procedure_selected = X_procedure[:, 0:best_n_features_selected_procedure]
X_visit_selected = X_visit[:, 0:best_n_features_selected_visit]

X_selected = np.concatenate((X_measurement_selected, X_age), axis=1)
X_selected = np.concatenate((X_selected, X_measurement_rt_pcr_selected), axis=1)
X_selected = np.concatenate((X_selected, X_condition_selected), axis=1)
X_selected = np.concatenate((X_selected, X_observation_selected), axis=1)
X_selected = np.concatenate((X_selected, X_device_exposure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_drug_exposure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_procedure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_visit_selected), axis=1)

selectorfile = '../../model/feature_selector_stage_2.sav'
selector = load(open(selectorfile, 'rb'))
X_selected = selector.transform(X_selected)

X_selected = transformer.transform(X_selected)

X_selected = np.concatenate((X_selected, X_gender_race_ethnicity), axis=1)

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

