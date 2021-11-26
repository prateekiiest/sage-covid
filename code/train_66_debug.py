import datetime as dt
import numpy as np
from datetime import date
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from joblib import dump
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

def feature_selection_measurement_auprc(X, Y):

        #because this is simple MIL model
        n_features = X.shape[1] // 3 

        X_remain, X_opt, Y_remain, Y_opt = train_test_split(X, Y, test_size=0.25, stratify=Y)

        #idx = CFS.cfs(X_opt, Y_opt)

        print("n features total: {}".format(n_features))

        feature_step_size = 5
        best_n_features_selected = 5
        if (n_features < feature_step_size):
                feature_step_size = 1
                best_n_features_selected = 1

        n_iterations = n_features // feature_step_size
        max_auprc = 0

        for i in range(n_iterations):

                #print(i+1)

                n_features_selected = (i+1)*feature_step_size
                X_min_opt_selected = X_opt[:, 0:n_features_selected]
                X_max_opt_selected = X_opt[:, n_features:(n_features+n_features_selected)]
                X_ave_opt_selected = X_opt[:, 2*n_features:(2*n_features+n_features_selected)]

                X_opt_selected = np.concatenate((X_min_opt_selected, X_max_opt_selected), axis=1)
                X_opt_selected = np.concatenate((X_opt_selected, X_ave_opt_selected), axis=1)

                clf = LogisticRegression(solver='saga')

                y_pred = cross_val_predict(clf, X_opt_selected, Y_opt, cv=2, method='predict_proba')
                precision, recall, thresholds = precision_recall_curve(Y_opt, y_pred[:,1])
                auprc = auc(recall, precision)

                #print("AUPRC={:.3f}".format(auprc))

                if (auprc > max_auprc):
                        max_auprc = auprc
                        best_n_features_selected = n_features_selected

        X_min_selected = X[:, 0:best_n_features_selected]
        X_max_selected = X[:, n_features:(n_features+best_n_features_selected)]
        X_ave_selected = X[:, 2*n_features:(2*n_features+best_n_features_selected)]

        X_selected = np.concatenate((X_min_selected, X_max_selected), axis=1)
        X_selected = np.concatenate((X_selected, X_ave_selected), axis=1)

        return X_selected, best_n_features_selected

def feature_selection_auprc(X, Y):

        n_features = X.shape[1]

        X_remain, X_opt, Y_remain, Y_opt = train_test_split(X, Y, test_size=0.25, stratify=Y)

        #idx = CFS.cfs(X_opt, Y_opt)

        print("n features total: {}".format(n_features))

        feature_step_size = 5
        best_n_features_selected = 5
        if (n_features < feature_step_size):
                feature_step_size = 1
                best_n_features_selected = 1

        n_iterations = n_features // feature_step_size
        max_auprc = 0

        for i in range(n_iterations):

                #print(i+1)

                n_features_selected = (i+1)*feature_step_size
                X_opt_selected = X_opt[:, 0:n_features_selected]

                #print(X_visit_opt_selected.shape)

                clf = LogisticRegression(solver='saga')

                y_pred = cross_val_predict(clf, X_opt_selected, Y_opt, cv=2, method='predict_proba')
                precision, recall, thresholds = precision_recall_curve(Y_opt, y_pred[:,1])
                auprc = auc(recall, precision)

                #print("AUPRC={:.3f}".format(auprc))

                if (auprc > max_auprc):
                        max_auprc = auprc
                        best_n_features_selected = n_features_selected

        X_selected = X[:, 0:best_n_features_selected]

        return X_selected, best_n_features_selected

print("Load person.csv")

print(datetime.now())
print("true labels")

measurement_concept_ids_filename = "../data/measurement_concept_ids.list"
condition_concept_ids_filename = "../data/condition_concept_ids.list"
observation_concept_ids_filename = "../data/observation_concept_ids.list"
device_exposure_concept_ids_filename = "../data/device_exposure_concept_ids.list"
drug_exposure_concept_ids_filename = "../data/drug_exposure_concept_ids.list"
procedure_concept_ids_filename = "../data/procedure_concept_ids.list"
visit_concept_ids_filename = "../data/visit_concept_ids.list"

gs = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/goldstandard.csv')

person_ids_positives = np.array(gs.loc[gs['status'] == 1.0, 'person_id'])
person_ids_negatives = np.array(gs.loc[gs['status'] == 0.0, 'person_id'])

print(datetime.now())

print("Load procedure_occurrence.csv")

procedure = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/procedure_occurrence.csv',usecols = ['procedure_concept_id', 'person_id', 'procedure_date'])

procedure = procedure.dropna(subset = ['procedure_concept_id'])
procedure = procedure.astype({"procedure_concept_id": int})
procedure = procedure.astype({"procedure_concept_id": str})
procedure['procedure_date'] = procedure['procedure_date'].fillna('1900-01-01')

print("Sort procedure concept ids")

procedure_positives_after_covid = procedure.query('procedure_date > "2019-11-17" and person_id in @person_ids_positives')
procedure_concept_ids_positives_after_covid = procedure_positives_after_covid['procedure_concept_id'].tolist()

print(procedure_concept_ids_positives_after_covid)

procedure_negatives_after_covid = procedure.query('procedure_date > "2019-11-17" and person_id in @person_ids_negatives and procedure_concept_id in @procedure_concept_ids_positives_after_covid')

print(procedure_negatives_after_covid)

procedure_positives_after_covid['counts_pos'] = procedure_positives_after_covid.groupby('procedure_concept_id')['procedure_concept_id'].transform('count')
procedure_negatives_after_covid['counts_neg'] = procedure_negatives_after_covid.groupby('procedure_concept_id')['procedure_concept_id'].transform('count')

print(procedure_positives_after_covid)
print(procedure_negatives_after_covid)

procedure_positives_after_covid = procedure_positives_after_covid.drop_duplicates('procedure_concept_id').sort_values(['procedure_concept_id'])
procedure_negatives_after_covid = procedure_negatives_after_covid.drop_duplicates('procedure_concept_id').sort_values(['procedure_concept_id'])
procedure_after_covid = pd.merge(procedure_positives_after_covid, procedure_negatives_after_covid, on= 'procedure_concept_id')
procedure_positives_after_covid_counts = procedure_after_covid['counts_pos'].to_numpy()
procedure_negatives_after_covid_counts = procedure_after_covid['counts_neg'].to_numpy()

print(procedure_positives_after_covid)
print(procedure_negatives_after_covid)
print(procedure_positives_after_covid_counts)
print(procedure_negatives_after_covid_counts)

procedure_concept_ids_after_covid = procedure_positives_after_covid['procedure_concept_id'].to_numpy()

print(procedure_concept_ids_after_covid)

ratios = np.true_divide(procedure_positives_after_covid_counts, (1+procedure_negatives_after_covid_counts))

print(ratios)

ratios_sorted, concept_ids_sorted = zip( *sorted( zip(ratios, procedure_concept_ids_after_covid) ) )
ratios_sorted = ratios_sorted[::-1]
concept_ids_sorted = concept_ids_sorted[::-1]

print(concept_ids_sorted)

procedure_concept_ids_sorted_wrt_ratios = concept_ids_sorted

sys.exit(0)

print("Load visit_occurrence.csv")

visit = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/visit_occurrence.csv',usecols = ['visit_concept_id', 'person_id', 'visit_start_date', 'visit_end_date'])

visit = visit.dropna(subset = ['visit_concept_id'])
visit = visit.astype({"visit_concept_id": int})
visit = visit.astype({"visit_concept_id": str})
visit['visit_start_date'] = visit['visit_start_date'].fillna('1900-01-01')
visit['visit_end_date'] = visit['visit_end_date'].fillna('2100-01-01')

print("Sort visit concept ids")

visit_positives_after_covid = visit.query('visit_end_date > "2019-11-17" and person_id in @person_ids_positives')
visit_concept_ids_positives_after_covid = visit_positives_after_covid['visit_concept_id'].tolist()

visit_negatives_after_covid = visit.query('visit_end_date > "2019-11-17" and person_id in @person_ids_negatives and visit_concept_id in @visit_concept_ids_positives_after_covid')

visit_positives_after_covid['counts'] = visit_positives_after_covid.groupby('visit_concept_id')['visit_concept_id'].transform('count')
visit_negatives_after_covid['counts'] = visit_negatives_after_covid.groupby('visit_concept_id')['visit_concept_id'].transform('count')

visit_positives_after_covid = visit_positives_after_covid.drop_duplicates('visit_concept_id').sort_values(['visit_concept_id'])
visit_negatives_after_covid = visit_negatives_after_covid.drop_duplicates('visit_concept_id').sort_values(['visit_concept_id'])
visit_positives_after_covid_counts = visit_positives_after_covid['counts'].to_numpy()
visit_negatives_after_covid_counts = visit_negatives_after_covid['counts'].to_numpy()

visit_concept_ids_after_covid = visit_positives_after_covid['visit_concept_id'].to_numpy()
ratios = np.true_divide(visit_positives_after_covid_counts, (1+visit_negatives_after_covid_counts))

ratios_sorted, concept_ids_sorted = zip( *sorted( zip(ratios, visit_concept_ids_after_covid) ) ) 
ratios_sorted = ratios_sorted[::-1]
concept_ids_sorted = concept_ids_sorted[::-1]

visit_concept_ids_sorted_wrt_ratios = concept_ids_sorted

measurement_feature_concept_ids = measurement_concept_ids_sorted_wrt_ratios[0:100]
condition_feature_concept_ids = condition_concept_ids_sorted_wrt_ratios[0:100]
observation_feature_concept_ids = observation_concept_ids_sorted_wrt_ratios[0:100]
device_exposure_feature_concept_ids = device_concept_ids_sorted_wrt_ratios[0:100]
drug_exposure_feature_concept_ids = drug_concept_ids_sorted_wrt_ratios[0:100]
procedure_feature_concept_ids = procedure_concept_ids_sorted_wrt_ratios[0:100]
visit_feature_concept_ids = visit_concept_ids_sorted_wrt_ratios[0:100]

print("Load person.csv")

person = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/person.csv')
today = date.today().year

'''generate the feature set'''
print(datetime.now())
print("Define feature arrays")

person = person.drop_duplicates(subset = ['person_id'])
person_index = dict(zip(person.person_id, range(len(person.person_id))))
measurement_feature_index = dict(zip(measurement_feature_concept_ids, range(len(measurement_feature_concept_ids))))
condition_feature_index = dict(zip(condition_feature_concept_ids, range(len(condition_feature_concept_ids))))
observation_feature_index = dict(zip(observation_feature_concept_ids, range(len(observation_feature_concept_ids))))
device_exposure_feature_index = dict(zip(device_exposure_feature_concept_ids, range(len(device_exposure_feature_concept_ids))))
drug_exposure_feature_index = dict(zip(drug_exposure_feature_concept_ids, range(len(drug_exposure_feature_concept_ids))))
procedure_feature_index = dict(zip(procedure_feature_concept_ids, range(len(procedure_feature_concept_ids))))
visit_feature_index = dict(zip(visit_feature_concept_ids, range(len(visit_feature_concept_ids))))

n_persons = person.shape[0]
n_measurement_types = len(measurement_feature_concept_ids)

X_min = np.zeros((len(person_index), n_measurement_types))
X_max = np.zeros((len(person_index), n_measurement_types))
X_ave = np.zeros((len(person_index), n_measurement_types))
X_min[:] = -10
X_max[:] = -10
X_ave[:] = -10

n_condition_types = len(condition_feature_concept_ids)
X_condition = np.zeros((len(person_index), n_condition_types))

n_observation_types = len(observation_feature_concept_ids)
X_observation = np.zeros((len(person_index), n_observation_types))

n_device_exposure_types = len(device_exposure_feature_concept_ids)
X_device_exposure = np.zeros((len(person_index), n_device_exposure_types))

n_drug_exposure_types = len(drug_exposure_feature_concept_ids)
X_drug_exposure = np.zeros((len(person_index), n_drug_exposure_types))

n_procedure_types = len(procedure_feature_concept_ids)
X_procedure = np.zeros((len(person_index), n_procedure_types))

n_visit_types = len(visit_feature_concept_ids)
X_visit = np.zeros((len(person_index), n_visit_types))

X_age = np.zeros((len(person_index), 1))
X_gender_race_ethnicity_raw = np.zeros((len(person_index), 3))

print(datetime.now())
person_ids_after_covid_set = set()

print("measurements")

le = preprocessing.LabelEncoder()

for i in measurement_feature_concept_ids:

        index_f = measurement_feature_index[i]

        #subm = measurement[measurement['measurement_concept_id'] == i]
        subm = measurement.query('measurement_concept_id in @i')

        if (i == '3003694'): #blood group and Rh group 

                subm = subm.query('value_as_number != -10')
                if (subm.empty):
                        continue

                subm = subm.drop_duplicates(subset = 'person_id')
                for person_id in subm['person_id']:
                        index_p = person_index[person_id]
                        X_min[index_p][index_f] = subm.loc[subm['person_id']==person_id, 'value_as_number'].iloc[0]
                        X_max[index_p][index_f] = X_min[index_p][index_f]
                        X_ave[index_p][index_f] = X_min[index_p][index_f]
                continue

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
if ('3003694' in measurement_feature_concept_ids):

        index_f = measurement_feature_index['3003694']
        le.fit(X_min[:, index_f])
        X_min[:, index_f] = le.transform(X_min[:, index_f])
        le.fit(X_max[:, index_f])
        X_max[:, index_f] = le.transform(X_max[:, index_f])
        le.fit(X_ave[:, index_f])
        X_ave[:, index_f] = le.transform(X_ave[:, index_f])

print(datetime.now())
print("conditions")

for i in condition_feature_concept_ids:

        index_f = condition_feature_index[i]
        subm = condition.query('condition_concept_id in @i')
        subm_after_covid = subm.query('condition_end_date > "2019-11-17"')
        #subm_before_covid = subm.query('condition_end_date <= 2019-11-17')

        #condition_person_ids = subm_after_covid.groupby('person_id')['person_id']
        
        #for person_id in subm_after_covid.drop_duplicates(subset = ['person_id'])['person_id']:
        for person_id in subm_after_covid['person_id']:

                index_p = person_index[person_id]
                X_condition[index_p][index_f] += 1.0

print(datetime.now())

print("observations")

for i in observation_feature_concept_ids:

        index_f = observation_feature_index[i]
        subm = observation.query('observation_concept_id in @i')
        #subm_after_covid = subm.query('observation_date > 2019-11-17')

        #for person_id in subm.drop_duplicates(subset = ['person_id'])['person_id']:
        for person_id in subm['person_id']:

                index_p = person_index[person_id]
                X_observation[index_p][index_f] += 1.0

print(datetime.now())

print("device exposures")

for i in device_exposure_feature_concept_ids:

        index_f = device_exposure_feature_index[i]
        subm = device_exposure.query('device_concept_id in @i')
        subm_after_covid = subm.query('device_exposure_end_date > "2019-11-17"')
        #subm_before_covid = subm.query('device_exposure_end_date <= 2019-11-17')

        #for person_id in subm_after_covid.drop_duplicates(subset = ['person_id'])['person_id']:
        for person_id in subm_after_covid['person_id']:

                index_p = person_index[person_id]
                X_device_exposure[index_p][index_f] += 1.0

print(datetime.now())

print("drug exposures")

for i in drug_exposure_feature_concept_ids:

        index_f = drug_exposure_feature_index[i]
        subm = drug_exposure.query('drug_concept_id in @i')
        subm_after_covid = subm.query('drug_exposure_end_date > "2019-11-17"')
        #subm_before_covid = subm.query('drug_exposure_end_date <= 2019-11-17')

        #for person_id in subm_after_covid.drop_duplicates(subset = ['person_id'])['person_id']:
        for person_id in subm_after_covid['person_id']:

                index_p = person_index[person_id]
                X_drug_exposure[index_p][index_f] += 1.0

print(datetime.now())
print("procedures")

for i in procedure_feature_concept_ids:

        index_f = procedure_feature_index[i]
        subm = procedure.query('procedure_concept_id in @i')
        subm_after_covid = subm.query('procedure_date > "2019-11-17"')

        #for person_id in subm_after_covid.drop_duplicates(subset = ['person_id'])['person_id']:
        for person_id in subm_after_covid['person_id']:

                index_p = person_index[person_id]
                X_procedure[index_p][index_f] += 1.0

print(datetime.now())
print("visits")

for i in visit_feature_concept_ids:

        index_f = visit_feature_index[i]
        subm = visit.query('visit_concept_id in @i') 
        subm_after_covid = subm.query('visit_end_date > "2019-11-17"')

        #for person_id in subm_after_covid.drop_duplicates(subset = ['person_id'])['person_id']:
        for person_id in subm_after_covid['person_id']:

                index_p = person_index[person_id]
                X_visit[index_p][index_f] += 1.0 

print(datetime.now())

print("ages")

person_ids = person['person_id'].to_numpy()
year_of_births = person['year_of_birth'].to_numpy()

for i in range(n_persons):

        #print(i)
        person_id = person_ids[i]
        year_of_birth = year_of_births[i]
        age = today - year_of_birth
        index_p = person_index[person_id]
        index_f = 0
        X_age[index_p, index_f] = age

print(datetime.now())

print("gender, race, ethnicity")

enc = OneHotEncoder(handle_unknown='ignore')

genders = person['gender_concept_id'].to_numpy()
races = person['race_concept_id'].to_numpy()
ethnicities = person['ethnicity_concept_id'].to_numpy()

for i in range(n_persons):

        X_gender_race_ethnicity_raw[i][0] = genders[i]
        X_gender_race_ethnicity_raw[i][1] = races[i]
        X_gender_race_ethnicity_raw[i][2] = ethnicities[i]

one_hot_encoder = enc.fit(X_gender_race_ethnicity_raw)
X_gender_race_ethnicity = enc.transform(X_gender_race_ethnicity_raw).toarray()

X = np.concatenate((X_min, X_max), axis=1)
X_measurement = np.concatenate((X, X_ave), axis=1)

print("true labels")

gs = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/goldstandard.csv')
person_status = person.merge(gs, how = 'left', on = ['person_id'])
person_status.drop_duplicates(subset=['person_id'], keep = 'first',inplace = True)
Y =  np.array(person_status[['status']]).ravel()
print("Y.shape")
print(Y.shape)
#clf = LogisticRegressionCV(cv = 10, penalty = 'l2', tol = 0.0001, fit_intercept = True, intercept_scaling = 1, class_weight = None, random_state = None,
#max_iter = 100, verbose = 0, n_jobs = None, scoring='roc_auc').fit(X,Y)
print(datetime.now())

dump(measurement_feature_concept_ids, measurement_concept_ids_filename)
dump(condition_feature_concept_ids, condition_concept_ids_filename)
dump(observation_feature_concept_ids, observation_concept_ids_filename)
dump(device_exposure_feature_concept_ids, device_exposure_concept_ids_filename)
dump(drug_exposure_feature_concept_ids, drug_exposure_concept_ids_filename)
dump(procedure_feature_concept_ids, procedure_concept_ids_filename)
dump(visit_feature_concept_ids, visit_concept_ids_filename)

print(datetime.now())
print("feature selection")

X_measurement_selected, best_n_features_selected_measurement = feature_selection_measurement_auprc(X_measurement, Y)
X_condition_selected, best_n_features_selected_condition = feature_selection_auprc(X_condition, Y)
X_observation_selected, best_n_features_selected_observation = feature_selection_auprc(X_observation, Y)
X_device_exposure_selected, best_n_features_selected_device_exposure = feature_selection_auprc(X_device_exposure, Y)
X_drug_exposure_selected, best_n_features_selected_drug_exposure = feature_selection_auprc(X_drug_exposure, Y)
X_procedure_selected, best_n_features_selected_procedure = feature_selection_auprc(X_procedure, Y)
X_visit_selected, best_n_features_selected_visit = feature_selection_auprc(X_visit, Y)

print("concatenation of feature matrices")

X_selected = np.concatenate((X_measurement_selected, X_age), axis=1)
X_selected = np.concatenate((X_selected, X_condition_selected), axis=1)
X_selected = np.concatenate((X_selected, X_observation_selected), axis=1)
X_selected = np.concatenate((X_selected, X_device_exposure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_drug_exposure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_procedure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_visit_selected), axis=1)
#X_selected = np.concatenate((X_selected, X_gender_race_ethnicity), axis=1)

selector = SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=50)).fit(X_selected, Y)
X_selected = selector.transform(X_selected)
X_selected = np.concatenate((X_selected, X_gender_race_ethnicity), axis=1)

selected_feature_indices = selector.get_support(indices=True)
scalerfile = '../model/feature_selector_stage_2.sav'
dump(selector, open(scalerfile, 'wb'))

print("X_selected.shape")
print(X_selected.shape)

print(datetime.now())
print("model training")

#clf = LogisticRegression(solver='saga').fit(X_selected, Y)
clf = MLPClassifier(learning_rate_init=0.01).fit(X_selected, Y)

print(datetime.now())

dump(clf, '../model/baseline.joblib')
dump(one_hot_encoder, '../model/one_hot_encoder')
dump(le, '../model/label_encoder')
dump(best_n_features_selected_measurement, '../model/best_n_features_selected_measurement')
dump(best_n_features_selected_condition, '../model/best_n_features_selected_condition')
dump(best_n_features_selected_observation, '../model/best_n_features_selected_observation')
dump(best_n_features_selected_device_exposure, '../model/best_n_features_selected_device_exposure')
dump(best_n_features_selected_drug_exposure, '../model/best_n_features_selected_drug_exposure')
dump(best_n_features_selected_procedure, '../model/best_n_features_selected_procedure')
dump(best_n_features_selected_visit, '../model/best_n_features_selected_visit')

print("Training stage finished")

