import datetime as dt
import numpy as np
from datetime import date
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
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
#from skfeature.function.statistical_based import CFS

def feature_selection_measurement_auprc(X, Y):

	#because this is simple MIL model
        n_features = X.shape[1] / 4

        X_remain, X_opt, Y_remain, Y_opt = train_test_split(X, Y, test_size=0.25, stratify=Y)

        #idx = CFS.cfs(X_opt, Y_opt)

        print("n features total: {}".format(n_features))

        feature_step_size = 5
        best_n_features_selected = 5
        if (n_features < feature_step_size):
                feature_step_size = 1
                best_n_features_selected = 1

        n_iterations = n_features / feature_step_size
        max_auprc = 0

        for i in range(n_iterations):

                #print(i+1)

                n_features_selected = (i+1)*feature_step_size
		X_min_opt_selected = X_opt[:, 0:n_features_selected]
		X_max_opt_selected = X_opt[:, n_features:(n_features+n_features_selected)]
		X_ave_opt_selected = X_opt[:, 2*n_features:(2*n_features+n_features_selected)]
		X_min_period_opt_selected = X_opt[:, 3*n_features:(3*n_features+n_features_selected)]

	        X_opt_selected = np.concatenate((X_min_opt_selected, X_max_opt_selected), axis=1)
        	X_opt_selected = np.concatenate((X_opt_selected, X_ave_opt_selected), axis=1)
                X_opt_selected = np.concatenate((X_opt_selected, X_min_period_opt_selected), axis=1)
		
                clf = LogisticRegression(solver='saga')
		#clf = GaussianNB()

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
	X_min_period_selected = X[:, 3*n_features:(3*n_features+best_n_features_selected)]

	X_selected = np.concatenate((X_min_selected, X_max_selected), axis=1)
	X_selected = np.concatenate((X_selected, X_ave_selected), axis=1)
        X_selected = np.concatenate((X_selected, X_min_period_selected), axis=1)

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

	n_iterations = n_features / feature_step_size
	max_auprc = 0

	for i in range(n_iterations):

		#print(i+1)

		n_features_selected = (i+1)*feature_step_size
		X_opt_selected = X_opt[:, 0:n_features_selected]

		#print(X_visit_opt_selected.shape)

		clf = LogisticRegression(solver='saga')
		#clf = MultinomialNB()

		y_pred = cross_val_predict(clf, X_opt_selected, Y_opt, cv=2, method='predict_proba')
		precision, recall, thresholds = precision_recall_curve(Y_opt, y_pred[:,1])
		auprc = auc(recall, precision)

		#print("AUPRC={:.3f}".format(auprc))

		if (auprc > max_auprc):
			max_auprc = auprc
			best_n_features_selected = n_features_selected

	X_selected = X[:, 0:best_n_features_selected]

	return X_selected, best_n_features_selected

print(datetime.now())
print("true labels")

measurement_concept_ids_filename = "../model/measurement_concept_ids.list"
condition_concept_ids_filename = "../model/condition_concept_ids.list"
observation_concept_ids_filename = "../model/observation_concept_ids.list"
device_exposure_concept_ids_filename = "../model/device_exposure_concept_ids.list"
drug_exposure_concept_ids_filename = "../model/drug_exposure_concept_ids.list"
procedure_concept_ids_filename = "../model/procedure_concept_ids.list"
visit_concept_ids_filename = "../model/visit_concept_ids.list"

gs = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/goldstandard.csv')

person_ids_positives = np.array(gs.loc[gs['status'] == 1.0, 'person_id'])

print(datetime.now())
print("Load measurement.csv")

measurement = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/measurement.csv',usecols = ['measurement_concept_id','value_as_number','person_id', 'measurement_date', 'measurement_time'])
measurement = measurement.dropna(subset = ['measurement_concept_id'])
measurement = measurement.astype({"measurement_concept_id": int})
measurement = measurement.astype({"measurement_concept_id": str})
measurement['value_as_number'] = measurement['value_as_number'].fillna(-10)
measurement['measurement_date'] = measurement['measurement_date'].fillna('1900-01-01')
measurement['measurement_time'] = measurement['measurement_time'].fillna('1900-01-01')
print(measurement.shape)

print(datetime.now())
print("Sort measurement concept ids")

measurement_positives_after_covid = measurement.query('measurement_date > "2020-01-01" and person_id in @person_ids_positives')

measurement_positives_after_covid['counts'] = measurement_positives_after_covid.groupby('measurement_concept_id')['measurement_concept_id'].transform('count')
measurement_positives_after_covid.sort_values('counts', inplace=True, ascending=False)
#measurement_positives_after_covid = measurement_positives_after_covid.query('counts > 15')
measurement_positives_after_covid = measurement_positives_after_covid.drop_duplicates('measurement_concept_id')
measurement_concept_ids_sorted_wrt_frequency = measurement_positives_after_covid['measurement_concept_id'].tolist()
#print(measurement_concept_ids_sorted_wrt_frequency)

print(datetime.now())

print("Load condition.csv")
condition = pd.read_csv("../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/condition_occurrence.csv",usecols = ['condition_concept_id','person_id', 'condition_start_date', 'condition_end_date'])

condition = condition.dropna(subset = ['condition_concept_id'])
condition = condition.astype({"condition_concept_id": int})
condition = condition.astype({"condition_concept_id": str})
condition['condition_end_date'] = condition['condition_end_date'].fillna('2100-01-01')
condition['condition_start_date'] = condition['condition_start_date'].fillna('1900-01-01')

print(datetime.now())

print("Sort condition concept ids")

condition_positives_after_covid = condition.query('condition_end_date > "2020-01-01" and person_id in @person_ids_positives')

condition_positives_after_covid['counts'] = condition_positives_after_covid.groupby('condition_concept_id')['condition_concept_id'].transform('count')
#print(measurement_positives_after_covid)
condition_positives_after_covid.sort_values('counts', inplace=True, ascending=False)
#condition_positives_after_covid = condition_positives_after_covid.query('counts > 15')
condition_concept_ids_sorted_wrt_frequency = condition_positives_after_covid.drop_duplicates('condition_concept_id')['condition_concept_id'].tolist()

print(datetime.now())

print("Load observation.csv")
observation = pd.read_csv("../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/observation.csv",usecols = ['observation_concept_id','person_id', 'observation_date'])

observation = observation.dropna(subset = ['observation_concept_id'])
observation = observation.astype({"observation_concept_id": int})
observation = observation.astype({"observation_concept_id": str})

print(datetime.now())

print("Sort observation concept ids")

observation_positives_after_covid = observation.query('person_id in @person_ids_positives')

observation_positives_after_covid['counts'] = observation_positives_after_covid.groupby('observation_concept_id')['observation_concept_id'].transform('count')
#print(measurement_positives_after_covid)
observation_positives_after_covid.sort_values('counts', inplace=True, ascending=False)
#observation_positives_after_covid = observation_positives_after_covid.query('counts > 15')
observation_concept_ids_sorted_wrt_frequency = observation_positives_after_covid.drop_duplicates('observation_concept_id')['observation_concept_id'].tolist()

print(datetime.now())

print("Load device_exposure.csv")
device_exposure = pd.read_csv("../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/device_exposure.csv",usecols = ['device_concept_id','person_id', 'device_exposure_start_date', 'device_exposure_end_date'])

device_exposure = device_exposure.dropna(subset = ['device_concept_id'])
device_exposure = device_exposure.astype({"device_concept_id": int})
device_exposure = device_exposure.astype({"device_concept_id": str})
device_exposure['device_exposure_end_date'] = device_exposure['device_exposure_end_date'].fillna('2100-01-01')
device_exposure['device_exposure_start_date'] = device_exposure['device_exposure_start_date'].fillna('1900-01-01')

print(datetime.now())

print("Sort device_exposure concept ids")

device_exposure_positives_after_covid = device_exposure.query('device_exposure_end_date > "2020-01-01" and person_id in @person_ids_positives')

device_exposure_positives_after_covid['counts'] = device_exposure_positives_after_covid.groupby('device_concept_id')['device_concept_id'].transform('count')
#print(measurement_positives_after_covid)
device_exposure_positives_after_covid.sort_values('counts', inplace=True, ascending=False)
#device_exposure_positives_after_covid = device_exposure_positives_after_covid.query('counts > 15')
device_concept_ids_sorted_wrt_frequency = device_exposure_positives_after_covid.drop_duplicates('device_concept_id')['device_concept_id'].tolist()

print(datetime.now())

print("Load drug_exposure.csv")
drug_exposure = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/drug_exposure.csv',usecols = ['drug_concept_id', 'person_id', 'drug_exposure_start_date', 'drug_exposure_end_date'])

drug_exposure = drug_exposure.dropna(subset = ['drug_concept_id'])
drug_exposure = drug_exposure.astype({"drug_concept_id": int})
drug_exposure = drug_exposure.astype({"drug_concept_id": str})
drug_exposure['drug_exposure_end_date'] = drug_exposure['drug_exposure_end_date'].fillna('2100-01-01')
drug_exposure['drug_exposure_start_date'] = drug_exposure['drug_exposure_start_date'].fillna('1900-01-01')

print(datetime.now())

print("Sort drug_exposure concept ids")

drug_exposure_positives_after_covid = drug_exposure.query('drug_exposure_end_date > "2020-01-01" and person_id in @person_ids_positives')

drug_exposure_positives_after_covid['counts'] = drug_exposure_positives_after_covid.groupby('drug_concept_id')['drug_concept_id'].transform('count')
#print(measurement_positives_after_covid)
drug_exposure_positives_after_covid.sort_values('counts', inplace=True, ascending=False)
#drug_exposure_positives_after_covid = drug_exposure_positives_after_covid.query('counts > 15')
drug_concept_ids_sorted_wrt_frequency = drug_exposure_positives_after_covid.drop_duplicates('drug_concept_id')['drug_concept_id'].tolist()

print(datetime.now())

print("Load procedure_occurrence.csv")

procedure = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/procedure_occurrence.csv',usecols = ['procedure_concept_id', 'person_id', 'procedure_date'])

procedure = procedure.dropna(subset = ['procedure_concept_id'])
procedure = procedure.astype({"procedure_concept_id": int})
procedure = procedure.astype({"procedure_concept_id": str})
procedure['procedure_date'] = procedure['procedure_date'].fillna('1900-01-01')

print("Sort procedure concept ids")

procedure_positives_after_covid = procedure.query('procedure_date > "2020-01-01" and person_id in @person_ids_positives')

procedure_positives_after_covid['counts'] = procedure_positives_after_covid.groupby('procedure_concept_id')['procedure_concept_id'].transform('count')
#print(measurement_positives_after_covid)
procedure_positives_after_covid.sort_values('counts', inplace=True, ascending=False)
#procedure_positives_after_covid = procedure_positives_after_covid.query('counts > 15')
procedure_concept_ids_sorted_wrt_frequency = procedure_positives_after_covid.drop_duplicates('procedure_concept_id')['procedure_concept_id'].tolist()

print("Load visit_occurrence.csv")

visit = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/visit_occurrence.csv',usecols = ['visit_concept_id', 'person_id', 'visit_start_date', 'visit_end_date'])

visit = visit.dropna(subset = ['visit_concept_id'])
visit = visit.astype({"visit_concept_id": int})
visit = visit.astype({"visit_concept_id": str})
visit['visit_start_date'] = visit['visit_start_date'].fillna('1900-01-01')
visit['visit_end_date'] = visit['visit_end_date'].fillna('2100-01-01')

print("Sort visit concept ids")

visit_positives_after_covid = visit.query('visit_end_date > "2020-01-01" and person_id in @person_ids_positives')

visit_positives_after_covid['counts'] = visit_positives_after_covid.groupby('visit_concept_id')['visit_concept_id'].transform('count')
#print(measurement_positives_after_covid)
visit_positives_after_covid.sort_values('counts', inplace=True, ascending=False)
#visit_positives_after_covid = visit_positives_after_covid.query('counts > 15')
visit_concept_ids_sorted_wrt_frequency = visit_positives_after_covid.drop_duplicates('visit_concept_id')['visit_concept_id'].tolist()

measurement_feature_concept_ids = measurement_concept_ids_sorted_wrt_frequency[0:100]
condition_feature_concept_ids = condition_concept_ids_sorted_wrt_frequency[0:100]
observation_feature_concept_ids = observation_concept_ids_sorted_wrt_frequency[0:100]
device_exposure_feature_concept_ids = device_concept_ids_sorted_wrt_frequency[0:100]
drug_exposure_feature_concept_ids = drug_concept_ids_sorted_wrt_frequency[0:100]
procedure_feature_concept_ids = procedure_concept_ids_sorted_wrt_frequency[0:100]
visit_feature_concept_ids = visit_concept_ids_sorted_wrt_frequency[0:100]

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
X_min_period = np.zeros((len(person_index), n_measurement_types))
X_min[:] = -10
X_max[:] = -10
X_ave[:] = -10
X_min_period[:] = 180

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

print(datetime.now())
person_ids_after_covid_set = set()

print("measurements")

le = preprocessing.LabelEncoder()
date_format = "%Y-%m-%d"

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
        subm_after_covid = subm.query('measurement_date > "2020-01-01"')
        subm_before_covid = subm.query('measurement_date <= "2020-01-01"')

        subm_after_covid_min = subm_after_covid.groupby('person_id')['value_as_number'].min()
        subm_after_covid_max = subm_after_covid.groupby('person_id')['value_as_number'].max()
        subm_after_covid_ave = subm_after_covid.groupby('person_id')['value_as_number'].mean()
        
        subm_before_covid_min = subm_before_covid.groupby('person_id')['value_as_number'].min()
        subm_before_covid_max = subm_before_covid.groupby('person_id')['value_as_number'].max()
        subm_before_covid_ave = subm_before_covid.groupby('person_id')['value_as_number'].mean()

        for person_id in subm_after_covid_min.keys():

		subm_after_covid_person_measurement_dates = subm_after_covid.query('person_id == @person_id')[['measurement_date']]
		subm_after_covid_person_measurement_dates.sort_values(['measurement_date'], inplace=True, ascending=True)
		subm_after_covid_person_measurement_dates = subm_after_covid_person_measurement_dates.to_numpy()
		n_dates = len(subm_after_covid_person_measurement_dates)

		min_date_difference = 180 

		for j in range(n_dates-1):

			date_1 = subm_after_covid_person_measurement_dates[j][0]
			date_2 = subm_after_covid_person_measurement_dates[j+1][0]
			date_1_formatted = datetime.strptime(date_1, date_format)
			date_2_formatted = datetime.strptime(date_2, date_format)
			date_difference = date_2_formatted - date_1_formatted
			date_difference_in_days = date_difference.days

			if (j == 0):
				min_date_difference = date_difference_in_days

			elif (date_difference_in_days < min_date_difference):
				min_date_difference = date_difference_in_days

                index_p = person_index[person_id]

		X_min_period[index_p][index_f] = min_date_difference
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
        subm_after_covid = subm.query('condition_end_date > "2020-01-01"')
        #subm_before_covid = subm.query('condition_end_date <= 2020-01-01')

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
        #subm_after_covid = subm.query('observation_date > 2020-01-01')

        #for person_id in subm.drop_duplicates(subset = ['person_id'])['person_id']:
        for person_id in subm['person_id']:

                index_p = person_index[person_id]
                X_observation[index_p][index_f] += 1.0

print(datetime.now())

print("device exposures")

for i in device_exposure_feature_concept_ids:

        index_f = device_exposure_feature_index[i]
        subm = device_exposure.query('device_concept_id in @i')
        subm_after_covid = subm.query('device_exposure_end_date > "2020-01-01"')
        #subm_before_covid = subm.query('device_exposure_end_date <= 2020-01-01')

        #for person_id in subm_after_covid.drop_duplicates(subset = ['person_id'])['person_id']:
        for person_id in subm_after_covid['person_id']:

                index_p = person_index[person_id]
                X_device_exposure[index_p][index_f] += 1.0

print(datetime.now())

print("drug exposures")

for i in drug_exposure_feature_concept_ids:

        index_f = drug_exposure_feature_index[i]
        subm = drug_exposure.query('drug_concept_id in @i')
        subm_after_covid = subm.query('drug_exposure_end_date > "2020-01-01"')
        #subm_before_covid = subm.query('drug_exposure_end_date <= 2020-01-01')

        #for person_id in subm_after_covid.drop_duplicates(subset = ['person_id'])['person_id']:
        for person_id in subm_after_covid['person_id']:

                index_p = person_index[person_id]
                X_drug_exposure[index_p][index_f] += 1.0

print(datetime.now())
print("procedures")

for i in procedure_feature_concept_ids:

        index_f = procedure_feature_index[i]
        subm = procedure.query('procedure_concept_id in @i')
        subm_after_covid = subm.query('procedure_date > "2020-01-01"')

        #for person_id in subm_after_covid.drop_duplicates(subset = ['person_id'])['person_id']:
        for person_id in subm_after_covid['person_id']:

                index_p = person_index[person_id]
                X_procedure[index_p][index_f] += 1.0

print(datetime.now())
print("visits")

for i in visit_feature_concept_ids:

        index_f = visit_feature_index[i]
        subm = visit.query('visit_concept_id in @i') 
        subm_after_covid = subm.query('visit_end_date > "2020-01-01"')

        #for person_id in subm_after_covid.drop_duplicates(subset = ['person_id'])['person_id']:
        for person_id in subm_after_covid['person_id']:

                index_p = person_index[person_id]
                X_visit[index_p][index_f] += 1.0 

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

X = np.concatenate((X_min, X_max), axis=1)
X = np.concatenate((X, X_ave), axis=1)
X_measurement = np.concatenate((X, X_min_period), axis=1)

print("true labels")

gs = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/goldstandard.csv')
person_status = person.merge(gs, how = 'left', on = ['person_id'])
person_status.drop_duplicates(subset=['person_id'], keep = 'first',inplace = True)
Y =  np.array(person_status[['status']]).ravel()
print("Y.shape")
print(Y.shape)
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

X_continuous_selected = np.concatenate((X_measurement_selected, X_age), axis=1)
X_selected = np.concatenate((X_continuous_selected, X_condition_selected), axis=1)
X_selected = np.concatenate((X_selected, X_observation_selected), axis=1)
X_selected = np.concatenate((X_selected, X_device_exposure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_drug_exposure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_procedure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_visit_selected), axis=1)

selector = SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=50)).fit(X_selected, Y)
X_selected = selector.transform(X_selected)
selected_feature_indices = selector.get_support(indices=True)
selector_file = '../model/feature_selector_stage_2.sav'
dump(selector, open(selector_file, 'wb'))

print("X_selected.shape")
print(X_selected.shape)

print(datetime.now())
print("model training")

indices_zero = (Y == 0.0)
X_selected_zero = X_selected[indices_zero, :]
X_selected_one = X_selected[np.invert(indices_zero), :]

mean_zero = np.mean(X_selected_zero, axis = 0)
cov_zero = np.cov(X_selected_zero, rowvar = False)

mean_one = np.mean(X_selected_one, axis = 0)
cov_one = np.cov(X_selected_one, rowvar = False)

class_priors = np.zeros(2)
count_zero = sum(Y==0.0)
count_one = sum(Y==1.0)

class_priors[0] = count_zero / float(count_zero+count_one)
class_priors[1] = count_one / float(count_zero+count_one)

print(datetime.now())

dump(mean_zero, '../model/mean_zero')
dump(cov_zero, '../model/cov_zero')
dump(mean_one, '../model/mean_one')
dump(cov_one, '../model/cov_one')
dump(class_priors, '../model/class_priors')

dump(best_n_features_selected_measurement, '../model/best_n_features_selected_measurement')
dump(best_n_features_selected_condition, '../model/best_n_features_selected_condition')
dump(best_n_features_selected_observation, '../model/best_n_features_selected_observation')
dump(best_n_features_selected_device_exposure, '../model/best_n_features_selected_device_exposure')
dump(best_n_features_selected_drug_exposure, '../model/best_n_features_selected_drug_exposure')
dump(best_n_features_selected_procedure, '../model/best_n_features_selected_procedure')
dump(best_n_features_selected_visit, '../model/best_n_features_selected_visit')

print("Training stage finished")

