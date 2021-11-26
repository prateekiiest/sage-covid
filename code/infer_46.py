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
from scipy.stats import norm
import math

print(datetime.now())
print("true labels")

def compute_log_likelihoods_gaussian(X_i, means, variances):

	log_likelihoods = []

	n_classes = means.shape[0]
	n_features = means.shape[1]

	for i in range(n_classes):

		log_likelihood = 0.0

		for j in range(n_features):

			mean = means[i][j]
			variance = variances[i][j]
			if (variance < 0.0):
				continue

			normal_distribution_value = norm.pdf(X_i[j], mean, math.sqrt(variance))
			if (normal_distribution_value <= 1e-300):
				log_likelihood += -1e10
			else:
				log_likelihood += math.log(normal_distribution_value)
		log_likelihoods += [log_likelihood]	

	return log_likelihoods

def compute_log_likelihoods_multinomial(X_i, log_probabilities_discrete):

	log_likelihoods = []

	n_classes = log_probabilities_discrete.shape[0]
	n_features = log_probabilities_discrete.shape[1]

	for i in range(n_classes):

		#log_likelihood = math.log(math.factorial(np.sum(X_i)))
		log_likelihood = 0.0

		for j in range(n_features):

			#log_likelihood += (X_i[j]*log_probabilities_discrete[i][j] - math.log(math.factorial(X_i[j])))
			log_likelihood += X_i[j]*log_probabilities_discrete[i][j]

		log_likelihoods += [log_likelihood]

	return log_likelihoods

measurement_concept_ids_filename = "../data/measurement_concept_ids.list"
condition_concept_ids_filename = "../data/condition_concept_ids.list"
observation_concept_ids_filename = "../data/observation_concept_ids.list"
device_exposure_concept_ids_filename = "../data/device_exposure_concept_ids.list"
drug_exposure_concept_ids_filename = "../data/drug_exposure_concept_ids.list"
procedure_concept_ids_filename = "../data/procedure_concept_ids.list"
visit_concept_ids_filename = "../data/visit_concept_ids.list"
best_n_features_selected_measurement_filename = "../model/best_n_features_selected_measurement"
best_n_features_selected_condition_filename = "../model/best_n_features_selected_condition"
best_n_features_selected_observation_filename = "../model/best_n_features_selected_observation"
best_n_features_selected_device_exposure_filename = "../model/best_n_features_selected_device_exposure"
best_n_features_selected_drug_exposure_filename = "../model/best_n_features_selected_drug_exposure"
best_n_features_selected_procedure_filename = "../model/best_n_features_selected_procedure"
best_n_features_selected_visit_filename = "../model/best_n_features_selected_visit"

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

print("Load condition.csv")
condition = pd.read_csv("../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/evaluation/condition_occurrence.csv",usecols = ['condition_concept_id','person_id', 'condition_start_date', 'condition_end_date'])

condition = condition.dropna(subset = ['condition_concept_id'])
condition = condition.astype({"condition_concept_id": int})
condition = condition.astype({"condition_concept_id": str})
condition['condition_end_date'] = condition['condition_end_date'].fillna('2100-01-01')
condition['condition_start_date'] = condition['condition_start_date'].fillna('1900-01-01')

print(datetime.now())

print("Load observation.csv")
observation = pd.read_csv("../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/evaluation/observation.csv",usecols = ['observation_concept_id','person_id', 'observation_date'])

observation = observation.dropna(subset = ['observation_concept_id'])
observation = observation.astype({"observation_concept_id": int})
observation = observation.astype({"observation_concept_id": str})

print(datetime.now())

print("Load device_exposure.csv")
device_exposure = pd.read_csv("../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/evaluation/device_exposure.csv",usecols = ['device_concept_id','person_id', 'device_exposure_start_date', 'device_exposure_end_date'])

device_exposure = device_exposure.dropna(subset = ['device_concept_id'])
device_exposure = device_exposure.astype({"device_concept_id": int})
device_exposure = device_exposure.astype({"device_concept_id": str})
device_exposure['device_exposure_end_date'] = device_exposure['device_exposure_end_date'].fillna('2100-01-01')
device_exposure['device_exposure_start_date'] = device_exposure['device_exposure_start_date'].fillna('1900-01-01')

print(datetime.now())

print("Load drug_exposure.csv")
drug_exposure = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/evaluation/drug_exposure.csv',usecols = ['drug_concept_id', 'person_id', 'drug_exposure_start_date', 'drug_exposure_end_date'])

drug_exposure = drug_exposure.dropna(subset = ['drug_concept_id'])
drug_exposure = drug_exposure.astype({"drug_concept_id": int})
drug_exposure = drug_exposure.astype({"drug_concept_id": str})
drug_exposure['drug_exposure_end_date'] = drug_exposure['drug_exposure_end_date'].fillna('2100-01-01')
drug_exposure['drug_exposure_start_date'] = drug_exposure['drug_exposure_start_date'].fillna('1900-01-01')

print(datetime.now())

print("Load procedure_occurrence.csv")

procedure = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/evaluation/procedure_occurrence.csv',usecols = ['procedure_concept_id', 'person_id', 'procedure_date'])

procedure = procedure.dropna(subset = ['procedure_concept_id'])
procedure = procedure.astype({"procedure_concept_id": int})
procedure = procedure.astype({"procedure_concept_id": str})
procedure['procedure_date'] = procedure['procedure_date'].fillna('1900-01-01')

print(datetime.now())

print("Load visit_occurrence.csv")

visit = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/evaluation/visit_occurrence.csv',usecols = ['visit_concept_id', 'person_id', 'visit_start_date', 'visit_end_date'])

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
best_n_features_selected_condition = load(open(best_n_features_selected_condition_filename, "rb"))
best_n_features_selected_observation = load(open(best_n_features_selected_observation_filename, "rb"))
best_n_features_selected_device_exposure = load(open(best_n_features_selected_device_exposure_filename, "rb"))
best_n_features_selected_drug_exposure = load(open(best_n_features_selected_drug_exposure_filename, "rb"))
best_n_features_selected_procedure = load(open(best_n_features_selected_procedure_filename, "rb"))
best_n_features_selected_visit = load(open(best_n_features_selected_visit_filename, "rb"))

print("Load person.csv")

person = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/evaluation/person.csv')
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
        subm_after_covid = subm.query('measurement_date > "2020-01-01"')
        subm_before_covid = subm.query('measurement_date <= "2020-01-01"')

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

print("concatenations")

X_min_selected = X_min[:, 0:best_n_features_selected_measurement]
X_max_selected = X_max[:, 0:best_n_features_selected_measurement]
X_ave_selected = X_ave[:, 0:best_n_features_selected_measurement]

X_measurement_selected = np.concatenate((X_min_selected, X_max_selected), axis=1)
X_measurement_selected = np.concatenate((X_measurement_selected, X_ave_selected), axis=1)
X_condition_selected = X_condition[:, 0:best_n_features_selected_condition]
X_observation_selected = X_observation[:, 0:best_n_features_selected_observation]
X_device_exposure_selected = X_device_exposure[:, 0:best_n_features_selected_device_exposure]
X_drug_exposure_selected = X_drug_exposure[:, 0:best_n_features_selected_drug_exposure]
X_procedure_selected = X_procedure[:, 0:best_n_features_selected_procedure]
X_visit_selected = X_visit[:, 0:best_n_features_selected_visit]

X_continuous_selected = np.concatenate((X_measurement_selected, X_age), axis=1)
X_selected = np.concatenate((X_continuous_selected, X_condition_selected), axis=1)
X_selected = np.concatenate((X_selected, X_observation_selected), axis=1)
X_selected = np.concatenate((X_selected, X_device_exposure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_drug_exposure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_procedure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_visit_selected), axis=1)

n_continuous_features_selected_1st_stage = X_continuous_selected.shape[1]

selector_file = '../model/feature_selector_stage_2.sav'
selector = load(open(selector_file, 'rb'))
X_selected = selector.transform(X_selected)
selected_feature_indices = selector.get_support(indices=True)

print("X_selected.shape")
print(X_selected.shape)

print(datetime.now())

is_continuous = selected_feature_indices < n_continuous_features_selected_1st_stage
X_continuous_selected_2 = X_selected[:, is_continuous]
X_discrete_selected_2 = X_selected[:, np.invert(is_continuous)]
clf_continuous = load('../model/baseline_continuous.joblib')
clf_discrete = load('../model/baseline_discrete.joblib')

#parameters of for the Gaussian distributions for the continuous valued features
classes_continuous = clf_continuous.classes_
means = clf_continuous.theta_
variances = clf_continuous.sigma_
class_priors = clf_continuous.class_prior_

#parameters of the multinomial distribution for the discrete valued features
classes_discrete = clf_discrete.classes_
log_probabilities_discrete = clf_discrete.feature_log_prob_

#print(classes_continuous)
#print(classes_discrete)
#print(class_priors)

person_ids_output_probs = pd.DataFrame(columns=['person_id', 'score'])
person_ids = person[['person_id']].to_numpy()

for i in range(len(person_ids)):

	person_id = person_ids[i][0]
	X_continuous_selected_2_i = X_continuous_selected_2[i, :]
	X_discrete_selected_2_i = X_discrete_selected_2[i, :]
	log_likelihoods_gaussian = compute_log_likelihoods_gaussian(X_continuous_selected_2_i, means, variances)
	log_likelihoods_multinomial = compute_log_likelihoods_multinomial(X_discrete_selected_2_i, log_probabilities_discrete)

	#print(log_likelihoods_gaussian)
	#print(log_likelihoods_multinomial)

	likelihood_zero = log_likelihoods_gaussian[0] + log_likelihoods_multinomial[0] + math.log(class_priors[0])
	likelihood_one = log_likelihoods_gaussian[1] + log_likelihoods_multinomial[1] + math.log(class_priors[1])
	
	numerator = math.exp(likelihood_one)
	numerator_complement = math.exp(likelihood_zero)

	denominator = numerator + numerator_complement

	if (not math.isnan(numerator)) and (not math.isnan(numerator_complement)) and (denominator != 0.0):
		probability_one = numerator / denominator
	else:
		probability_one = 0.0

	row = pd.DataFrame([[person_id, probability_one]],columns=['person_id', 'score'])
	person_ids_output_probs = person_ids_output_probs.append(row)

person_ids_output_probs.to_csv('../output/predictions.csv', index = False)
print("Inferring stage finished")
