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
person_index = dict(zip(person.person_id, range(len(person.person_id))))
measurement_feature_indices = dict(zip(measurement_feature_concept_ids, range(len(measurement_feature_concept_ids))))

n_persons = person.shape[0]

print(n_persons)

print(datetime.now())

print("measurements")

le = preprocessing.LabelEncoder()

n_measurement_features = len(measurement_feature_concept_ids)

X_measurement = np.array([]).reshape(0, n_measurement_features+1)

person_ids = []

measurement_concept_ids_for_rules = {'3020891':37.5,'3027018':100,'3012888':80,'3004249':120,
'3023314':52,'3013650':8,'3004327':1.0,'3016502':95, '3010156':0.8, '3023091':5.9, '3024929':100, '3022250':210, '42870366':0.5, '3021337':0.4}

measurement_after_covid = measurement.query('measurement_date > "2020-01-01"')
measurement_before_covid = measurement.query('measurement_date <= "2020-01-01"')
#person_counter = 0
person_ids_after_covid = measurement_after_covid[['person_id']].drop_duplicates(['person_id']).values[:,0]

for person_id in person[['person_id']].values[:,0]:

	#print(person_counter+1)

	if (person_id in person_ids_after_covid):
		measurement = measurement_after_covid
	else:
		#measurement = measurement_before_covid
		feature_vector = np.zeros((1, n_measurement_features+1))
                X_measurement = np.vstack([X_measurement, feature_vector])
                person_ids += [person_id]
		continue

        sub_matrix = measurement.query('person_id == @person_id')

        for measurement_date in sub_matrix[['measurement_date']].drop_duplicates(['measurement_date']).values[:,0]:

                feature_vector = np.zeros((1, n_measurement_features+1))
                sub_matrix_2 = sub_matrix.query('measurement_date == @measurement_date')
                measurement_concept_ids = sub_matrix_2[['measurement_concept_id']].values[:,0]
                measurement_values = sub_matrix_2[['value_as_number']].values[:,0]
                measurement_values_available = []
                measurement_feature_indices_per_person_per_date = []

                for i in range(len(measurement_concept_ids)):
                        key = measurement_concept_ids[i]
                        if (key in measurement_feature_indices):
                                if (not math.isnan(measurement_values[i])):
                                        measurement_values_available += [measurement_values[i]]
                                        measurement_feature_indices_per_person_per_date += [measurement_feature_indices[key]]
                measurement_feature_indices_per_person_per_date = np.array(measurement_feature_indices_per_person_per_date)
                measurement_values_available = np.array(measurement_values_available)

                if (measurement_feature_indices_per_person_per_date.size > 0):
                        feature_vector[0][measurement_feature_indices_per_person_per_date] = measurement_values_available

                        biomarker_count = 0
                        for key,value in measurement_concept_ids_for_rules.items():

                                if (key in measurement_feature_indices):
                                        feature_index = measurement_feature_indices[key]
                                else:
                                        continue

                                if (key == '3016502') or (key == '3024929') or (key == '3004327'):
                                        if (feature_vector[0][feature_index] < value):
                                                biomarker_count += 1
                                else:
                                        if (feature_vector[0][feature_index] > value):
                                                biomarker_count += 1

                        feature_vector[0][n_measurement_features] = biomarker_count
                        X_measurement = np.vstack([X_measurement, feature_vector])
			person_ids += [person_id]

	#person_counter += 1

if ('3003694' in measurement_feature_indices):

	index_f = measurement_feature_indices['3003694']
	le.fit(X_measurement[:, index_f])
	X_measurement[:, index_f] = le.transform(X_measurement[:, index_f])

X_measurement_selected = X_measurement[:, 0:best_n_features_selected_measurement]

X_selected = X_measurement_selected

print("X_selected.shape")
print(X_selected.shape)

print(datetime.now())

person_ids_output_probs = pd.DataFrame(columns=['person_id', 'score'])

person_id_pre = -1

clf =  load('../model/baseline.joblib')
y_pred = clf.predict_proba(X_selected)[:,1]
max_prob = 0.0

for i in range(X_selected.shape[0]):

	person_id = person_ids[i]
	
	if (person_id != person_id_pre):

		if (i > 0):
			row = pd.DataFrame([[person_id_pre, max_prob]],columns=['person_id', 'score'])
			person_ids_output_probs = person_ids_output_probs.append(row)

		person_id_pre = person_id
		max_prob = 0.0
		
	if (person_id == person_id_pre):

		if (y_pred[i] > max_prob):
			max_prob = y_pred[i]

row = pd.DataFrame([[person_id_pre, max_prob]],columns=['person_id', 'score'])
person_ids_output_probs = person_ids_output_probs.append(row)			
person_ids_output_probs.to_csv('../output/predictions.csv', index = False)
print("Inferring stage finished")

