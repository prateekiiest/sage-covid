#performs 10-fold CV using logistic regression with 44 numeric features (measurement, condition, age)

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
import math

def feature_selection_measurement_auprc(X, Y):

	#because this is simple MIL model
        n_features = X.shape[1] / 3 

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

	n_iterations = n_features / feature_step_size
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

print(datetime.now())
print("true labels")

measurement_concept_ids_filename = "../model/measurement_concept_ids.list"

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

measurement_feature_concept_ids = measurement_concept_ids_sorted_wrt_frequency[0:100]
measurement_feature_indices = dict(zip(measurement_feature_concept_ids, range(len(measurement_feature_concept_ids))))

measurement = measurement.query('measurement_date > "2020-01-01"')
measurement.sort_values(['person_id', 'measurement_date'], inplace=True)

n_measurement_features = len(measurement_feature_concept_ids)

person = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/person.csv')
person = person.drop_duplicates(subset = ['person_id'])
person_indices = dict(zip(person.person_id, range(len(person.person_id))))

person_status = person.merge(gs, how = 'left', on = ['person_id'])
person_status.drop_duplicates(subset=['person_id'], keep = 'first',inplace = True)
Y =  np.array(person_status[['status']]).ravel()

X_min = np.zeros((len(person_indices), n_measurement_features))
X_max = np.zeros((len(person_indices), n_measurement_features))
X_ave = np.zeros((len(person_indices), n_measurement_features))

X_biomarker = np.zeros((len(person_indices), 1))

le = preprocessing.LabelEncoder()

measurement_concept_ids_values_for_rules = {'3020891':37.5,'3027018':100,'3012888':80,'3004249':120, '3023314':52,'3013650':8,'3004327':1.0,'3016502':95, '3010156':0.8, '3023091':5.9, '3024929':100, '3022250':210, '42870366':0.5, '3021337':0.4}

for i in measurement_concept_ids_values_for_rules.keys():

	sub_matrix = measurement[measurement['measurement_concept_id'] == i]

	if (i != '3016502') and (i != '3024929') and (i != '3004327'):
		sub_matrix['count'] = sub_matrix[sub_matrix['value_as_number'] > measurement_concept_ids_values_for_rules[i]].groupby('person_id')['person_id'].transform('count')
	else:
		sub_matrix['count'] = sub_matrix[sub_matrix['value_as_number'] < measurement_concept_ids_values_for_rules[i]].groupby('person_id')['person_id'].transform('count')

	sub_matrix = sub_matrix.drop_duplicates('person_id')
	sub_matrix['count'] = sub_matrix['count'].fillna(0)
	person_ids = sub_matrix['person_id'].to_numpy()
	count_values = sub_matrix['count'].to_numpy()

	for j in range(len(person_ids)):
		person_id = person_ids[j]
		count_value = count_values[j]
		person_index = person_indices[person_id]
		X_biomarker[person_index][0] = count_value

n_positives = 0
n_negatives = 0

for person_id in measurement[['person_id']].drop_duplicates(['person_id']).values[:,0]:

	person_index = person_indices[person_id]
	class_label = Y[person_index]

	sub_matrix = measurement.query('person_id == @person_id')
	sub_matrix_min = sub_matrix.groupby('measurement_concept_id')['value_as_number'].min()
        sub_matrix_max = sub_matrix.groupby('measurement_concept_id')['value_as_number'].max()
        sub_matrix_ave = sub_matrix.groupby('measurement_concept_id')['value_as_number'].mean()

	measurement_values_available_min = []
        measurement_values_available_max = []
        measurement_values_available_ave = []
	measurement_feature_indices_per_person = []
	
	for measurement_concept_id in sub_matrix_min.keys():

		if (measurement_concept_id in measurement_feature_indices):		

			measurement_value_min = sub_matrix_min[measurement_concept_id]
                        measurement_value_max = sub_matrix_max[measurement_concept_id]
                        measurement_value_ave = sub_matrix_ave[measurement_concept_id]

			if (not math.isnan(measurement_value_min)) and (not math.isnan(measurement_value_max)) and (not math.isnan(measurement_value_ave)):	
				measurement_values_available_min += [measurement_value_min]
                                measurement_values_available_max += [measurement_value_max]
                                measurement_values_available_ave += [measurement_value_ave]
				measurement_feature_indices_per_person += [measurement_feature_indices[measurement_concept_id]]

	measurement_values_available_min = np.array(measurement_values_available_min)
	measurement_values_available_max = np.array(measurement_values_available_max)
	measurement_values_available_ave = np.array(measurement_values_available_ave)
	measurement_feature_indices_per_person = np.array(measurement_feature_indices_per_person)
	if (measurement_feature_indices_per_person.size > 0):
		X_min[person_index][measurement_feature_indices_per_person] = measurement_values_available_min
		X_max[person_index][measurement_feature_indices_per_person] = measurement_values_available_max
		X_ave[person_index][measurement_feature_indices_per_person] = measurement_values_available_ave

	if (class_label == 1.0):
		n_positives += 1
	else:
		n_negatives += 1

	if (n_positives >= 2000) and (n_negatives >= 2000):
		break

	#print(person_id)

if ('3003694' in measurement_feature_indices):

	feature_index = measurement_feature_indices['3003694']
        le.fit(X_min[:, feature_index])
        X_min[:, feature_index] = le.transform(X_min[:, feature_index])
        le.fit(X_max[:, feature_index])
        X_max[:, feature_index] = le.transform(X_max[:, feature_index])
        le.fit(X_ave[:, feature_index])
        X_ave[:, feature_index] = le.transform(X_ave[:, feature_index])

X_measurement = np.concatenate((X_min, X_max), axis=1)
X_measurement = np.concatenate((X_measurement, X_ave), axis=1)

print(datetime.now())
print("feature selection")

X_measurement_selected, best_n_features_selected_measurement = feature_selection_measurement_auprc(X_measurement, Y)
X_selected = np.concatenate((X_measurement_selected, X_biomarker), axis=1)

selector = SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=50)).fit(X_selected, Y)
X_selected = selector.transform(X_selected)
selected_feature_indices = selector.get_support(indices=True)
scalerfile = '../model/feature_selector_stage_2.sav'
dump(selector, open(scalerfile, 'wb'))

print("X_selected.shape")
print(X_selected.shape)

dump(measurement_feature_concept_ids, measurement_concept_ids_filename)

print("model training")

clf = LogisticRegression(solver='saga').fit(X_selected, Y)

print(datetime.now())

dump(clf, '../model/baseline.joblib')
dump(best_n_features_selected_measurement, '../model/best_n_features_selected_measurement')

print("Training stage finished")

