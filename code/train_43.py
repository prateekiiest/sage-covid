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
person = pd.read_csv('../ehr-dream-challenges/examples/covid19-question-1/synthetic_data/training/person.csv')

measurement = measurement.query('measurement_date > "2020-01-01"')
measurement.sort_values(['person_id', 'measurement_date'], inplace=True)

n_measurement_features = len(measurement_feature_concept_ids)+1

X_measurement = np.array([]).reshape(0, n_measurement_features)

person_status = person.merge(gs, how = 'left', on = ['person_id'])
person_status.drop_duplicates(subset=['person_id'], keep = 'first',inplace = True)
Y =  np.array(person_status[['status']]).ravel()
y_measurement = np.array([]).reshape(0, 1)

le = preprocessing.LabelEncoder()

n_positives = 0
n_negatives = 0

measurement_concept_ids_for_rules = {'3020891':37.5,'3027018':100,'3012888':80,'3004249':120,
'3023314':52,'3013650':8,'3004327':1.0,'3016502':95, '3010156':0.8, '3023091':5.9, '3024929':100, '3022250':210, '42870366':0.5, '3021337':0.4}

person = person.drop_duplicates(subset = ['person_id'])
n_persons = person.shape[0]

n_instances_list = []

for person_id in measurement[['person_id']].drop_duplicates(['person_id']).values[:,0]:

	sub_matrix = measurement.query('person_id == @person_id')

	class_label = person_status.query('person_id == @person_id')[['status']].values[:,0]

	n_instances_per_person = 0
	for measurement_date in sub_matrix[['measurement_date']].drop_duplicates(['measurement_date']).values[:,0]:

		feature_vector = np.zeros((1, n_measurement_features))
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
	
			feature_vector[0][n_measurement_features-1] = biomarker_count

			if (class_label == 1.0) and (biomarker_count == 0):
				class_label = 0
				
			X_measurement = np.vstack([X_measurement, feature_vector])
			y_measurement = np.vstack([y_measurement, np.array(class_label)])

			n_instances_per_person += 1

			if (class_label == 1.0):
				n_positives += 1
			else:
				n_negatives += 1

	n_instances_list += [n_instances_per_person]

	if (n_positives >= 2000) and (n_negatives >= 2000):
		break

	#print(person_id)

y_measurement = y_measurement.ravel()

if ('3003694' in measurement_feature_indices):

	index_f = measurement_feature_indices['3003694']
	le.fit(X_measurement[:, index_f])
	X_measurement[:, index_f] = le.transform(X_measurement[:, index_f])

print("X_measurement.shape")
print(X_measurement.shape)

print(datetime.now())
print("feature selection")

X_measurement_selected, best_n_features_selected_measurement = feature_selection_auprc(X_measurement, y_measurement)

X_selected = X_measurement_selected
n_features_selected = X_selected.shape[1]
n_persons_included = len(n_instances_list)

X_arff_filename = "../model/measurements_train.arff"
X_arff_file = open(X_arff_filename, 'w')
X_arff_file.write('@relation measurements_multi_instance\n')
X_arff_file.write('@attribute bag_id {')
#for i in range(n_persons_included-1):
for i in range(99999):
        X_arff_file.write('{},'.format(i+1))
#X_arff_file.write('{}}}\n'.format(n_persons_included))
X_arff_file.write('{}}}\n'.format(100000))
X_arff_file.write('@attribute bag relational\n')
for i in range(n_features_selected):
        X_arff_file.write('@attribute f{} numeric\n'.format(i+1))
X_arff_file.write('@end bag\n')
X_arff_file.write('@attribute class{0,1}\n')
X_arff_file.write('@data\n')

instance_index = 0

for i in range(n_persons_included):

	n_instances_per_person = n_instances_list[i]
	person_index = i+1
	X_arff_file.write('{},\''.format(person_index))
	class_label = y_measurement[instance_index]

	for j in range(n_instances_per_person-1):
		
		for k in range(n_features_selected-1):
			X_arff_file.write('{:.2f},'.format(X_selected[instance_index][k]))
		X_arff_file.write('{:.2f}\\n'.format(X_selected[instance_index][n_features_selected-1]))
		instance_index += 1

        for j in range(1): #for the last bag

                for k in range(n_features_selected-1):
                        X_arff_file.write('{:.2f},'.format(X_selected[instance_index][k]))
                X_arff_file.write('{:.2f}\',{}\n'.format(X_selected[instance_index][n_features_selected-1],int(class_label)))
                instance_index += 1

X_arff_file.close()

dump(measurement_feature_concept_ids, measurement_concept_ids_filename)

#print("model training")
#clf = LogisticRegression(solver='saga').fit(X_selected, y_measurement)

print(datetime.now())

#dump(clf, '../model/baseline.joblib')
dump(best_n_features_selected_measurement, '../model/best_n_features_selected_measurement')
dump(n_persons_included, '../model/n_persons_included_in_train_set')

#print("Training stage finished")

