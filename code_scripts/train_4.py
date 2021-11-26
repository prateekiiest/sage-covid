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
#from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb

def add_COVID_measurement_date():
    measurement = pd.read_csv("../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/measurement.csv",usecols =['person_id','measurement_date','measurement_concept_id','value_as_concept_id'])
    measurement = measurement.loc[measurement['measurement_concept_id']==706163]
    measurement['value_as_concept_id'] = measurement['value_as_concept_id'].astype(int)
    measurement = measurement.loc[(measurement['value_as_concept_id']==45877985.0) | (measurement['value_as_concept_id']==45884084.0)]
    measurement = measurement.sort_values(['measurement_date'],ascending=False).groupby('person_id').head(1)
    covid_measurement = measurement[['person_id','measurement_date']]
    return covid_measurement

def feature_selection_measurement_auprc(X, Y):

        #because this is simple MIL model
        n_features = X.shape[1] // 3 

        #X_remain, X_opt, Y_remain, Y_opt = train_test_split(X, Y, test_size=0.25, stratify=Y)
	X_opt = X
	Y_opt = Y

        #idx = CFS.cfs(X_opt, Y_opt)

        print("n features total: {}".format(n_features))

        feature_step_size = 1
        best_n_features_selected = 1
        if (n_features < feature_step_size):
                feature_step_size = 1
                best_n_features_selected = 0

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

        #X_remain, X_opt, Y_remain, Y_opt = train_test_split(X, Y, test_size=0.25, stratify=Y)
	X_opt = X
	Y_opt = Y

        #idx = CFS.cfs(X_opt, Y_opt)

        print("n features total: {}".format(n_features))

        feature_step_size = 1
        best_n_features_selected = 1
        if (n_features < feature_step_size):
                feature_step_size = 1
                best_n_features_selected = 0

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

def report_features(selected_feature_indices, cumulative_sums_best_n_features_selected_list, concept_ids_selected_first_step, feature_importances, one_hot_encoder_categories, feature_reporting_filename):

        ofp = open(feature_reporting_filename, 'w')
        ofp.write('[{\n')
        ofp.write('\t"DateIndices": [{\n')
        ofp.write('\t\t"id": 1,\n')
        ofp.write('\t\t"name": "Date of RT-PCR Test",\n')
        ofp.write('\t\t"reference_column": "measurement_date",\n')
        ofp.write('\t\t"aggregator function": "None",\n')
        ofp.write('\t\t"date": "01-01-2020"\n')
        ofp.write('\t}],\n')
        ofp.write('\t"TimeRanges": [{\n')
        ofp.write('\t\t"id": 2,\n')
        ofp.write('\t\t"name": "measurement after date",\n')
        ofp.write('\t\t"date_index_id": 1,\n')
        ofp.write('\t\t"time_unit": "None",\n')
        ofp.write('\t\t"time_range": ">"\n')
        ofp.write('\t},\n')
        ofp.write('\t{\n')
        ofp.write('\t\t"id": 3,\n')
        ofp.write('\t\t"name": "measurement before date",\n')
        ofp.write('\t\t"date_index_id": 1,\n')
        ofp.write('\t\t"time_unit": "None",\n')
        ofp.write('\t\t"time_range": "<="\n')
        ofp.write('\t}],\n')
        ofp.write('\t"Features": [\n')

        n_feature_intervals = len(cumulative_sums_best_n_features_selected_list) - 1
        feature_id = 4
        feature_ids_or_rules = []
        blood_group_included_flag = 0
        feature_index_2 = 0

	#print(selected_feature_indices)
	#print(concept_ids_selected_first_step)
	#print(len(concept_ids_selected_first_step))
	#print(cumulative_sums_best_n_features_selected_list)

        for feature_index in selected_feature_indices:

                for interval_index in range(1, n_feature_intervals+1):

                        lower_limit = cumulative_sums_best_n_features_selected_list[interval_index-1]
                        upper_limit = cumulative_sums_best_n_features_selected_list[interval_index]
                        if (feature_index >= lower_limit) and (feature_index < upper_limit):
                                feature_interval_index = interval_index
                                break

                feature_weight = feature_importances[feature_index_2]
                feature_index_2 += 1

                if (feature_index != cumulative_sums_best_n_features_selected_list[-1]-1):
			#print("feature index=", feature_index)
                        concept_id = concept_ids_selected_first_step[feature_index]
			
                if (feature_interval_index >= 1) and (feature_interval_index <= 4) and ('3003694' in concept_id) and (blood_group_included_flag == 0):

                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "A Negative",\n')
                        ofp.write('\t\t"description": "Blood Type",\n')
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_concept_id"],\n')
                        ofp.write('\t\t"reference_value": 46237061,\n')
                        ofp.write('\t\t"function": "equal to",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "A Positive",\n')
                        ofp.write('\t\t"description": "Blood Type",\n')
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_concept_id"],\n')
                        ofp.write('\t\t"reference_value": 46238174,\n')
                        ofp.write('\t\t"function": "equal to",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "B Negative",\n')
                        ofp.write('\t\t"description": "Blood Type",\n')
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_concept_id"],\n')
                        ofp.write('\t\t"reference_value": 46237872,\n')
                        ofp.write('\t\t"function": "equal to",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "B Positive",\n')
                        ofp.write('\t\t"description": "Blood Type",\n')
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_concept_id"],\n')
                        ofp.write('\t\t"reference_value": 46237993,\n')
                        ofp.write('\t\t"function": "equal to",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "AB Negative",\n')
                        ofp.write('\t\t"description": "Blood Type",\n')
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_concept_id"],\n')
                        ofp.write('\t\t"reference_value": 46237994,\n')
                        ofp.write('\t\t"function": "equal to",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "AB Positive",\n')
                        ofp.write('\t\t"description": "Blood Type",\n')
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_concept_id"],\n')
                        ofp.write('\t\t"reference_value": 46237873,\n')
                        ofp.write('\t\t"function": "equal to",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "0 Negative",\n')
                        ofp.write('\t\t"description": "Blood Type",\n')
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_concept_id"],\n')
                        ofp.write('\t\t"reference_value": 46237435,\n')
                        ofp.write('\t\t"function": "equal to",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "0 Positive",\n')
                        ofp.write('\t\t"description": "Blood Type",\n')
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_concept_id"],\n')
                        ofp.write('\t\t"reference_value": 46237992,\n')
                        ofp.write('\t\t"function": "equal to",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1
                        blood_group_included_flag = 1

                #X measurement min (simple MIL)
                elif (feature_interval_index == 1):

                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "Minimum %s",\n' % str(concept_id))
                        ofp.write('\t\t"description": "Minimum for measurement %s",\n' % str(concept_id))
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_number"],\n')
                        ofp.write('\t\t"reference_value": "None",\n')
                        ofp.write('\t\t"function": "min",\n')
                        ofp.write('\t\t"time_range": 2\n')
                        ofp.write('\t},\n')
                        feature_id += 1
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "Minimum %s",\n' % str(concept_id))
                        ofp.write('\t\t"description": "Minimum for measurement %s",\n' % str(concept_id))
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_number"],\n')
                        ofp.write('\t\t"reference_value": "None",\n')
                        ofp.write('\t\t"function": "min",\n')
                        ofp.write('\t\t"time_range": 3\n')
                        ofp.write('\t},\n')
                        feature_ids_or_rules.append([feature_id, feature_id-1])
                        feature_id += 1

                #X measurement max (simple MIL)
                elif (feature_interval_index == 2):
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "Maximum %s",\n' % str(concept_id))
                        ofp.write('\t\t"description": "Maximum for measurement %s",\n' % str(concept_id))
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_number"],\n')
                        ofp.write('\t\t"reference_value": "None",\n')
                        ofp.write('\t\t"function": "max",\n')
                        ofp.write('\t\t"time_range": 2\n')
                        ofp.write('\t},\n')
                        feature_id += 1
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "Maximum %s",\n' % str(concept_id))
                        ofp.write('\t\t"description": "Maximum for measurement %s",\n' % str(concept_id))
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_number"],\n')
                        ofp.write('\t\t"reference_value": "None",\n')
                        ofp.write('\t\t"function": "max",\n')
                        ofp.write('\t\t"time_range": 3\n')
                        ofp.write('\t},\n')
                        feature_ids_or_rules.append([feature_id, feature_id-1])
                        feature_id += 1

                #X measurement mean (simple MIL)
                elif (feature_interval_index == 3):
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "Mean %s",\n' % str(concept_id))
                        ofp.write('\t\t"description": "Mean for measurement %s",\n' % str(concept_id))
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_number"],\n')
                        ofp.write('\t\t"reference_value": "None",\n')
                        ofp.write('\t\t"function": "mean",\n')
                        ofp.write('\t\t"time_range": 2\n')
                        ofp.write('\t},\n')
                        feature_id += 1
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "Mean %s",\n' % str(concept_id))
                        ofp.write('\t\t"description": "Mean for measurement %s",\n' % str(concept_id))
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_number"],\n')
                        ofp.write('\t\t"reference_value": "None",\n')
                        ofp.write('\t\t"function": "mean",\n')
                        ofp.write('\t\t"time_range": 3\n')
                        ofp.write('\t},\n')
                        feature_ids_or_rules.append([feature_id, feature_id-1])
                        feature_id += 1

                elif (feature_interval_index == 4):
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "Count %s",\n' % str(concept_id))
                        ofp.write('\t\t"description": "Count for measurement %s",\n' % str(concept_id))
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_concept_id"],\n')
                        ofp.write('\t\t"reference_value": 45877985,\n')
                        ofp.write('\t\t"function": "count",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "Count %s",\n' % str(concept_id))
                        ofp.write('\t\t"description": "Count for measurement %s",\n' % str(concept_id))
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": ["value_as_concept_id"],\n')
                        ofp.write('\t\t"reference_value": 45884084,\n')
                        ofp.write('\t\t"function": "count",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_ids_or_rules.append([feature_id, feature_id-1])
                        feature_id += 1

                elif (feature_interval_index == 5):
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "Count %s",\n' % str(concept_id))
                        ofp.write('\t\t"description": "Count for condition %s",\n' % str(concept_id))
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": "None",\n')
                        ofp.write('\t\t"reference_value": "None",\n')
                        ofp.write('\t\t"function": "count",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1

                elif (feature_interval_index == 6):
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "Count %s",\n' % str(concept_id))
                        ofp.write('\t\t"description": "Count for observation %s",\n' % str(concept_id))
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": "None",\n')
                        ofp.write('\t\t"reference_value": "None",\n')
                        ofp.write('\t\t"function": "count",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1

                elif (feature_interval_index == 7):
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "Count %s",\n' % str(concept_id))
                        ofp.write('\t\t"description": "Count for device exposure %s",\n' % str(concept_id))
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": "None",\n')
                        ofp.write('\t\t"reference_value": "None",\n')
                        ofp.write('\t\t"function": "count",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1

                elif (feature_interval_index == 8):
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "Count %s",\n' % str(concept_id))
                        ofp.write('\t\t"description": "Count for drug exposure %s",\n' % str(concept_id))
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": "None",\n')
                        ofp.write('\t\t"reference_value": "None",\n')
                        ofp.write('\t\t"function": "count",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1

                elif (feature_interval_index == 9):
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "Count %s",\n' % str(concept_id))
                        ofp.write('\t\t"description": "Count for procedure %s",\n' % str(concept_id))
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": "None",\n')
                        ofp.write('\t\t"reference_value": "None",\n')
                        ofp.write('\t\t"function": "count",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1

                elif (feature_interval_index == 10):
                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "Count %s",\n' % str(concept_id))
                        ofp.write('\t\t"description": "Count for visit %s",\n' % str(concept_id))
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": [%d],\n' % int(concept_id))
                        ofp.write('\t\t"value_column": "None",\n')
                        ofp.write('\t\t"reference_value": "None",\n')
                        ofp.write('\t\t"function": "count",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1

                elif (feature_interval_index == 11):

                        ofp.write('\t{\n')
                        ofp.write('\t\t"id": %d,\n' % feature_id)
                        ofp.write('\t\t"name": "Age",\n')
                        ofp.write('\t\t"description": "Age",\n')
                        ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                        ofp.write('\t\t"concept_codes": "None",\n')
                        ofp.write('\t\t"value_column": "None",\n')
                        ofp.write('\t\t"reference_value": "None",\n')
                        ofp.write('\t\t"function": "None",\n')
                        ofp.write('\t\t"time_range": "None"\n')
                        ofp.write('\t},\n')
                        feature_id += 1

        #gender information
        n_gender_categories = len(one_hot_encoder_categories[0])

        for i in range(n_gender_categories):

                feature_weight = feature_importances[feature_index_2]
                feature_index_2 += 1
                ofp.write('\t{\n')
                ofp.write('\t\t"id": %d,\n' % feature_id)
                ofp.write('\t\t"name": "%s",\n' % str(one_hot_encoder_categories[0][i]))
                ofp.write('\t\t"description": "Gender information %s",\n' % str(one_hot_encoder_categories[0][i]))
                ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                ofp.write('\t\t"concept_codes": [%d],\n' % int(one_hot_encoder_categories[0][i]))
                ofp.write('\t\t"value_column": "None",\n')
                ofp.write('\t\t"reference_value": "None",\n')
                ofp.write('\t\t"function": "presence",\n')
                ofp.write('\t\t"time_range": "None"\n')
                ofp.write('\t},\n')
                feature_id += 1

        #race information
        n_race_categories = len(one_hot_encoder_categories[1])

        for i in range(n_race_categories):

                feature_weight = feature_importances[feature_index_2]
                feature_index_2 += 1
                ofp.write('\t{\n')
                ofp.write('\t\t"id": %d,\n' % feature_id)
                ofp.write('\t\t"name": "%s",\n' % str(one_hot_encoder_categories[1][i]))
                ofp.write('\t\t"description": "Race information %s",\n' % str(one_hot_encoder_categories[1][i]))
                ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                ofp.write('\t\t"concept_codes": [%d],\n' % int(one_hot_encoder_categories[1][i]))
                ofp.write('\t\t"value_column": "None",\n')
                ofp.write('\t\t"reference_value": "None",\n')
                ofp.write('\t\t"function": "presence",\n')
                ofp.write('\t\t"time_range": "None"\n')
                ofp.write('\t},\n')
                feature_id += 1

        #ethnicity information
        n_ethnicity_categories = len(one_hot_encoder_categories[2])

        for i in range(n_ethnicity_categories):

                feature_weight = feature_importances[feature_index_2]
                feature_index_2 += 1
                ofp.write('\t{\n')
                ofp.write('\t\t"id": %d,\n' % feature_id)
                ofp.write('\t\t"name": "%s",\n' % str(one_hot_encoder_categories[2][i]))
                ofp.write('\t\t"description": "Ethnicity information %s",\n' % str(one_hot_encoder_categories[2][i]))
                ofp.write('\t\t"weight": %.11f,\n' % feature_weight)
                ofp.write('\t\t"concept_codes": [%d],\n' % int(one_hot_encoder_categories[2][i]))
                ofp.write('\t\t"value_column": "None",\n')
                ofp.write('\t\t"reference_value": "None",\n')
                ofp.write('\t\t"function": "presence",\n')
                ofp.write('\t\t"time_range": "None"\n')
                if (i == n_ethnicity_categories-1):
                        ofp.write('\t}],\n')
                else:
                        ofp.write('\t},\n')
                feature_id += 1

        n_or_rules = len(feature_ids_or_rules)

        ofp.write('\t"LogicalOperators": [\n')
        for i in range(n_or_rules):

                ofp.write('\t{\n')
                ofp.write('\t\t"id": %d,\n' % feature_id)
                ofp.write('\t\t"feature_id_1": %d,\n' % feature_ids_or_rules[i][1])
                ofp.write('\t\t"feature_id_2": %d,\n' % feature_ids_or_rules[i][0])
                ofp.write('\t\t"logical operator": "or"\n')
                ofp.write('\t},\n')
                feature_id += 1
	ofp.write('\t]\n')
        ofp.write('}]\n')
        ofp.close()

print(datetime.now())

measurement_concept_ids_filename = "../../data/measurement_concept_ids.list"
condition_concept_ids_filename = "../../data/condition_concept_ids.list"
observation_concept_ids_filename = "../../data/observation_concept_ids.list"
device_exposure_concept_ids_filename = "../../data/device_exposure_concept_ids.list"
drug_exposure_concept_ids_filename = "../../data/drug_exposure_concept_ids.list"
procedure_concept_ids_filename = "../../data/procedure_concept_ids.list"
visit_concept_ids_filename = "../../data/visit_concept_ids.list"

gs = pd.read_csv('../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/goldstandard.csv')

person_ids_positives = np.array(gs.loc[gs['status'] == 1.0, 'person_id'])

print(datetime.now())
print("Load measurement.csv")

measurement = pd.read_csv('../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/measurement.csv',usecols = ['measurement_concept_id','value_as_number','person_id', 'measurement_date', 'measurement_time', 'value_as_concept_id'])
measurement = measurement.dropna(subset = ['measurement_concept_id'])
measurement = measurement.astype({"measurement_concept_id": int})
measurement = measurement.astype({"measurement_concept_id": str})
measurement['value_as_number'] = measurement['value_as_number'].fillna(0.0)
measurement['measurement_date'] = measurement['measurement_date'].fillna('1900-01-01')
measurement['measurement_time'] = measurement['measurement_time'].fillna('1900-01-01')
measurement_rt_pcr = measurement.loc[(measurement['value_as_concept_id']==45877985.0) | (measurement['value_as_concept_id']==45884084.0)]

print(measurement.shape)

print(datetime.now())
print("Sort measurement concept ids")

measurement_positives_after_covid = measurement.query('measurement_date > "2019-11-17" and person_id in @person_ids_positives')

if (measurement_positives_after_covid.empty):
        measurement_positives_after_covid = measurement.query('person_id in @person_ids_positives')

measurement_positives_after_covid['counts'] = measurement_positives_after_covid.groupby('measurement_concept_id')['measurement_concept_id'].transform('count')
measurement_positives_after_covid.sort_values('counts', inplace=True, ascending=False)
#measurement_positives_after_covid = measurement_positives_after_covid.query('counts > 15')
measurement_positives_after_covid = measurement_positives_after_covid.drop_duplicates('measurement_concept_id')
measurement_concept_ids_sorted_wrt_frequency = measurement_positives_after_covid['measurement_concept_id'].tolist()
#print(measurement_concept_ids_sorted_wrt_frequency)

print(datetime.now())

print("Load condition.csv")
condition = pd.read_csv("../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/condition_occurrence.csv",usecols = ['condition_concept_id','person_id', 'condition_start_date', 'condition_end_date'])

condition = condition.dropna(subset = ['condition_concept_id'])
condition = condition.astype({"condition_concept_id": int})
condition = condition.astype({"condition_concept_id": str})
condition['condition_end_date'] = condition['condition_end_date'].fillna('2100-01-01')
condition['condition_start_date'] = condition['condition_start_date'].fillna('1900-01-01')

print(datetime.now())

print("Sort condition concept ids")

condition_positives_after_covid = condition.query('condition_end_date > "2019-11-17" and person_id in @person_ids_positives')

if (condition_positives_after_covid.empty):
        condition_positives_after_covid = condition.query('person_id in @person_ids_positives')

condition_positives_after_covid['counts'] = condition_positives_after_covid.groupby('condition_concept_id')['condition_concept_id'].transform('count')
#print(measurement_positives_after_covid)
condition_positives_after_covid.sort_values('counts', inplace=True, ascending=False)
#condition_positives_after_covid = condition_positives_after_covid.query('counts > 15')
condition_concept_ids_sorted_wrt_frequency = condition_positives_after_covid.drop_duplicates('condition_concept_id')['condition_concept_id'].tolist()

print(datetime.now())

print("Load observation.csv")
observation = pd.read_csv("../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/observation.csv",usecols = ['observation_concept_id','person_id', 'observation_date'])

observation = observation.dropna(subset = ['observation_concept_id'])
observation = observation.astype({"observation_concept_id": int})
observation = observation.astype({"observation_concept_id": str})

print(datetime.now())

print("Sort observation concept ids")

observation_positives_after_covid = observation.query('person_id in @person_ids_positives')

if (observation_positives_after_covid.empty):
        observation_positives_after_covid = observation.query('person_id in @person_ids_positives')

observation_positives_after_covid['counts'] = observation_positives_after_covid.groupby('observation_concept_id')['observation_concept_id'].transform('count')
#print(measurement_positives_after_covid)
observation_positives_after_covid.sort_values('counts', inplace=True, ascending=False)
#observation_positives_after_covid = observation_positives_after_covid.query('counts > 15')
observation_concept_ids_sorted_wrt_frequency = observation_positives_after_covid.drop_duplicates('observation_concept_id')['observation_concept_id'].tolist()

print(datetime.now())

print("Load device_exposure.csv")
device_exposure = pd.read_csv("../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/device_exposure.csv",usecols = ['device_concept_id','person_id', 'device_exposure_start_date', 'device_exposure_end_date'])

device_exposure = device_exposure.dropna(subset = ['device_concept_id'])
device_exposure = device_exposure.astype({"device_concept_id": int})
device_exposure = device_exposure.astype({"device_concept_id": str})
device_exposure['device_exposure_end_date'] = device_exposure['device_exposure_end_date'].fillna('2100-01-01')
device_exposure['device_exposure_start_date'] = device_exposure['device_exposure_start_date'].fillna('1900-01-01')

print(datetime.now())

print("Sort device_exposure concept ids")

device_exposure_positives_after_covid = device_exposure.query('device_exposure_end_date > "2019-11-17" and person_id in @person_ids_positives')

if (device_exposure_positives_after_covid.empty):
        device_exposure_positives_after_covid = device_exposure.query('person_id in @person_ids_positives')

device_exposure_positives_after_covid['counts'] = device_exposure_positives_after_covid.groupby('device_concept_id')['device_concept_id'].transform('count')
#print(measurement_positives_after_covid)
device_exposure_positives_after_covid.sort_values('counts', inplace=True, ascending=False)
#device_exposure_positives_after_covid = device_exposure_positives_after_covid.query('counts > 15')
device_concept_ids_sorted_wrt_frequency = device_exposure_positives_after_covid.drop_duplicates('device_concept_id')['device_concept_id'].tolist()

print(datetime.now())

print("Load drug_exposure.csv")
drug_exposure = pd.read_csv('../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/drug_exposure.csv',usecols = ['drug_concept_id', 'person_id', 'drug_exposure_start_date', 'drug_exposure_end_date'])

drug_exposure = drug_exposure.dropna(subset = ['drug_concept_id'])
drug_exposure = drug_exposure.astype({"drug_concept_id": int})
drug_exposure = drug_exposure.astype({"drug_concept_id": str})
drug_exposure['drug_exposure_end_date'] = drug_exposure['drug_exposure_end_date'].fillna('2100-01-01')
drug_exposure['drug_exposure_start_date'] = drug_exposure['drug_exposure_start_date'].fillna('1900-01-01')

print(datetime.now())

print("Sort drug_exposure concept ids")

drug_exposure_positives_after_covid = drug_exposure.query('drug_exposure_end_date > "2019-11-17" and person_id in @person_ids_positives')

if (drug_exposure_positives_after_covid.empty):
	drug_exposure_positives_after_covid = drug_exposure.query('person_id in @person_ids_positives')

drug_exposure_positives_after_covid['counts'] = drug_exposure_positives_after_covid.groupby('drug_concept_id')['drug_concept_id'].transform('count')
#print(measurement_positives_after_covid)
drug_exposure_positives_after_covid.sort_values('counts', inplace=True, ascending=False)
#drug_exposure_positives_after_covid = drug_exposure_positives_after_covid.query('counts > 15')
drug_concept_ids_sorted_wrt_frequency = drug_exposure_positives_after_covid.drop_duplicates('drug_concept_id')['drug_concept_id'].tolist()

print(datetime.now())

print("Load procedure_occurrence.csv")

procedure = pd.read_csv('../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/procedure_occurrence.csv',usecols = ['procedure_concept_id', 'person_id', 'procedure_date'])

procedure = procedure.dropna(subset = ['procedure_concept_id'])
procedure = procedure.astype({"procedure_concept_id": int})
procedure = procedure.astype({"procedure_concept_id": str})
procedure['procedure_date'] = procedure['procedure_date'].fillna('1900-01-01')

print("Sort procedure concept ids")

procedure_positives_after_covid = procedure.query('procedure_date > "2019-11-17" and person_id in @person_ids_positives')

if (procedure_positives_after_covid.empty):
	procedure_positives_after_covid = procedure.query('person_id in @person_ids_positives')

procedure_positives_after_covid['counts'] = procedure_positives_after_covid.groupby('procedure_concept_id')['procedure_concept_id'].transform('count')
#print(measurement_positives_after_covid)
procedure_positives_after_covid.sort_values('counts', inplace=True, ascending=False)
#procedure_positives_after_covid = procedure_positives_after_covid.query('counts > 15')
procedure_concept_ids_sorted_wrt_frequency = procedure_positives_after_covid.drop_duplicates('procedure_concept_id')['procedure_concept_id'].tolist()

print("Load visit_occurrence.csv")

visit = pd.read_csv('../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/visit_occurrence.csv',usecols = ['visit_concept_id', 'person_id', 'visit_start_date', 'visit_end_date'])

visit = visit.dropna(subset = ['visit_concept_id'])
visit = visit.astype({"visit_concept_id": int})
visit = visit.astype({"visit_concept_id": str})
visit['visit_start_date'] = visit['visit_start_date'].fillna('1900-01-01')
visit['visit_end_date'] = visit['visit_end_date'].fillna('2100-01-01')

print("Sort visit concept ids")

visit_positives_after_covid = visit.query('visit_end_date > "2019-11-17" and person_id in @person_ids_positives')

if (visit_positives_after_covid.empty):
	visit_positives_after_covid = visit.query('person_id in @person_ids_positives')

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

person = pd.read_csv('../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/person.csv')
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
#query_str = 'condition_end_date > "2020-01-01" and person_id == @person_id'
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
#query_str = 'device_exposure_end_date > "2020-01-01" and person_id == @person_id'
query_str = 'person_id == @person_id'
concept_id_str = 'device_concept_id'
feature_indices = device_exposure_feature_indices
X = X_device_exposure
X_device_exposure = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, concept_id_str, feature_indices, X)

print(datetime.now())
print("drug exposures")

df = drug_exposure
#query_str = 'drug_exposure_end_date > "2020-01-01" and person_id == @person_id'
query_str = 'person_id == @person_id'
concept_id_str = 'drug_concept_id'
feature_indices = drug_exposure_feature_indices
X = X_drug_exposure
X_drug_exposure = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, concept_id_str, feature_indices, X)

print(datetime.now())
print("procedures")

df = procedure
#query_str = 'procedure_date > "2020-01-01" and person_id == @person_id'
query_str = 'person_id == @person_id'
concept_id_str = 'procedure_concept_id'
feature_indices = procedure_feature_indices
X = X_procedure
X_procedure = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, concept_id_str, feature_indices, X)

print(datetime.now())
print("visits")

df = visit
#query_str = 'visit_end_date > "2020-01-01" and person_id == @person_id'
query_str = 'person_id == @person_id'
concept_id_str = 'visit_concept_id'
feature_indices = visit_feature_indices
X = X_visit
X_visit = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, concept_id_str, feature_indices, X)

print(datetime.now())
print("ages")

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

enc = OneHotEncoder(handle_unknown='ignore')

genders = person['gender_concept_id'].to_numpy()
races = person['race_concept_id'].to_numpy()
ethnicities = person['ethnicity_concept_id'].to_numpy()

for i in range(n_persons):

        person_id = person_ids[i]
        index_p = person_indices[person_id]

        X_gender_race_ethnicity_raw[index_p][0] = genders[i]
        X_gender_race_ethnicity_raw[index_p][1] = races[i]
        X_gender_race_ethnicity_raw[index_p][2] = ethnicities[i]

one_hot_encoder = enc.fit(X_gender_race_ethnicity_raw)
X_gender_race_ethnicity = enc.transform(X_gender_race_ethnicity_raw).toarray()
one_hot_encoder_categories = one_hot_encoder.categories_

X = np.concatenate((X_min, X_max), axis=1)
X_measurement = np.concatenate((X, X_ave), axis=1)

print("true labels")

gs = pd.read_csv('../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/goldstandard.csv')
person_status = person.merge(gs, how = 'left', on = ['person_id'])
person_status.drop_duplicates(subset=['person_id'], keep = 'first',inplace = True)
person_status.fillna(0,inplace = True)
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
X_measurement_rt_pcr_selected, best_n_features_selected_measurement_rt_pcr = feature_selection_auprc(X_measurement_rt_pcr, Y)
X_condition_selected, best_n_features_selected_condition = feature_selection_auprc(X_condition, Y)
X_observation_selected, best_n_features_selected_observation = feature_selection_auprc(X_observation, Y)
X_device_exposure_selected, best_n_features_selected_device_exposure = feature_selection_auprc(X_device_exposure, Y)
X_drug_exposure_selected, best_n_features_selected_drug_exposure = feature_selection_auprc(X_drug_exposure, Y)
X_procedure_selected, best_n_features_selected_procedure = feature_selection_auprc(X_procedure, Y)
X_visit_selected, best_n_features_selected_visit = feature_selection_auprc(X_visit, Y)

best_n_features_selected_list = [best_n_features_selected_measurement, best_n_features_selected_measurement, best_n_features_selected_measurement, best_n_features_selected_measurement_rt_pcr, best_n_features_selected_condition, best_n_features_selected_observation, best_n_features_selected_device_exposure, best_n_features_selected_drug_exposure, best_n_features_selected_procedure, best_n_features_selected_visit, 1]

cumulative_sums_best_n_features_selected_list = [sum(best_n_features_selected_list[:x]) for x in range(0,len(best_n_features_selected_list)+1)]

measurement_feature_concept_ids_selected_first_step = measurement_feature_concept_ids[:best_n_features_selected_measurement]
measurement_feature_concept_ids_selected_first_step_2 = measurement_feature_concept_ids[:best_n_features_selected_measurement_rt_pcr]
condition_feature_concept_ids_selected_first_step = condition_feature_concept_ids[:best_n_features_selected_condition]
observation_feature_concept_ids_selected_first_step = observation_feature_concept_ids[:best_n_features_selected_observation]
device_exposure_feature_concept_ids_selected_first_step = device_exposure_feature_concept_ids[:best_n_features_selected_device_exposure]
drug_exposure_feature_concept_ids_selected_first_step = drug_exposure_feature_concept_ids[:best_n_features_selected_drug_exposure]
procedure_feature_concept_ids_selected_first_step = procedure_feature_concept_ids[:best_n_features_selected_procedure]
visit_feature_concept_ids_selected_first_step = visit_feature_concept_ids[:best_n_features_selected_visit]

concept_ids_selected_first_step = []
concept_ids_selected_first_step = measurement_feature_concept_ids_selected_first_step + measurement_feature_concept_ids_selected_first_step + measurement_feature_concept_ids_selected_first_step + measurement_feature_concept_ids_selected_first_step_2 + condition_feature_concept_ids_selected_first_step + observation_feature_concept_ids_selected_first_step + device_exposure_feature_concept_ids_selected_first_step + drug_exposure_feature_concept_ids_selected_first_step + procedure_feature_concept_ids_selected_first_step + visit_feature_concept_ids_selected_first_step

print("concatenation of feature matrices")

X_selected = np.concatenate((X_measurement_selected, X_measurement_rt_pcr_selected), axis=1)
X_selected = np.concatenate((X_selected, X_condition_selected), axis=1)
X_selected = np.concatenate((X_selected, X_observation_selected), axis=1)
X_selected = np.concatenate((X_selected, X_device_exposure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_drug_exposure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_procedure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_visit_selected), axis=1)
X_selected = np.concatenate((X_selected, X_age), axis=1)

#X_selected = np.concatenate((X_selected, X_gender_race_ethnicity), axis=1)

selector = SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=50)).fit(X_selected, Y)
X_selected = selector.transform(X_selected)
selected_feature_indices = selector.get_support(indices=True) #these indices start from zero

#transformer = RobustScaler().fit(X_selected)
transformer = MinMaxScaler().fit(X_selected)
X_selected = transformer.transform(X_selected)

X_selected = np.concatenate((X_selected, X_gender_race_ethnicity), axis=1)

selected_feature_indices = selector.get_support(indices=True)
scalerfile = '../../model/feature_selector_stage_2.sav'
dump(selector, open(scalerfile, 'wb'))

print("X_selected.shape")
print(X_selected.shape)

print(datetime.now())
print("model training")

#clf = LogisticRegression(solver='saga').fit(X_selected, Y)
#clf = MLPClassifier(learning_rate_init=0.01).fit(X_selected, Y)
clf = lgb.LGBMClassifier(n_jobs = 4).fit(X_selected, Y)
feature_importances = clf.feature_importances_
feature_importances = feature_importances.astype('float32')

sum_feature_importances = sum(feature_importances)
for i in range(len(feature_importances)):
        feature_importances[i] /= float(sum_feature_importances)

feature_reporting_filename = '../../model/features.json'

report_features(selected_feature_indices, cumulative_sums_best_n_features_selected_list, concept_ids_selected_first_step, feature_importances, one_hot_encoder_categories, feature_reporting_filename)

print(datetime.now())

dump(clf, '../../model/baseline.joblib')
dump(one_hot_encoder, '../../model/one_hot_encoder')
dump(le, '../../model/label_encoder')
dump(transformer, '../../model/normalizer')
dump(best_n_features_selected_measurement, '../../model/best_n_features_selected_measurement')
dump(best_n_features_selected_measurement_rt_pcr, '../../model/best_n_features_selected_measurement_rt_pcr')
dump(best_n_features_selected_condition, '../../model/best_n_features_selected_condition')
dump(best_n_features_selected_observation, '../../model/best_n_features_selected_observation')
dump(best_n_features_selected_device_exposure, '../../model/best_n_features_selected_device_exposure')
dump(best_n_features_selected_drug_exposure, '../../model/best_n_features_selected_drug_exposure')
dump(best_n_features_selected_procedure, '../../model/best_n_features_selected_procedure')
dump(best_n_features_selected_visit, '../../model/best_n_features_selected_visit')

print("Training stage finished")

