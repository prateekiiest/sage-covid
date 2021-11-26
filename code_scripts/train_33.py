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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from xgboost import XGBClassifier

def read_data_frame(csv_filename, usecols_list, dropna_subset, concept_id_name, fillna_field, fillna_value, fillna_field_2='', fillna_value_2=0.0):

        df = pd.read_csv(csv_filename, usecols=usecols_list)
        df = df.dropna(subset = dropna_subset)
        df = df.astype({str(concept_id_name): int})
        df = df.astype({str(concept_id_name): str})
        df[fillna_field] = df[fillna_field].fillna(fillna_value)
        if (not fillna_field_2):
                df[fillna_field_2] = df[fillna_field_2].fillna(fillna_value_2)

        return df

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

                clf = LogisticRegression(solver='saga', n_jobs=-1)

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
                best_n_features_selected = 1

        n_iterations = n_features // feature_step_size
        max_auprc = 0

        for i in range(n_iterations):

                #print(i+1)

                n_features_selected = (i+1)*feature_step_size
                X_opt_selected = X_opt[:, 0:n_features_selected]

                #print(X_visit_opt_selected.shape)

                clf = LogisticRegression(solver='saga', n_jobs=-1)

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

		measurement_date = "2020-01-01"

                subm_after_covid = measurement.query('measurement_date >= @measurement_date and person_id == @person_id')
                subm_before_covid = measurement.query('measurement_date < @measurement_date and person_id == @person_id')

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

def prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, query_str_2,  concept_id_str, feature_indices, X):

        for person_id in person_ids:

                index_p = person_indices[person_id]

                measurement_date_array = covid_measurement[covid_measurement['person_id'] == person_id][['measurement_date']].to_numpy()

                if (measurement_date_array.size != 0):
                        measurement_date = measurement_date_array[0][0]
                else:
                        continue

                subm_after_covid = df.query(query_str + ' >= @measurement_date and ' + query_str_2)

                if (not subm_after_covid.empty):

                        subm_covid_count = subm_after_covid.groupby(concept_id_str)['person_id'].count()
                        concept_ids = subm_covid_count.index.tolist()
		
                        for concept_id in concept_ids:
                                if (concept_id not in feature_indices):
                                        continue
                                index_f = feature_indices[concept_id]
                                X[index_p][index_f] = subm_covid_count[concept_id]
        return X

def prepare_feature_matrix_count_based_2(person_ids, person_indices, df, query_str, concept_id_str, feature_indices, X):

        for person_id in person_ids:

                index_p = person_indices[person_id]

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

def prepare_feature_matrix_drug(df, feature_indices, query_string, X_table):

        concept_ids = list(feature_indices.keys())

        for i in concept_ids:

                index_f = feature_indices[i]

                subm = df.query(query_string)

                if (subm.empty):
                        continue

                person_ids = subm['person_id'].to_numpy()
                quantities = subm['quantity'].to_numpy()

                for j in range(len(person_ids)):
                        person_id = person_ids[j]
                        quantity = quantities[j]
                        if (np.isnan(quantity)):
                                quantity = 1.0

                        index_p = person_indices[person_id]
                        X_table[index_p][index_f] += quantity

        return X_table

def sort_concept_ids_wrt_ratio_of_frequency_of_positives_and_frequency_of_negatives(df, query_string, query_string_2, person_ids_positives, person_ids_negatives, concept_id_name):

        concept_ids_start = df.drop_duplicates(concept_id_name)[concept_id_name].tolist()

        df_positives_after_covid = df.query(query_string)

        if (df_positives_after_covid.empty):
                return concept_ids_start

        concept_ids_positives_after_covid = df_positives_after_covid[concept_id_name].tolist()

        df_negatives_after_covid = df.query(query_string_2)

        if (df_negatives_after_covid.empty):
                df_positives_after_covid['counts_positives'] = df_positives_after_covid.groupby(concept_id_name)[concept_id_name].transform('count')
                df_positives_after_covid = df_positives_after_covid.drop_duplicates(concept_id_name).sort_values([concept_id_name])
                positives_after_covid_counts = df_positives_after_covid['counts_positives'].to_numpy()
                concept_ids_after_covid = df_positives_after_covid[concept_id_name].to_numpy()
                ratios = positives_after_covid_counts
                ratios_sorted, concept_ids_sorted = zip( *sorted( zip(ratios, concept_ids_after_covid) ) )
                ratios_sorted = ratios_sorted[::-1]
                concept_ids_sorted = concept_ids_sorted[::-1]
                return concept_ids_sorted

        df_positives_after_covid['counts_positives'] = df_positives_after_covid.groupby(concept_id_name)[concept_id_name].transform('count')
        df_negatives_after_covid['counts_negatives'] = df_negatives_after_covid.groupby(concept_id_name)[concept_id_name].transform('count')

        df_positives_after_covid = df_positives_after_covid.drop_duplicates(concept_id_name).sort_values([concept_id_name])
        df_negatives_after_covid = df_negatives_after_covid.drop_duplicates(concept_id_name).sort_values([concept_id_name])

        df_after_covid = pd.merge(df_positives_after_covid, df_negatives_after_covid, how='left', on = concept_id_name)

        positives_after_covid_counts = df_after_covid['counts_positives'].to_numpy()
        negatives_after_covid_counts = df_after_covid['counts_negatives'].to_numpy()

        positives_after_covid_counts = np.nan_to_num(positives_after_covid_counts)
        negatives_after_covid_counts = np.nan_to_num(negatives_after_covid_counts)

        concept_ids_after_covid = df_positives_after_covid[concept_id_name].to_numpy()
        ratios = np.true_divide(positives_after_covid_counts, (1+negatives_after_covid_counts))

        ratios_sorted, concept_ids_sorted = zip( *sorted( zip(ratios, concept_ids_after_covid) ) )
        ratios_sorted = ratios_sorted[::-1]
        concept_ids_sorted = concept_ids_sorted[::-1]

        return concept_ids_sorted

def sort_concept_ids_wrt_ratio_of_frequency_of_positives_and_frequency_of_negatives_2(df, query_string, query_string_2, person_ids_positives, person_ids_negatives, concept_id_name):

        df = df.drop_duplicates('person_id')
	
	concept_ids_start = df.drop_duplicates(concept_id_name)[concept_id_name].tolist()

        df_positives_after_covid = df.query(query_string)

	if (df_positives_after_covid.empty):
		return concept_ids_start
	
        concept_ids_positives_after_covid = df_positives_after_covid[concept_id_name].tolist()

        df_negatives_after_covid = df.query(query_string_2)

	if (df_negatives_after_covid.empty):
		df_positives_after_covid['counts_positives'] = df_positives_after_covid.groupby(concept_id_name)[concept_id_name].transform('count')
		df_positives_after_covid = df_positives_after_covid.drop_duplicates(concept_id_name).sort_values([concept_id_name])
		positives_after_covid_counts = df_positives_after_covid['counts_positives'].to_numpy()
		concept_ids_after_covid = df_positives_after_covid[concept_id_name].to_numpy()
		ratios = positives_after_covid_counts
	        ratios_sorted, concept_ids_sorted = zip( *sorted( zip(ratios, concept_ids_after_covid) ) )
        	ratios_sorted = ratios_sorted[::-1]
        	concept_ids_sorted = concept_ids_sorted[::-1]
		return concept_ids_sorted
		
        df_positives_after_covid['counts_positives'] = df_positives_after_covid.groupby(concept_id_name)[concept_id_name].transform('count')
        df_negatives_after_covid['counts_negatives'] = df_negatives_after_covid.groupby(concept_id_name)[concept_id_name].transform('count')

        df_positives_after_covid = df_positives_after_covid.drop_duplicates(concept_id_name).sort_values([concept_id_name])
        df_negatives_after_covid = df_negatives_after_covid.drop_duplicates(concept_id_name).sort_values([concept_id_name])

        df_after_covid = pd.merge(df_positives_after_covid, df_negatives_after_covid, how='left', on = concept_id_name)

        positives_after_covid_counts = df_after_covid['counts_positives'].to_numpy()
        negatives_after_covid_counts = df_after_covid['counts_negatives'].to_numpy()

	positives_after_covid_counts = np.nan_to_num(positives_after_covid_counts)
	negatives_after_covid_counts = np.nan_to_num(negatives_after_covid_counts)

        concept_ids_after_covid = df_positives_after_covid[concept_id_name].to_numpy()
        ratios = np.true_divide(positives_after_covid_counts, (1+negatives_after_covid_counts))

        ratios_sorted, concept_ids_sorted = zip( *sorted( zip(ratios, concept_ids_after_covid) ) )
        ratios_sorted = ratios_sorted[::-1]
        concept_ids_sorted = concept_ids_sorted[::-1]

        return concept_ids_sorted

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
person_ids_negatives = np.array(gs.loc[gs['status'] == 0.0, 'person_id'])

print(datetime.now())
print("Load measurement.csv")

csv_filename = '../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/measurement.csv'
usecols_list = ['measurement_concept_id','value_as_number','person_id', 'measurement_date', 'measurement_time', 'value_as_concept_id']

dropna_subset = ['measurement_concept_id']
concept_id_name = 'measurement_concept_id'
fillna_field = 'value_as_number'
fillna_value = 0.0
fillna_field_2 = 'measurement_date'
fillna_value_2 = '1900-01-01'
measurement = read_data_frame(csv_filename, usecols_list, dropna_subset, concept_id_name, fillna_field, fillna_value, fillna_field_2, fillna_value_2)
measurement_rt_pcr = measurement.loc[(measurement['value_as_concept_id']==45877985.0) | (measurement['value_as_concept_id']==45884084.0)]

print(measurement.shape)

print(datetime.now())
print("Sort measurement concept ids")

concept_id_name = 'measurement_concept_id'
query_string = 'measurement_date > "2020-01-01" and person_id in @person_ids_positives and value_as_number != 0.0'
query_string_2 = 'measurement_date > "2020-01-01" and person_id in @person_ids_negatives and measurement_concept_id in @concept_ids_positives_after_covid and value_as_number != 0.0'
measurement_concept_ids_sorted_wrt_frequency = sort_concept_ids_wrt_ratio_of_frequency_of_positives_and_frequency_of_negatives_2(measurement, query_string, query_string_2, person_ids_positives, person_ids_negatives, concept_id_name)

print(datetime.now())

print("Load condition.csv")
csv_filename = '../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/condition_occurrence.csv'
usecols_list = ['condition_concept_id', 'person_id', 'condition_start_date', 'condition_end_date']
dropna_subset = ['condition_concept_id']
concept_id_name = 'condition_concept_id'
fillna_field = 'condition_start_date'
fillna_value = '1900-01-01'
fillna_field_2 = 'condition_end_date'
fillna_value_2 = '2100-01-01'
condition = read_data_frame(csv_filename, usecols_list, dropna_subset, concept_id_name, fillna_field, fillna_value, fillna_field_2, fillna_value_2)

print(datetime.now())

print("Sort condition concept ids")

query_string = 'condition_end_date > "2020-01-01" and person_id in @person_ids_positives'
concept_id_name = 'condition_concept_id'
query_string_2 = 'condition_end_date > "2020-01-01" and person_id in @person_ids_negatives and condition_concept_id in @concept_ids_positives_after_covid'
condition_concept_ids_sorted_wrt_frequency = sort_concept_ids_wrt_ratio_of_frequency_of_positives_and_frequency_of_negatives_2(condition, query_string, query_string_2, person_ids_positives, person_ids_negatives, concept_id_name)

print(datetime.now())

print("Load observation.csv")

csv_filename = '../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/observation.csv'
usecols_list = ['observation_concept_id', 'person_id', 'observation_date']
dropna_subset = ['observation_concept_id']
concept_id_name = 'observation_concept_id'
fillna_field = 'observation_date'
fillna_value = '1900-01-01'
observation = read_data_frame(csv_filename, usecols_list, dropna_subset, concept_id_name, fillna_field, fillna_value, fillna_field_2, fillna_value_2)

print(datetime.now())

print("Sort observation concept ids")

query_string = 'person_id in @person_ids_positives'
concept_id_name = 'observation_concept_id'
query_string_2 = 'person_id in @person_ids_negatives and observation_concept_id in @concept_ids_positives_after_covid'
observation_concept_ids_sorted_wrt_frequency = sort_concept_ids_wrt_ratio_of_frequency_of_positives_and_frequency_of_negatives_2(observation, query_string, query_string_2, person_ids_positives, person_ids_negatives, concept_id_name)

print(datetime.now())

print("Load device_exposure.csv")

csv_filename = '../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/device_exposure.csv'
usecols_list = ['device_concept_id', 'person_id', 'device_exposure_start_date', 'device_exposure_end_date']
dropna_subset = ['device_concept_id']
concept_id_name = 'device_concept_id'
fillna_field = 'device_exposure_start_date'
fillna_value = '1900-01-01'
fillna_field_2 = 'device_exposure_end_date'
fillna_value_2 = '2100-01-01'
device_exposure = read_data_frame(csv_filename, usecols_list, dropna_subset, concept_id_name, fillna_field, fillna_value, fillna_field_2, fillna_value_2)

print(datetime.now())

print("Sort device_exposure concept ids")

query_string = 'device_exposure_end_date > "2020-01-01" and person_id in @person_ids_positives'
concept_id_name = 'device_concept_id'
query_string_2 = 'device_exposure_end_date > "2020-01-01" and person_id in @person_ids_negatives and device_concept_id in @concept_ids_positives_after_covid'
device_concept_ids_sorted_wrt_frequency = sort_concept_ids_wrt_ratio_of_frequency_of_positives_and_frequency_of_negatives_2(device_exposure, query_string, query_string_2, person_ids_positives, person_ids_negatives, concept_id_name)

print(datetime.now())

print("Load drug_exposure.csv")

csv_filename = '../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/drug_exposure.csv'
usecols_list = ['drug_concept_id', 'person_id', 'drug_exposure_start_date', 'drug_exposure_end_date', 'quantity']
dropna_subset = ['drug_concept_id']
concept_id_name = 'drug_concept_id'
fillna_field = 'drug_exposure_start_date'
fillna_value = '1900-01-01'
fillna_field_2 = 'drug_exposure_end_date'
fillna_value_2 = '2100-01-01'
drug_exposure = read_data_frame(csv_filename, usecols_list, dropna_subset, concept_id_name, fillna_field, fillna_value, fillna_field_2, fillna_value_2)

print(datetime.now())

print("Sort drug_exposure concept ids")

query_string = 'drug_exposure_end_date > "2020-01-01" and person_id in @person_ids_positives'
concept_id_name = 'drug_concept_id'
query_string_2 = 'drug_exposure_end_date > "2020-01-01" and person_id in @person_ids_negatives and drug_concept_id in @concept_ids_positives_after_covid'
drug_concept_ids_sorted_wrt_frequency = sort_concept_ids_wrt_ratio_of_frequency_of_positives_and_frequency_of_negatives_2(drug_exposure, query_string, query_string_2, person_ids_positives, person_ids_negatives, concept_id_name)

print(datetime.now())

print("Load procedure_occurrence.csv")

csv_filename = '../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/procedure_occurrence.csv'
usecols_list = ['procedure_concept_id', 'person_id', 'procedure_date']
dropna_subset = ['procedure_concept_id']
concept_id_name = 'procedure_concept_id'
fillna_field = 'procedure_date'
fillna_value = '1900-01-01'
procedure = read_data_frame(csv_filename, usecols_list, dropna_subset, concept_id_name, fillna_field, fillna_value, fillna_field_2, fillna_value_2)

print("Sort procedure concept ids")

query_string = 'procedure_date > "2020-01-01" and person_id in @person_ids_positives'
concept_id_name = 'procedure_concept_id'
query_string_2 = 'procedure_date > "2020-01-01" and person_id in @person_ids_negatives and procedure_concept_id in @concept_ids_positives_after_covid'
procedure_concept_ids_sorted_wrt_frequency = sort_concept_ids_wrt_ratio_of_frequency_of_positives_and_frequency_of_negatives_2(procedure, query_string, query_string_2, person_ids_positives, person_ids_negatives, concept_id_name)

print("Load visit_occurrence.csv")

csv_filename = '../../ehr-dream-challenges/examples/covid19-question-2/synthetic_data/training/visit_occurrence.csv'
usecols_list = ['visit_concept_id', 'person_id', 'visit_start_date', 'visit_end_date']
dropna_subset = ['visit_concept_id']
concept_id_name = 'visit_concept_id'
fillna_field = 'visit_start_date'
fillna_value = '1900-01-01'
fillna_field_2 = 'visit_end_date'
fillna_value_2 = '2100-01-01'
visit = read_data_frame(csv_filename, usecols_list, dropna_subset, concept_id_name, fillna_field, fillna_value, fillna_field_2, fillna_value_2)

print("Sort visit concept ids")

query_string = 'visit_end_date > "2020-01-01" and person_id in @person_ids_positives'
concept_id_name = 'visit_concept_id'
query_string_2 = 'visit_end_date > "2020-01-01" and person_id in @person_ids_negatives and visit_concept_id in @concept_ids_positives_after_covid'
visit_concept_ids_sorted_wrt_frequency = sort_concept_ids_wrt_ratio_of_frequency_of_positives_and_frequency_of_negatives_2(visit, query_string, query_string_2, person_ids_positives, person_ids_negatives, concept_id_name)

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

X_n_tests = np.zeros((len(person_indices), 1))

covid_measurement = add_COVID_measurement_date()
covid_measurement['counts'] = covid_measurement.groupby('person_id')['person_id'].transform('count')
covid_measurement = covid_measurement.drop_duplicates('person_id')
n_tests_given_person = dict(zip(covid_measurement.person_id, covid_measurement.counts))

#measurement_positives_after_covid = measurement.query('measurement_date > "2019-11-17" and person_id in @person_ids_positives')

#measurement_positives_after_covid['counts'] = measurement_positives_after_covid.groupby('measurement_concept_id')['measurement_concept_id'].transform('count')
#measurement_positives_after_covid.sort_values('counts', inplace=True, ascending=False)
##measurement_positives_after_covid = measurement_positives_after_covid.query('counts > 15')
#measurement_positives_after_covid = measurement_positives_after_covid.drop_duplicates('measurement_concept_id')
#measurement_concept_ids_sorted_wrt_frequency = measurement_positives_after_covid['measurement_concept_id'].tolist()

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
query_str = 'condition_end_date > "2020-01-01" and person_id == @person_id'
#query_str = 'condition_end_date'
#query_str_2 = 'person_id == @person_id'
concept_id_str = 'condition_concept_id'
feature_indices = condition_feature_indices
X = X_condition 
#X_condition = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, query_str_2, concept_id_str, feature_indices, X)
X_condition = prepare_feature_matrix_count_based_2(person_ids, person_indices, df, query_str, concept_id_str, feature_indices, X)

print(datetime.now())
print("observations")

df = observation
#query_str = 'observation_date > "2010-01-01" and person_id == @person_id'
query_str = 'person_id == @person_id'
#query_str = 'observation_date'
#query_str_2 = 'person_id == @person_id'
concept_id_str = 'observation_concept_id'
feature_indices = observation_feature_indices
X = X_observation
#X_observation = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, query_str_2, concept_id_str, feature_indices, X)
X_observation = prepare_feature_matrix_count_based_2(person_ids, person_indices, df, query_str, concept_id_str, feature_indices, X)

print(datetime.now())
print("device exposures")

df = device_exposure
query_str = 'device_exposure_end_date > "2020-01-01" and person_id == @person_id'
#query_str = 'device_exposure_end_date' 
#query_str_2 = 'person_id == @person_id'
concept_id_str = 'device_concept_id'
feature_indices = device_exposure_feature_indices
X = X_device_exposure
#X_device_exposure = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, query_str_2, concept_id_str, feature_indices, X)
X_device_exposure = prepare_feature_matrix_count_based_2(person_ids, person_indices, df, query_str, concept_id_str, feature_indices, X)

print(datetime.now())
print("drug exposures")

df = drug_exposure
#query_str = 'drug_exposure_end_date > "2020-01-01" and person_id == @person_id'
#query_str = 'drug_exposure_end_date'
#query_str_2 = 'person_id == @person_id'
#concept_id_str = 'drug_concept_id'
#feature_indices = drug_exposure_feature_indices
#X = X_drug_exposure
#X_drug_exposure = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, query_str_2, concept_id_str, feature_indices, X)
#X_drug_exposure = prepare_feature_matrix_count_based_2(person_ids, person_indices, df, query_str, concept_id_str, feature_indices, X)

query_string = 'drug_exposure_end_date > "2020-01-01" and drug_concept_id in @i'
X_drug_exposure = prepare_feature_matrix_drug(df, drug_exposure_feature_indices, query_string, X_drug_exposure)

print(datetime.now())
print("procedures")

df = procedure
query_str = 'procedure_date > "2020-01-01" and person_id == @person_id'
#query_str = 'procedure_date'
#query_str_2 = 'person_id == @person_id'
concept_id_str = 'procedure_concept_id'
feature_indices = procedure_feature_indices
X = X_procedure
#X_procedure = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, query_str_2, concept_id_str, feature_indices, X)
X_procedure = prepare_feature_matrix_count_based_2(person_ids, person_indices, df, query_str, concept_id_str, feature_indices, X)

print(datetime.now())
print("visits")

df = visit
query_str = 'visit_end_date > "2020-01-01" and person_id == @person_id'
#query_str = 'visit_end_date'
#query_str_2 = 'person_id == @person_id'
concept_id_str = 'visit_concept_id'
feature_indices = visit_feature_indices
X = X_visit
#X_visit = prepare_feature_matrix_count_based(person_ids, person_indices, covid_measurement, df, query_str, query_str_2, concept_id_str, feature_indices, X)
X_visit = prepare_feature_matrix_count_based_2(person_ids, person_indices, df, query_str, concept_id_str, feature_indices, X)

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
print("number of tests")

for i in range(n_persons):

	person_id = person_ids[i]
	index_p = person_indices[person_id]
	index_f = 0
	if (person_id not in n_tests_given_person):
		continue
	X_n_tests[index_p, index_f] = n_tests_given_person[person_id]

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

print("concatenation of feature matrices")

#X_selected = np.concatenate((X_measurement_selected, X_age), axis=1)
#X_selected = np.concatenate((X_selected, X_measurement_rt_pcr_selected), axis=1)
#X_selected = np.concatenate((X_selected, X_condition_selected), axis=1)
#X_selected = np.concatenate((X_selected, X_observation_selected), axis=1)
#X_selected = np.concatenate((X_selected, X_device_exposure_selected), axis=1)
#X_selected = np.concatenate((X_selected, X_drug_exposure_selected), axis=1)
#X_selected = np.concatenate((X_selected, X_procedure_selected), axis=1)
#X_selected = np.concatenate((X_selected, X_visit_selected), axis=1)
##X_selected = np.concatenate((X_selected, X_gender_race_ethnicity), axis=1)

X_selected = np.concatenate((X_measurement_selected, X_condition_selected), axis=1)
X_selected = np.concatenate((X_selected, X_observation_selected), axis=1)
X_selected = np.concatenate((X_selected, X_drug_exposure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_device_exposure_selected), axis=1)
X_selected = np.concatenate((X_selected, X_procedure), axis=1)
X_selected = np.concatenate((X_selected, X_visit), axis=1)
X_selected = np.concatenate((X_selected, X_age), axis=1)
#X_selected = X_age
X_selected = np.concatenate((X_selected, X_n_tests), axis=1)

selector = SelectFromModel(estimator=XGBClassifier(n_jobs=4)).fit(X_selected, Y)
#selector = SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=50)).fit(X_selected, Y)
X_selected = selector.transform(X_selected)

#transformer = RobustScaler().fit(X_selected)
transformer = MinMaxScaler().fit(X_selected)
X_selected = transformer.transform(X_selected)

X_selected = np.concatenate((X_selected, X_gender_race_ethnicity), axis=1)

selected_feature_indices = selector.get_support(indices=True)
scalerfile = '../../model/feature_selector_stage_2.sav'
dump(selector, open(scalerfile, 'wb'))

print(datetime.now())
print("X_selected.shape")
print(X_selected.shape)

print(datetime.now())
print("Model training")

clf = LogisticRegression(solver='saga', n_jobs=-1).fit(X_selected, Y)
#clf = MLPClassifier(learning_rate_init=0.01).fit(X_selected, Y)
#clf = lgb.LGBMClassifier(n_jobs = 4).fit(X_selected, Y)

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

