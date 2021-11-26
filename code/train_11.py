#performs 10-fold CV using logistic regression with 44 numeric features (measurement, condition, age)

import datetime as dt
import numpy as np
from datetime import date
import pandas as pd
import sklearn
#from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from joblib import dump
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import sys
from datetime import datetime
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

print(datetime.now())
print("Load measurement.csv")

measurement = pd.read_csv('../data/release_07-06-2020/training/measurement.csv',usecols = ['measurement_concept_id','value_as_number','person_id', 'measurement_date', 'measurement_time'])

measurement_feature_concept_ids = ['3020891', '3027018', '3012888', '3004249', '3023314', '3013650', '3004327', '3016502', '3010156', '3023091', '3024929', '3000963', '3024171', '3022250', '42870366', '3021337', '3000905', '3015242', '3016723', '3007220', '3011960', '3006923', '3013721', '3027801', '3016407', '3004501', '3022192', '3028167', '3024128', '3027114', '3019897', '3013682', '3013707', '3023548', '3046279', '3018405', '3008037', '3005491', '3003396', '3018677', '3027946', '3014576', '3020716', '3033891', '3012544', '3023103', '3019550', '3024561', '3045716', '3008152', '3019977', '40765161', '3041623', '3012158', '3006262', '3025023', '3042596', '3042194', '3038288', '3003694', '3044254', '3038297', '3044938', '3025634', '3025315']

condition_feature_concept_ids=['78232', ['4170554', '194133', '372409', '134736', '4001450', '312998'], '75909', '78508', '254761', '24134', '378253', '437663', ['28060', '259153'], '312437', '4305080', ['4223659', '4216771'], '442752', '257683', '433596', '196523', '75860', '201618', '4103703', '4185711', '43530714', ['4114720', '254669'], ['137809', '380055', '135777', '196360', '439392'], ['4115367', '442980', '4116166', '77635'], '77670', ['4115171', '138525', '73754', '193322', '762297', '4117695', '37109843'], ['4329041', '436096', '4150125', '140821'], '374375', ['372448', '433316'], ['192450', '4245252', '4041285', '197684', '198803'], ['27674', '31967'], ['4041283', '4180628'], '436230', ['315078', '444070'], '4236484', '197925', ['197607', '194696', '196168', '40443308'], '377091', ['4171917', '433595'], '437677', '4310235', '31317', '4155909', ['320136', '257004', '260139', '40480893', '4110056'], ['4079750', '4079749', '80180', '80809'], ['40483287', '193782', '201620', '443611', '443597', '46271022'], ['433736', '434005'], ['316139', '319835', '4229440', '4158911', '315286', '319844'], ['40481919', '4127089', '46273649', '764123', '36712779', '37312532', '46270162', '46270163', '4152384', '4110961', '373503'], ['321319', '320746', '45757557', '4163710'], ['22281', '321263', '443738'], ['42538119', '42539502', '42538117', '42539698'], ['201254', '443412', '201826', '443731', '376065','443729', '37018196', '45770830', '4042728', '443733', '4214376'], ['192680', '312902', '317895', '319826', '320128', '320128', '443771', '4013643', '4110948', '4118910', '4120094', '4167493', '4313767', '4322024', '4339214', '43020910', '44782715', '44783628'], ['192855', '37396808', '4271013'], ['313236', '4138760', '4146581', '4279553', '45768910', '45768963', '45768964', '45768965', '37116845', '45769350', '45769351', '45769352', '45769438', '45771045', '46270082', '46273487'], '381591', ['4110056', '255848'], ['193174', '254320', '441267'], ['440371', '200762', '441269', '4146936', '4223448', '4281109', '45765493', '436659', '45768812', '4146209', '439777', '137820', '257628', '4031164'], ['374888', '378726', '4182210', '43530666'], ['4055224', '4058694', '4058695', '4064161', '4001171', '198964', '439674', '197494'], ['253797', '45763750'], ['30978', '4278669', '4287844'], ['4141360', '313217', '4154290', '4232697'], ['257007', '259848'], ['260123', '257012'], ['378735', '318736', '377844', '381549'], '443454', '321318', '136788', ['435243', '433753'], ['4188598', '4132434'], '438120', '200174', '380094', '378427', '4209423', '80502', ['140168', '81931'], '438485', '201461', '134898', '4142875', ['442588', '436962', '313459', '435524'], ['442077', '4211231', '434613', '438409', '436676'], ['436665', '4286201', '435220', '433758', '4282096', '380378', '433440', '4077577', '440383', '436222'], ['439297', '373478'], '4084966', '440674', '440360', ['440417', '4327889', '40481089'], '26662', '25297', '133141', '141095', '137053', '79864', '435463', '439727', '134460', ['30437', '197381'], '140214', '40405599', '4098604', ['439082', '195321'] ]

observation_feature_concept_ids = ['37208405', '4005823']
device_exposure_feature_concept_ids = ['45768197']

measurement = measurement.dropna(subset = ['measurement_concept_id'])
measurement = measurement.astype({"measurement_concept_id": int})
measurement = measurement.astype({"measurement_concept_id": str})
measurement['value_as_number'] = measurement['value_as_number'].fillna(-10)
measurement['measurement_date'] = measurement['measurement_date'].fillna('1900-01-01')
measurement['measurement_time'] = measurement['measurement_time'].fillna('1900-01-01')

#print(measurement['person_id'][0])
print(measurement.shape)

print(datetime.now())

print("Load condition.csv")
condition = pd.read_csv("../data/release_07-06-2020/training/condition_occurrence.csv",usecols = ['condition_concept_id','person_id', 'condition_start_date', 'condition_end_date'])

condition = condition.dropna(subset = ['condition_concept_id'])
condition = condition.astype({"condition_concept_id": int})
condition = condition.astype({"condition_concept_id": str})
condition['condition_end_date'] = condition['condition_end_date'].fillna('2100-01-01')
condition['condition_start_date'] = condition['condition_start_date'].fillna('1900-01-01')

print(datetime.now())

print("Load observation.csv")
observation = pd.read_csv("../data/release_07-06-2020/training/observation.csv",usecols = ['observation_concept_id','person_id', 'observation_date'])

observation = observation.dropna(subset = ['observation_concept_id'])
observation = observation.astype({"observation_concept_id": int})
observation = observation.astype({"observation_concept_id": str})

print(datetime.now())

print("Load device_exposure.csv")
device_exposure = pd.read_csv("../data/release_07-06-2020/training/device_exposure.csv",usecols = ['device_concept_id','person_id', 'device_exposure_start_date', 'device_exposure_end_date'])

device_exposure = device_exposure.dropna(subset = ['device_concept_id'])
device_exposure = device_exposure.astype({"device_concept_id": int})
device_exposure = device_exposure.astype({"device_concept_id": str})
device_exposure['device_exposure_end_date'] = device_exposure['device_exposure_end_date'].fillna('2100-01-01')
device_exposure['device_exposure_start_date'] = device_exposure['device_exposure_start_date'].fillna('1900-01-01')

print(datetime.now())

print("Load person.csv")

person = pd.read_csv('../data/release_07-06-2020/training/person.csv')
today = date.today().year

'''generate the feature set'''
print(datetime.now())
print("Define feature arrays")

person = person.drop_duplicates(subset = ['person_id'])
person_index = dict(zip(person.person_id, range(len(person.person_id))))
measurement_feature_index = dict(zip(measurement_feature_concept_ids, range(len(measurement_feature_concept_ids))))

condition_feature_index = {}

for i in range(len(condition_feature_concept_ids)):

        condition_array = condition_feature_concept_ids[i]
        for j in range(len(condition_array)):
                condition_concept_id = condition_array[j]
                condition_feature_index[condition_concept_id] = i                

#condition_feature_index = dict(zip(condition_feature_concept_ids, range(len(condition_feature_concept_ids))))
observation_feature_index = dict(zip(observation_feature_concept_ids, range(len(observation_feature_concept_ids))))
device_exposure_feature_index = dict(zip(device_exposure_feature_concept_ids, range(len(device_exposure_feature_concept_ids))))

n_persons = person.shape[0]
n_measurement_types = len(measurement_feature_concept_ids)

X_min = np.zeros((len(person_index), n_measurement_types))
X_max = np.zeros((len(person_index), n_measurement_types))
X_ave = np.zeros((len(person_index), n_measurement_types))
X_min[:] = -10
X_max[:] = -10
X_ave[:] = -10

n_condition_feature_groups = len(condition_feature_concept_ids)
X_condition = np.zeros((len(person_index), n_condition_feature_groups))

X_age = np.zeros((len(person_index), 1))

n_observation_types = len(observation_feature_concept_ids)
X_observation = np.zeros((len(person_index), n_observation_types))

n_device_exposure_types = len(device_exposure_feature_concept_ids)
X_device_exposure = np.zeros((len(person_index), n_device_exposure_types))

print(datetime.now())
person_ids_after_covid_set = set()

print("measurements")

le = preprocessing.LabelEncoder()

for i in measurement_feature_concept_ids:

        #print(i)

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

        #print(subm_after_covid_min)

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
        
        index_f = condition_feature_index[i[0]]

        subm = condition.query('condition_concept_id in @i')
        subm_after_covid = subm.query('condition_end_date > "2019-11-17"')
        #subm_before_covid = subm.query('condition_end_date <= 2019-11-17')

        #condition_person_ids = subm_after_covid.groupby('person_id')['person_id']
        
        for person_id in subm_after_covid.drop_duplicates(subset = ['person_id'])['person_id']:

                index_p = person_index[person_id]
                X_condition[index_p][index_f] = 1.0

print(datetime.now())

print("observations")

for i in observation_feature_concept_ids:

        index_f = observation_feature_index[i]
        subm = observation.query('observation_concept_id in @i')
        #subm_after_covid = subm.query('observation_date > 2019-11-17')

        for person_id in subm.drop_duplicates(subset = ['person_id'])['person_id']:

                index_p = person_index[person_id]
                X_observation[index_p][index_f] = 1.0

print(datetime.now())

print("device exposures")

for i in device_exposure_feature_concept_ids:

        index_f = device_exposure_feature_index[i]
        subm = device_exposure.query('device_concept_id in @i')
        subm_after_covid = subm.query('device_exposure_end_date > "2019-11-17"')
        #subm_before_covid = subm.query('device_exposure_end_date <= 2019-11-17')

        for person_id in subm_after_covid.drop_duplicates(subset = ['person_id'])['person_id']:

                index_p = person_index[person_id]
                X_device_exposure[index_p][index_f] = 1.0

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

X = np.concatenate((X_min, X_max), axis=1)
X = np.concatenate((X, X_ave), axis=1)
X = np.concatenate((X, X_age), axis=1)
X = np.concatenate((X, X_condition), axis=1)
X = np.concatenate((X, X_observation), axis=1)
X = np.concatenate((X, X_device_exposure), axis=1)

print("X.shape")
print(X.shape)

print(datetime.now())

print("true labels")

gs = pd.read_csv('../data/release_07-06-2020/training/goldstandard.csv')
person_status = person.merge(gs, how = 'left', on = ['person_id'])
person_status.drop_duplicates(subset=['person_id'], keep = 'first',inplace = True)
Y =  np.array(person_status[['status']]).ravel()
print("Y.shape")
print(Y.shape)
#clf = LogisticRegressionCV(cv = 10, penalty = 'l2', tol = 0.0001, fit_intercept = True, intercept_scaling = 1, class_weight = None, random_state = None,
#max_iter = 100, verbose = 0, n_jobs = None, scoring='roc_auc').fit(X,Y)
print(datetime.now())

print("feature selection")

#selector = SelectFromModel(estimator=LogisticRegression()).fit(X, Y)
selector = SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=50)).fit(X, Y)

X = selector.transform(X)

print(datetime.now())

print("model training")

clf = LogisticRegression(solver='saga').fit(X,Y)
#clf = RandomForestClassifier()

#******************************************
#NEED TO REMOVE THESE LINES
#y_pred = cross_val_predict(clf, X, Y, cv=3, method='predict_proba')
#print(y_pred.shape)
#precision, recall, thresholds = precision_recall_curve(Y, y_pred[:,1])
#auprc = auc(recall, precision)
#print("AUPRC={:.3f}".format(auprc))

Y = np.reshape(Y, (Y.size, 1))
data_set = np.concatenate((X, Y), axis=1)
df = pd.DataFrame(data_set)
df.to_csv("../data/train_set_extended_features_numeric.csv")
#******************************************

print(datetime.now())

dump(clf, '../model/baseline.joblib')
print("Training stage finished")
#print(clf.score(X,Y))

