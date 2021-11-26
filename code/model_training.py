import sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from joblib import dump

def read_dataset_from_arff_file(dataset_filename):

        dataset_file = open(dataset_filename, 'r')

        n_features = 0
        for line in dataset_file:
                if ('@data' in line):
                        break
                if ('@attribute' in line):
                        n_features += 1

        n_features -= 1

        X = np.zeros((0,n_features))
        y = np.zeros((0,))

        for line in dataset_file:

                line = line.rstrip()
                token_list = line.split(',')

		if (len(token_list) == 0) or (token_list[0] == ''):
			continue
		
		class_label = int(token_list[-1])
		y_sample = np.zeros((1,))
                y_sample[0] = class_label
               	y = np.concatenate((y, y_sample), axis=0)

                if (',' not in line):
                        continue
 
                feature_set = np.array(list(map(float, token_list[:-1]))).reshape(1,n_features)
                X = np.concatenate((X, feature_set), axis=0)

        dataset_file.close()

        return [X, y]
        
if __name__ == "__main__":

        selected_feature_indices_filename = sys.argv[1]
        features_wo_gender_race_ethnicity_filename = sys.argv[2]
        features_gender_race_ethnicity_filename = sys.argv[3]
	class_labels_filename = sys.argv[4]
        trained_model_filename = sys.argv[5]

	selected_feature_indices_file = open(selected_feature_indices_filename, "r")
	selected_feature_indices_string = selected_feature_indices_file.readlines()
	selected_feature_indices = [int(i) for i in selected_feature_indices_string]

        features_wo_gender_race_ethnicity_file = open(features_wo_gender_race_ethnicity_filename, 'rb')
	features_gender_race_ethnicity_file = open(features_gender_race_ethnicity_filename, 'rb')
	class_labels_file = open(class_labels_filename, 'rb')

        X_wo_gender_race_ethnicity = np.load(features_wo_gender_race_ethnicity_file)
        X_gender_race_ethnicity = np.load(features_gender_race_ethnicity_file)
	y = np.load(class_labels_file)

	if (len(selected_feature_indices) > 0):
		X_selected = X_wo_gender_race_ethnicity[:, np.array(selected_feature_indices)]
		X_selected = np.concatenate((X_selected, X_gender_race_ethnicity), axis=1)
	else:
		X_selected = X_gender_race_ethnicity

        clf = MLPClassifier(learning_rate_init=0.01).fit(X_selected, y)
        dump(clf, trained_model_filename)        
