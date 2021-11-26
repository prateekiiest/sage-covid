import sys

data_filename = "../data/release_07-06-2020/training/measurement_concept_ids_sorted.dat"

feature_concept_ids = ['3020891', '3027018', '3012888', '3004249', '3023314', '3013650', '3004327', '3016502', '3010156', '3023091', '3024929', '3000963', '3024171', '3022250', '42870366', '3021337', '3000905', '3015242', '3016723', '3007220', '3011960', '3006923', '3013721', '3027801', '3016407', '3004501', '3022192', '3028167', '3024128', '3027114', '3019897', '3013682', '3013707', '3023548', '3046279', '3018405', '3008037', '3005491', '3003396', '3018677', '3027946', '3014576', '3020716', '3033891', '3012544', '3023103', '3019550', '3024561', '3045716', '3008152', '3019977', '3025315']

data_file = open(data_filename, 'r')
data_file_inside = data_file.readlines()
data_file.close()
n_lines = len(data_file_inside)

n_features = len(feature_concept_ids)
feature_indices = dict(zip(feature_concept_ids, range(len(feature_concept_ids))))
feature_weights = [0]*n_features

total_frequency = 0
for i in range(n_lines):

	line = data_file_inside[i].rstrip()
	token_list = line.split()
	frequency = int(token_list[0])
	#concept_id = token_list[1]
	total_frequency += frequency

for i in range(n_lines):

        line = data_file_inside[i].rstrip()
        token_list = line.split()
        frequency = int(token_list[0])
        concept_id = token_list[1]
	if (concept_id not in feature_concept_ids):
		continue

	weight = float(frequency) / total_frequency
	feature_index = feature_indices[concept_id]
	feature_weights[feature_index] = weight

print(feature_weights)

