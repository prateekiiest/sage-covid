import sys

dataset_filename = "../data/train_set_52_rule_based_measurement_features_time_constraint.csv"
dataset_file = open(dataset_filename, 'r')
dataset_file_inside = dataset_file.readlines()
dataset_file.close()
n_lines = len(dataset_file_inside)

output_filename = "../data/n_missing_values_per_person.dat"

output_file = open(output_filename, 'w')

n_missing_values_array = []

for i in range(n_lines):

	if (i == 0):
		continue

	line = dataset_file_inside[i].rstrip()

	n_missing_values_person = 0
	token_list = line.split(',')
	for j in range(len(token_list)-1):
		value = float(token_list[j])
		if (value == -1000.0):
			n_missing_values_person += 1

	n_missing_values_array += [n_missing_values_person]


n_missing_values_array.sort()

for i in range(len(n_missing_values_array)):
	
	output_file.write("%d\n" % n_missing_values_array[i])

output_file.close()		

