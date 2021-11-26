import sys

if __name__ == "__main__":

	weka_predictions_filename = sys.argv[1]
	person_ids_filename = sys.argv[2]
	output_filename = sys.argv[3]

	person_ids_file = open(person_ids_filename, 'r')
	person_ids_file_inside = person_ids_file.readlines()
	person_ids_file.close()

	weka_predictions_file = open(weka_predictions_filename, 'r')
	
	output_file = open(output_filename, 'w')
	output_file.write('person_id,score\n')

	person_index = 0
	for line in weka_predictions_file:

		if (',' not in line):
			continue

		person_id = person_ids_file_inside[person_index].rstrip()
		token_list = line.split(',')
		prediction_score = float(token_list[-1].replace('*', ''))
		output_file.write('{},{:.17f}\n'.format(person_id, prediction_score))
		person_index += 1

	weka_predictions_file.close()
	output_file.close()

