#!/bin/bash

java_dir="/usr/bin/"
weka_dir="/vol1/software/weka/3.8.4/"
data_dir="../model/"
models_dir="../model/"
outputs_dir="../output/"

person_ids_filename="${models_dir}person_ids.list"
dataset_filename="${data_dir}measurements_infer.arff"
model_filename="${models_dir}weka_tlc.model"
weka_output_filename="${models_dir}weka_tlc_infer.out"
prediction_outputs_filename="${outputs_dir}predictions.csv"

python infer_43.py
#python /app/infer.py

${java_dir}java -classpath ${weka_dir}weka.jar:${weka_dir}multiInstanceFilters.jar:${weka_dir}multiInstanceLearning.jar weka.classifiers.mi.TLC -l $model_filename -T $dataset_filename -p 0 -distribution > $weka_output_filename

python convert_weka_output_to_challenge_format.py $weka_output_filename $person_ids_filename $prediction_outputs_filename

#python /app/convert_weka_output_to_challenge_format.py $weka_output_filename $person_ids_filename $prediction_outputs_filename

