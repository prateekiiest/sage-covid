#!/bin/bash

java_dir="/usr/bin/"
weka_dir="/vol1/software/weka/3.8.4/"
data_dir="../model/"
models_dir="../model/"

#data_dir="${weka_dir}data/"

dataset_filename="${data_dir}train.arff"
#dataset_filename="${data_dir}breast-cancer.arff"

feature_selected_dataset_filename="${models_dir}train_fs_2.arff"
selected_feature_indices_filename="${models_dir}selected_feature_indices"
gender_race_ethnicity_filename="${models_dir}train_gender_race_ethnicity.npy"
trained_model_filename="${models_dir}baseline.joblib"

#model_filename="${models_dir}weka.model"
#weka_output_filename="${models_dir}weka_fs.out"

date

#python train_43.py
#python /app/feature_extraction.py

echo "Feature selection"

#${java_dir}java -classpath ${weka_dir}weka.jar weka.filters.supervised.attribute.AttributeSelection -E "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.functions.Logistic -F 3 -T 0.01 -R 1 -E AUPRC -- -R 1.0E-8 -M -1 -num-decimal-places 4" -S "weka.attributeSelection.BestFirst -D 1 -N 5" -i $dataset_filename -o $feature_selected_dataset_filename

#${java_dir}java -classpath ${weka_dir}weka.jar weka.filters.supervised.attribute.AttributeSelection -E "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.functions.Logistic -F 3 -T 0.01 -R 1 -E AUPRC -- -R 1.0E-8 -M -1 -num-decimal-places 4" -S "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1" -i $dataset_filename -o $feature_selected_dataset_filename

${java_dir}java -classpath ${weka_dir}weka.jar weka.filters.supervised.attribute.AttributeSelection -E "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.bayes.NaiveBayes -F 3 -T 0.01 -R 1 -E AUPRC --" -S "weka.attributeSelection.BestFirst -D 1 -N 5" -i $dataset_filename -o $feature_selected_dataset_filename

#${java_dir}java -classpath ${weka_dir}weka.jar weka.filters.supervised.attribute.AttributeSelection -E "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.bayes.NaiveBayes -F 2 -T 0.01 -R 1 -E AUPRC --" -S "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1" -i $dataset_filename -o $feature_selected_dataset_filename


date

grep "@attribute" $feature_selected_dataset_filename | cut -f 2 -d ' ' | head -n -1 > $selected_feature_indices_filename

python model_training.py $feature_selected_dataset_filename $gender_race_ethnicity_filename $trained_model_filename

date
