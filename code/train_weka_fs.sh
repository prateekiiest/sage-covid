#!/bin/bash

java_dir="/usr/bin/"
weka_dir="/vol1/software/weka/3.8.4/"
data_dir="../model/"
models_dir="../model/"

#data_dir="${weka_dir}data/"

dataset_reduced_filename="${data_dir}train_opt.arff"
#dataset_reduced_filename="${data_dir}breast-cancer.arff"

feature_selected_dataset_reduced_filename="${models_dir}train_opt_fs.arff"
selected_feature_indices_filename="${models_dir}selected_feature_indices"
features_wo_gender_race_ethnicity_filename="${models_dir}train_wo_gender_race_ethnicity.npy"
features_gender_race_ethnicity_filename="${models_dir}train_gender_race_ethnicity.npy"
class_labels_filename="${models_dir}class_labels.npy"

trained_model_filename="${models_dir}baseline.joblib"

#model_filename="${models_dir}weka.model"
#weka_output_filename="${models_dir}weka_fs.out"

date

#python train_43.py
#python /app/feature_extraction.py

echo "Feature selection"

#${java_dir}java -classpath ${weka_dir}weka.jar weka.filters.supervised.attribute.AttributeSelection -E "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.functions.Logistic -F 3 -T 100 -R 1 -E AUPRC -- -R 1.0E-8 -M -1 -num-decimal-places 4" -S "weka.attributeSelection.BestFirst -D 2 -N 5" -i $dataset_reduced_filename -o $feature_selected_dataset_reduced_filename

#${java_dir}java -classpath ${weka_dir}weka.jar weka.filters.supervised.attribute.AttributeSelection -E "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.functions.Logistic -F 3 -T 0.01 -R 1 -E AUPRC -- -R 1.0E-8 -M -1 -num-decimal-places 4" -S "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1" -i $dataset_reduced_filename -o $feature_selected_dataset_reduced_filename

#${java_dir}java -classpath ${weka_dir}weka.jar weka.filters.supervised.attribute.AttributeSelection -E "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.bayes.NaiveBayes -F 3 -T 0.01 -R 1 -E AUPRC --" -S "weka.attributeSelection.BestFirst -D 2 -N 5" -i $dataset_reduced_filename -o $feature_selected_dataset_reduced_filename

#${java_dir}java -classpath ${weka_dir}weka.jar weka.filters.supervised.attribute.AttributeSelection -E "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.bayes.NaiveBayes -F 2 -T 100 -R 1 -E AUPRC --" -S "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1" -i $dataset_reduced_filename -o $feature_selected_dataset_reduced_filename

#${java_dir}java -classpath ${weka_dir}weka.jar weka.filters.supervised.attribute.AttributeSelection -E "weka.attributeSelection.CfsSubsetEval -P 1 -E 1" -S "weka.attributeSelection.BestFirst -D 2 -N 5" -i $dataset_reduced_filename -o $feature_selected_dataset_reduced_filename

#${java_dir}java -classpath ${weka_dir}weka.jar weka.filters.supervised.attribute.AttributeSelection -E "weka.attributeSelection.ClassifierSubsetEval -B weka.classifiers.functions.Logistic -T -H \"Click to set hold out or test instances\" -E AUPRC -- -R 1.0E-8 -M -1 -num-decimal-places 4" -S "weka.attributeSelection.BestFirst -D 2 -N 5" -i $dataset_reduced_filename -o $feature_selected_dataset_reduced_filename

#${java_dir}java -classpath ${weka_dir}weka.jar weka.filters.supervised.attribute.AttributeSelection -E "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.trees.J48 -F 3 -T 0.01 -R 1 -E AUPRC -- -C 0.25 -M 2" -S "weka.attributeSelection.BestFirst -D 2 -N 5" -i $dataset_reduced_filename -o $feature_selected_dataset_reduced_filename

${java_dir}java -classpath ${weka_dir}weka.jar weka.filters.supervised.attribute.AttributeSelection -E "weka.attributeSelection.CfsSubsetEval -P 4 -E 4" -S "weka.attributeSelection.BestFirst -D 2 -N 5" -i $dataset_reduced_filename -o $feature_selected_dataset_reduced_filename

date

grep "@attribute" $feature_selected_dataset_reduced_filename | cut -f 2 -d ' ' | head -n -1 > $selected_feature_indices_filename

python model_training.py $selected_feature_indices_filename $features_wo_gender_race_ethnicity_filename $features_gender_race_ethnicity_filename $class_labels_filename $trained_model_filename

date
