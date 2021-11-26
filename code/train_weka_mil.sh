#!/bin/bash

java_dir="/usr/bin/"
weka_dir="/vol1/software/weka/3.8.4/"
data_dir="../model/"
models_dir="../model/"

dataset_filename="${data_dir}measurements_train.arff"
#dataset_filename="${data_dir}measurements_train_2.arff"
model_filename="${models_dir}weka_tlc.model"
weka_output_filename="${models_dir}weka_tlc_train.out"
#n_cv_folds=10

#python train_43.py
#python /app/train.py

${java_dir}java -classpath ${weka_dir}weka.jar:${weka_dir}multiInstanceFilters.jar:${weka_dir}multiInstanceLearning.jar weka.classifiers.mi.TLC -t $dataset_filename -d $model_filename -P "weka.classifiers.trees.J48 -C 0.25 -M 2" -W weka.classifiers.meta.LogitBoost -- -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump > $weka_output_filename

