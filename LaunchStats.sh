#!/bin/bash

MAXTSTS=1
GPU=0

declare -A DATASETS
declare -A MODELS

DATASETS["restaurants"]="gijon barcelona madrid newyorkcity paris"
DATASETS["pois"]="barcelona madrid newyorkcity paris london"
DATASETS["amazon"]="digital_music fashion"

MODELS["BOW2ITM"]="" 
MODELS["ATT2ITM"]="" 
MODELS["USEM2ITM"]="" 

for DATASET_NAME in ${!DATASETS[@]}; do 
  for SUBSET_NAME in ${DATASETS[$DATASET_NAME]}; do
    echo "[$DATASET_NAME] -> $SUBSET_NAME"
    for MODEL_NAME in ${!MODELS[@]}; do 
      echo "  ╚═ $MODEL_NAME"

      nohup /media/nas/pperez/miniconda3/envs/TAVtext/bin/python -u ModelStats.py -mn $MODEL_NAME -dst $DATASET_NAME -sst $SUBSET_NAME &

      # Si se alcanza el máximo de procesos simultaneos, esperar
      while [ $(jobs -r | wc -l) -eq $MAXTSTS ];
      do
        sleep 5
      done

      #Esperar X segundos entre pruebas para que le de tiempo a ocupar memoria en GPU
      sleep 10

    done  
  done
done


# Esperar por los últimos
while [ $(jobs -r | wc -l) -gt 0 ];
do
  sleep 5
done

