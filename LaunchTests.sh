#!/bin/bash

MAXTSTS=1
STAGE=1 # GRIDSEARCH o TRAIN
GPU=0

declare -a CITIES=( "paris") 
declare -a BOWNWORDS=( 10 )

declare -A MODELS
declare -A BATCHES
declare -A LRATES

# MODELS["BOW2RST"]="3"
# BATCHES["BOW2RST"]="64 128 256 512"
# LRATES["BOW2RST"]="5e-6 1e-5 5e-5 1e-4 5e-4"

# MODELS["BOW2VAL"]="3"
# BATCHES["BOW2VAL"]="32 64 128"
# LRATES["BOW2VAL"]="1e-5 5e-5 1e-4 5e-4"

# MODELS["LSTM2VAL"]="3"
# BATCHES["LSTM2VAL"]="32 64 128 256"
# LRATES["LSTM2VAL"]="5e-5 1e-4 5e-4"

# MODELS["LSTM2RST"]="2" 
# BATCHES["LSTM2RST"]="4096" #"2048 256 512 1024"
# LRATES["LSTM2RST"]="5e-4 1e-3 5e-3" 

MODELS["BOW2RST"]="0" 
BATCHES["BOW2RST"]="0"
LRATES["BOW2RST"]="0" 


for CITY in "${CITIES[@]}" ;do
  echo "$CITY"
  for MODEL_NAME in ${!MODELS[@]}; do 
    echo "-$MODEL_NAME"
    for MODEL_VERSION in ${MODELS[$MODEL_NAME]}; do
      echo "--$MODEL_VERSION"
      for BATCH in ${BATCHES[$MODEL_NAME]}; do
        echo "---$BATCH"
        for LRATE in ${LRATES[$MODEL_NAME]}; do
          echo "----$LRATE"

          nohup venv/bin/python3 -u  Main.py -gpu $GPU -stg $STAGE -ct $CITY -mn $MODEL_NAME -mv $MODEL_VERSION -bs $BATCH -lr $LRATE -bownws $BOWNWORDS >> "scripts/out/"$CITY"/"$MODEL_NAME"["$MODEL_VERSION"]_"$BOWNWORDS"_("$BATCH"_"$LRATE").txt" &
        
          GPU=$(($(($GPU+1%2))%2))

          echo "----[$!] $MODEL_NAME"

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
  done
done

# Esperar por los últimos
while [ $(jobs -r | wc -l) -gt 0 ];
do
  sleep 5
done