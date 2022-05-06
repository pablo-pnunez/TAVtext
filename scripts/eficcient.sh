#!/bin/bash

MAXTSTS=4
GPU=0

declare -a BATCHES=( 128 256 512 )
declare -a LRATES=( 5e-4 1e-3 5e-3 )

 for BATCH in "${BATCHES[@]}" ;do
    echo "-$BATCH"
    for LRATE in "${LRATES[@]}" ;do
      echo "--$LRATE"
  
      nohup venv/bin/python3 -u  Main.py  -gpu $GPU -stg $STAGE -ct $CITY -mv $MODEL -bs $BATCH -lr $LRATE -bownws $BOWWRDS > "scripts/out/"$CITY"/model_"$MODEL"_"$BATCH"_"$BOWWRDS"_["$LRATE"].txt" &      
      GPU=$(($(($GPU+1%2))%2))

      # Si se alcanza el máximo de procesos simultaneos, esperar
      while [ $(jobs -r | wc -l) -eq $MAXTSTS ];
      do
        sleep 5
      done

      #Esperar X segundos entre pruebas para que le de tiempo a ocupar memoria en GPU
      sleep 10

    done
done

# Esperar por los últimos
while [ $(jobs -r | wc -l) -gt 0 ];
do
  sleep 5
done