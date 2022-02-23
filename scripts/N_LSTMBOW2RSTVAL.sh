#!/bin/bash

MAXTSTS=4
STAGE=1 # GRIDSEARCH o TRAIN
GPU=0

declare -a CITIES=("gijon" "madrid" "barcelona" "paris" "newyorkcity" )

#declare -a MODELS=( "0" "1" "2")
declare -a MODELS=( "0")
# declare -a BATCHES=( 512 256 128 )
declare -a BATCHES=( 32 )
# declare -a LRATES=( 1e-4 5e-4 1e-3 5e-3 )
declare -a LRATES=( 5e-4 )

declare -a BOWNWORDS=( 10 )

# declare -a BOWNWORDS=( 400 )

for CITY in "${CITIES[@]}" ;do
  echo "$CITY"

  for BATCH in "${BATCHES[@]}" ;do
    echo "-$BATCH"

    for LRATE in "${LRATES[@]}" ;do
      echo "--$LRATE"

      for BOWWRDS in "${BOWNWORDS[@]}" ;do
        echo "---$BOWWRDS"

        for MODEL in "${MODELS[@]}" ;do
          echo "----$MODEL"

          #MANUAL GPU
          nohup venv/bin/python3 -u  Main.py -gpu $GPU -stg $STAGE -ct $CITY -mv $MODEL -bs $BATCH -lr $LRATE -bownws $BOWWRDS > "scripts/out/"$CITY"/model_"$MODEL"_"$BATCH"_"$BOWWRDS"_["$LRATE"].txt" &
          
          GPU=$(($(($GPU+1%2))%2))

          echo "   -[$!] $MODEL"

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