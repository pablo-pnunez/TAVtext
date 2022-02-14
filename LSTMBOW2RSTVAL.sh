#!/bin/bash

MAXTSTS=2
STAGE=0 # GRIDSEARCH o TRAIN
i=0
GPU=0

declare -a CITIES=( "paris" )

declare -a MODELS=( "0" "1" "2")
#declare -a BATCHES=( 512 256 128 )
declare -a BATCHES=( 1024 2048 4096 )
declare -a LRATES=( 5e-4 1e-3 5e-3 )
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
          nohup venv/bin/python3 -u  Main.py  -gpu $GPU -stg $STAGE -ct $CITY -mv $MODEL -bs $BATCH -lr $LRATE -bownws $BOWWRDS > "scripts/out/"$CITY"/model_"$MODEL"_"$BATCH"_"$BOWWRDS"_["$LRATE"].txt" &
          
          GPU=$(($(($GPU+1%2))%2))

          # Almacenar los PID en una lista hasta alcanzar el máximo de procesos
          pids[${i}]=$!
          i+=1

          echo "   -[$!] $MODEL"

          # Si se alcanza el máximo de procesos simultaneos, esperar
          if [ "${#pids[@]}" -eq $MAXTSTS ];
          then

            # Esperar a que acaben los X
            for pid in ${pids[*]}; do
                wait $pid
            done
            pids=()
            i=0
          fi

          #Esperar X segundos entre pruebas para que le de tiempo a ocupar memoria en GPU
          sleep 10

        done

      done

    done

  done

done