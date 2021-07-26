#!/bin/bash

MAXTSTS=8
STAGE=0 # GRIDSEARCH o TRAIN
i=0

declare -a CITIES=( "gijon" )

declare -a MODELS=( "0" "3" )
declare -a LRATES=( 5e-6 1e-5 5e-5 1e-4 5e-4)
declare -a BATCHES=( 32 64 128 256 512 1024 )
declare -a BOWNWORDS=( 350 400 )

for CITY in "${CITIES[@]}" ;do
  echo "$CITY"

  for MODEL in "${MODELS[@]}" ;do
    echo "-$MODEL"

    for LRATE in "${LRATES[@]}" ;do
      echo "--$LRATE"

      for BOWWRDS in "${BOWNWORDS[@]}" ;do
        echo "---$BOWWRDS"

        for BATCH in "${BATCHES[@]}" ;do
          echo "----$BATCH"

          #MANUAL GPU
          nohup venv/bin/python3.8 -u  Main.py  -stg $STAGE -ct $CITY -mv $MODEL -bs $BATCH -lr $LRATE -bownws $BOWWRDS > "scripts/out/"$CITY"/model_"$MODEL"_"$BATCH"_"$BOWWRDS"_["$LRATE"].txt" &

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