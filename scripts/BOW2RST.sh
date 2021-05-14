#!/bin/bash

MAXTSTS=10
STAGE=0 # GRIDSEARCH o TRAIN
i=0

declare -a CITIES=( "gijon" )

declare -a MODELS=( "0" "1" "2" "3" "4")
declare -a LRATES=( 5e-5 1e-4 5e-4 1e-3 )
declare -a BATCHES=( 256 512 1024 )

for CITY in "${CITIES[@]}" ;do
  echo "$CITY"

  for MODEL in "${MODELS[@]}" ;do
    echo "-$MODEL"

    for LRATE in "${LRATES[@]}" ;do
      echo "--$LRATE"

      for BATCH in "${BATCHES[@]}" ;do
        echo "----$BATCH"

        #MANUAL GPU
        nohup /usr/bin/python3.6 -u  Main.py  -stg $STAGE -ct $CITY -mv $MODEL -bs $BATCH -lr $LRATE > "scripts/out/"$CITY"/model_"$MODEL"_"$BATCH"_["$LRATE"].txt" &

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