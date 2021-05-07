#!/bin/bash

BATCH=1024
MAXTSTS=4
STAGE=1 # GRIDSEARCH o TRAIN
i=0

declare -a CITIES=( "gijon" )
declare -a MODELS=( "0" )
declare -a PCTGS=( .25 )
declare -a LRATES=( 5e-4 )

for CITY in "${CITIES[@]}" ;do
  echo "$CITY"

  for MODEL in "${MODELS[@]}" ;do
    echo "-$MODEL"

    for PCTG in "${PCTGS[@]}" ;do
      echo "--$PCTG"

      for LRATE in "${LRATES[@]}" ;do
        echo "---$LRATE"

        #MANUAL GPU
        nohup /usr/bin/python3.6 -u  Main.py  -s $STAGE -c $CITY -m $MODEL -p $PCTG -bs $BATCH -lr $LRATE > "out/cluster_explanation/gridsearch/"$CITY"/model_"$MODEL"_"$PCTG"_["$LRATE"].txt" &

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