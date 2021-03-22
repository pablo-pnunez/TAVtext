#!/bin/bash

BATCH=1024
MAXTSTS=1
STAGE=100

MODEL="hl5"
PCTG=.25
LRATE=5e-4

i=0

declare -a CITIES=( "london" )

declare -a IMG_SLC_MTHS=( 0 1 ) # Seleccionar las N_IMGS representativas mediante clustering (0) o aleatoriamente (1)
declare -a CS_MODES=( "m5" "m10" "m15" "m20" )
declare -a N_IMGS=( 1 2 3 4 5 )

for CITY in "${CITIES[@]}" ;do
  echo "$CITY"

  for IMG_SLC_MTH in "${IMG_SLC_MTHS[@]}" ;do
    echo "-$IMG_SLC_MTH"

    for CS_MODE in "${CS_MODES[@]}" ;do
      echo "--$CS_MODE"

      for N_IMG in "${N_IMGS[@]}" ;do
        echo "---$N_IMG"

        nohup /usr/bin/python3.6 -u  Main.py -s $STAGE -c $CITY -m $MODEL -p $PCTG -bs $BATCH -lr $LRATE -csm $CS_MODE -inc $N_IMG -ism $IMG_SLC_MTH > "out/cluster_explanation/test/"$CITY"/"$CS_MODE"-"$N_IMG"-["$IMG_SLC_MTH"].txt" &

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
        sleep 1 # 600

      done

    done

  done

done