#!/bin/bash
# ejecuta las secuencias 2 4 y 5 con oversampling

PYTHON_COMAND='python'

SCRIP_COMAND='train_net.py'

BASE_PATH='../prototxt/'
SOLVER_FILE='/solver_50_test_3.prototxt'

array=( "sequence2" "sequence3" "sequence4" "sequence5" )
array=(  "sequence2oversample" "sequence4oversample" "sequence5oversample"  )
for i in "${array[@]}"
do
	CONCAT=$BASE_PATH$i$SOLVER_FILE
	echo $CONCAT
	echo $PYTHON_COMAND  $SCRIP_COMAND $BASE_PATH$i$SOLVER_FILE
        $PYTHON_COMAND  $SCRIP_COMAND $BASE_PATH$i$SOLVER_FILE
done
