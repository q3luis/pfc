#!/bin/bash

PYTHON_COMAND='python'

SCRIP_COMAND='train_net.py'

BASE_PATH='../prototxt/'
SOLVER_FILE='/solver_25.prototxt'

array=( "sequence2" "sequence3" "sequence4" "sequence5" )
array=(  "sequence1"  )
for i in "${array[@]}"
do
	CONCAT=$BASE_PATH$i$SOLVER_FILE
	echo $CONCAT
	echo $PYTHON_COMAND  $SCRIP_COMAND $BASE_PATH$i$SOLVER_FILE
        $PYTHON_COMAND  $SCRIP_COMAND $BASE_PATH$i$SOLVER_FILE
done
