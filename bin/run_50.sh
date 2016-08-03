#!/bin/bash

PYTHON_COMAND='python'

SCRIP_COMAND='train_net.py'

BASE_PATH='../prototxt/'
SOLVER_FILE='/solver_50.prototxt'

echo "init run 50"
sleep 2h
echo "end sleep run train "

array=("sequence1" "sequence2" "sequence3" "sequence4" "sequence5" )
for i in "${array[@]}"
do
	CONCAT=$BASE_PATH$i$SOLVER_FILE
	echo $CONCAT
	echo $PYTHON_COMAND  $SCRIP_COMAND $BASE_PATH$i$SOLVER_FILE
        $PYTHON_COMAND  $SCRIP_COMAND $BASE_PATH$i$SOLVER_FILE
done