

from subprocess import call

sequences=["sequence2","sequence3","sequence4","sequence5"]

# nohup  python train_net.py  /vol/pfc/prototxt/sequence1/solver.prototxt >  /vol/tmp/seq1_25.log &


#NOHUP_COMAND='nohup'
PYTHON_COMAND='python'

SCRIP_COMAND='train_net.py'

BASE_PATH='../prototxt/'
SOLVER_FILE='solver_25.prototxt '


call(["ls", "-l"])

for i in sequences:

    comand=PYTHON_COMAND+" "+SCRIP_COMAND+" "+BASE_PATH+i+"/"+SOLVER_FILE

    print comand
    call([PYTHON_COMAND,SCRIP_COMAND,"'"+BASE_PATH+i+"/"+SOLVER_FILE+"'"])

