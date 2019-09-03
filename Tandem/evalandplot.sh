#!/bin/bash


#PID=26512
#while [ -e /proc/$PID ]
#do
#    echo "Process: $PID is still running" 
#        sleep 10m
#done
python evaluate.py
python 'test.py'
