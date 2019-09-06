#!/bin/bash


#PID=5023
#while [ -e /proc/$PID ]
#do
#    echo "Process: $PID is still running" 
#        sleep 10m
#done

TIME=`date`
PWD=`pwd`
#COMMAND=hyperswipe.py
#COMMAND=train.py
COMMAND=test_plot.py
#COMMAND=evaluate.py
SPACE='        '
SECONDS=0
nohup python $COMMAND 1>running.log 2>running.err & 
echo $! > pidfile.txt

PID=`cat pidfile.txt`
while [ -e /proc/$PID ]
do
    echo "Process: $PID is still running" 
        sleep 3m
done
#If the running time is less than 200 seconds (check every 180s), it must have been an error, abort
duration=$SECONDS
limit=200
if (( $duration < $limit )) 
then
    echo The program ends very shortly after its launch, probably it failed
    exit
fi

H=$(( $duration/3600 ))
M=$((( ($duration%3600 )) / 60 ))
S=$(( $duration%60 ))
#echo $H
#echo $M
#echo $S

CURRENTTIME=`date`
{
	echo To: rensimiao.ben@gmail.com
	echo From: Cerus Machine
	echo Subject: Your Job has finished!
	echo -e "Dear mighty Machine Learning researcher Ben, \n \n"
	echo -e  "    Your job has been finished and again, you saved so many fairies!!!\n \n"
	echo -e  "Details of your job:\n
        Job:  $COMMAND \n   
	PID:   `cat pidfile.txt` \n 
	TIME SPENT: $H hours $M minutes and $S seconds \n
        StartTime:   $TIME \n 
        ENDTIME: $CURRENTTIME \n
	PWD:  $PWD\n"
        cat parameters.py
} | ssmtp rensimiao.ben@gmail.com

echo "Process $PID has finished"

#Copying the parameters to the models folder as a record
#Lastfile=`ls -t models/ | head -1`
#mv parameters.txt models/$Lastfile/.
#cp parameters.py models/$Lastfile/.
#cp running.log models/$Lastfile/.
#cp running.err models/$Lastfile/.
