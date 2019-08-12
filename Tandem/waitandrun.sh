#PID=22228
#while [ -e /proc/$PID ]
#do
#    echo "Process: $PID is still running" 
#        sleep 10m
#done
TIME=`date`
PWD=`pwd`
COMMAND=train.py
SPACE='        '
SECONDS=0
nohup python $COMMAND 1>running.log 2>running.err & 
echo $! > pidfile.txt

PID=`cat pidfile.txt`
while [ -e /proc/$PID ]
do
    echo "Process: $PID is still running" 
        sleep 10m
done

duration=$SECONDS

{
	echo To: rensimiao.ben@gmail.com
	echo From: Cerus Machine
	echo Subject: Your Job has finished!
	echo -e "Dear mighty Machine Learning researcher Ben, \n \n"
	echo -e  "    Your job has been finished and again, you saved so many fairies!!!\n \n"
	echo -e  "Details of your job:\n
        Job:  $COMMAND \n   
	PID:   `cat pidfile.txt` \n 
	TIME SPENT: $(($duration / 3600)) hours $((($duration % 3600) / 60)) minutes and $(($duration % 60)) seconds \n
        StartTime:   $TIME \n 
        ENDTIME:

	PWD:  $PWD\n"
        cat parameters.py
} | ssmtp rensimiao.ben@gmail.com

echo "Process $PID has finished"

#Copying the parameters to the models folder as a record
Lastfile=`ls -t models/ | head -1`
cp parameters.py models/$Lastfile/.
cp running.log models/$Lastfile/.
cp running.err models/$Lastfile/.
