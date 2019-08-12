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

nohup python $COMMAND 1>running.log 2>running.err & 
echo $! > pidfile.txt


PID=`cat pidfile.txt`
while [ -e /proc/$PID ]
do
    echo "Process: $PID is still running" 
        sleep 10m
done


{
	echo To: rensimiao.ben@gmail.com
	echo From: Cerus Machine
	echo Subject: Your Job has finished!
	echo -e "Dear mighty Machine Learning researcher Ben, \n \n"
	echo -e "    Your job has been finished and again, you saved so many fairies!!!\n \n"
	echo -e "Details of your job:\n 
	Job:  $COMMAND \n  
	PID:   `cat pidfile.txt` \n 
	StartTime:   $TIME \n 
	PWD:  $PWD\n"
} | ssmtp rensimiao.ben@gmail.com
echo "Process $PID has finished"

