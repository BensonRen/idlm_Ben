PID=22228
while [ -e /proc/$PID ]
do
    echo "Process: $PID is still running" 
        sleep 10m
done

#nohup matlab -nodisplay -nosplash -nodesktop < ~/matlabmd/xxxxxxx 1 > running.log 2>error.log

nohup python train.py 1>running.log 2>running.err 

{
	echo To: rensimiao.ben@gmail.com
	echo From: Cerus Machine
	echo Subject: Your Job has finished!
	echo "Hello your mighty Machine Learning researcher, your job has been finished and again, you saved so many fairies!!!"
} | ssmtp rensimiao.ben@gmail.com
echo "Process $PID has finished"

