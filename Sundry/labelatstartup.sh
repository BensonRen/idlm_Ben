#!/bin/bash



TIME=`date`



echo -e "Hi, Cerus is started or rebooted at time $TIME . Please note that this is a scipt added to the start sequence" >> ~/Documents/Ben/startup.txt

CURRENTTIME=`date`
{
	echo To: rensimiao.ben@gmail.com
	echo From: Cerus Machine
	echo Subject: Your Cerus has been rooted!
	echo -e "Dear mighty Machine Learning researcher Ben, \n \n"
	echo -e  "    Your Cerus has just been rebooted at time $TIME"
} | ssmtp rensimiao.ben@gmail.com


#Copying the parameters to the models folder as a record
#Lastfile=`ls -t models/ | head -1`
#mv parameters.txt models/$Lastfile/.
#cp parameters.py models/$Lastfile/.
#cp running.log models/$Lastfile/.
#cp running.err models/$Lastfile/.
