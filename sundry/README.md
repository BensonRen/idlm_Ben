# Sundry folder where sunderies are put!

## waitnrun.sh

Some Comments about this program:

The best program I've ever written (Simiao Ren, 2019)
Very Cool! (Evan Stump, 2019)

### This is a bash script that wait until your PID job finishs and then run some job after which it will email you the details of the job, Cool right?

Params:
1. PID: The job PID that you would like to wait
2. COMMAND: The job that you would like to run
3. Line18 nohup python: change to whatever platform you are using
4. limit: The time below which your program would be considered failure to run and would **not** send you email
5. The Email body starting from Line44. Feel free to change to your own name instead of mine.

The email that you would recieve contains:
1. Greeting messages (Most important one of course)
2. What did you run, the command
3. When did you run this script
4. When did it finished
5. How long did it took (Since you wait for another job, it just count the real execution time for current command)
6. The directory that you runned this command
7. The parameters that you filled into this run

You can also do whatever you want before or after your job
The email system used is ssmtp. You have to configure your own ssmtp settings before using this code.


## Cerus_config.md
This is the configuration log that I've done to the machine Cerus. Nothing special about it just in case something goes wrong (hopefully not)

## hyperswipe_1C.py
The hyperswipe code that deals with the 1D convolution layer. To keep a record therefore stored here.
Hyperswipe.py always changes because of the different thing that I have to swipe over. Versions would be kept a copy here.
