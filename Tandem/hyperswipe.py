"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train

if __name__ == '__main__':
    flags = flag_reader.read_flag()  	#setting the base case
	

	#Setting the loop for setting the parameter
	for i in list(x):
		flags.XXX = i
		train.train_from_flag(flags)


