import matplotlib
matplotlib.use('agg') 		#To make it silent and dont output image and thus cause error
import matplotlib.pyplot as plt
import numpy as np
import plotsAnalysis
import pandas as pd
import evaluate 
import flag_reader
import os
import plotaAnalysis



#flags = flag_reader.read_flag()
#pred_file = 'data/test_Ypred_20190821_225753.csv'
#truth_file = 'data/test_Ytruth_20190821_225753.csv'
#
#mae, mse = evaluate.compare_truth_pred(pred_file, truth_file)
#plt.figure(figsize=(12, 6))
#plt.hist(mse, bins=100)
#plt.xlabel('Mean Squared Error')
#plt.ylabel('cnt')
#plt.suptitle('Backprop (Avg MSE={:.4e})'.format(np.mean(mse)))
#plt.savefig(os.path.join(os.path.abspath(''), 'data',
#                             'Backprop_{}.png'.format(flags.model_name)))
#plt.show()
#
#plotsAnalysis.SpectrumComparisonNGeometryComparison(3,2, (13,8), flags.model_name, [-1, 1, -1, 1])	

