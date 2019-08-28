"""
This function is the higher trigger for doing evaluation time anaylysis. It just calls
the evalution iteratively with different size of test case each time to get a evaluation statistic
"""
import numpy as np
import evaluate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
if __name__ == '__main__':
    #Swipe through different size of test sets and evaluate them all
    #for ratio in np.arange(0.01, 1, 0.02):
    #    evaluate.evaluate_with_ratio(ratio)

    #Plot the anaylsis graph into a real analysis
    time_list = []
    model_name_list = ["VAE","Tandem","Backprop"]
    for model_name in model_name_list:
        time_analysis = pd.read_csv('../Results/{}_time_keeper.txt'.format(model_name),names = ["number", "time"])
        time_analysis = time_analysis.sort_values(by = ['number'])
        time_list.append(time_analysis)
    #time_analysis.info()
    #time_analysis = time_analysis.sort_values(by = ['number'])

    #Start plotting
    f = plt.figure()
    for cnt, model_name in enumerate(model_name_list):
        plt.plot(time_list[cnt]["number"], time_list[cnt]["time"],'-x',label = model_name)
    plt.ylabel('Time spent (s)')
    plt.xlabel('Number of inferenced points')
    plt.legend()
    plt.ylim(0,300)
    plt.title('Inference Comparison for 3 models')
    f.savefig("Model Inference Time Analysis.png")
