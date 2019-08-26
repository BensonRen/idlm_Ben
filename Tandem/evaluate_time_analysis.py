"""
This function is the higher trigger for doing evaluation time anaylysis. It just calls
the evalution iteratively with different size of test case each time to get a evaluation statistic
"""
import numpy as np
import evaluate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
if __name__ == '__main__':
    #Swipe through different size of test sets and evaluate them all
    for ratio in np.arange(0.01, 1, 0.02):
        evaluate.evaluate_with_ratio(ratio)

    #Plot the anaylsis graph into a real analysis
    time_analysis = pd.read_csv("data/time_keeper.txt",names = ["number", "time"])
    time_analysis.info()
    time_analysis = time_analysis.sort_values(by = ['number'])

    #Start plotting
    f = plt.figure()
    plt.plot(time_analysis["number"], time_analysis["time"],'-x')
    plt.ylabel('Time spent (s)')
    plt.xlabel('Number of inferenced points')
    plt.title('Inference Time vs Sample Size for Tandem Model')
    f.savefig("Tandem Model Inference Time Analysis.png")
