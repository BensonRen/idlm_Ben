"""
This function is to calculate the histogram of the VAE stucture evaluation result.
It is called after the program calls the hidden Tandem Prediction module
"""
import evaluate
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ = '__main__':
    for filename in os.listdir('data/')
        if ("Ytruth" in filename):
            truth_file = filename
            print("Truth File found", filename)
        if ("Ypred"  in filename):
            pred_file = filename
            print("Pred File found",filename)
    assert(isinstance(pred_file, str) and isinstance(truth_file, str), "One of your pred file or truth file is missing!!!")

    mae, mse = evaluate.compare_truth_pred(pred_file, truth_file)

    plt.figure(figsize=(12, 6))
    plt.hist(mse, bins=100)
    plt.xlabel('Mean Squared Error')
    plt.ylabel('cnt')
    plt.suptitle('VAE (Avg MSE={:.4e})'.format(np.mean(mse)))
    plt.savefig(os.path.join(os.path.abspath(''), 'data',
                             'VAE_{}.png'.format(flags.model_name)))
    plt.show()
    print('VAE (Avg MSE={:.4e})'.format(np.mean(mse)))
