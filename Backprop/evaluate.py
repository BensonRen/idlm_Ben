import argparse
import tensorflow as tf
import data_reader
import model_maker
import Backprop_network_maker
import network_helper
import plotsAnalysis
import os

INPUT_SIZE = 2
CLIP = 15
BACKWARD_FC_FILTERS = (100, 500, 1000, 1500, 500, 500, 300,100,8)
FORWARD_FC_FILTERS = (100, 500, 1000, 1500, 500, 2000, 1000, 500, 165)
TCONV_FNUMS = (4, 4, 4)
TCONV_DIMS = (165, 165, 330)
TCONV_FILTERS = (8, 4, 4)
N_FILTER = [15]
N_BRANCH = 2
REG_SCALE = 5e-8
CROSS_VAL = 5
VAL_FOLD = 0
BATCH_SIZE = 10
SHUFFLE_SIZE = 2000
VERB_STEP = 100
EVAL_STEP = 500
TRAIN_STEP = 2000
LEARN_RATE = 1e-4
DECAY_STEP = 1000
DECAY_RATE = 0.5
X_RANGE = [i for i in range(2, 10 )]
Y_RANGE = [i for i in range(10 , 2011 )]
# TRAIN_FILE = 'bp2_OutMod.csv'
# VALID_FILE = 'bp2_OutMod.csv'
FORWARDMODEL_CKPT = '20190508_155720'
FORCE_RUN = True
MODEL_NAME  = '20190807_003824'
DATA_DIR = '../'
def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=INPUT_SIZE, help='input size')
    parser.add_argument('--clip', type=int, default=CLIP, help='points clipped from each end of output after final conv')
    parser.add_argument('--forward-fc-filters', type=tuple, default=FORWARD_FC_FILTERS, help='#neurons in each fully connected layers')
    parser.add_argument('--backward-fc-filters', type=tuple, default=BACKWARD_FC_FILTERS, help='#neurons in each fully connected layers')
    parser.add_argument('--tconv-Fnums', type=tuple, default=TCONV_FNUMS, help='#0th shape dim of each tconv layer')
    parser.add_argument('--tconv-dims', type=tuple, default=TCONV_DIMS,
                        help='dimensionality of data after each transpose convolution')
    parser.add_argument('--tconv-filters', type=tuple, default=TCONV_FILTERS,
                        help='#filters at each transpose convolution')
    parser.add_argument('--n-filter', type=int, default=N_FILTER, help='#neurons in the tensor module'),
    parser.add_argument('--n-branch', type=int, default=N_BRANCH, help='#parallel branches in the tensor module')
    parser.add_argument('--reg-scale', type=float, default=REG_SCALE, help='#scale for regularization of dense layers')
    parser.add_argument('--x-range', type=list, default=X_RANGE, help='columns of input parameters')
    parser.add_argument('--y-range', type=list, default=Y_RANGE, help='columns of output parameters')
    parser.add_argument('--cross-val', type=int, default=CROSS_VAL, help='# cross validation folds')
    parser.add_argument('--val-fold', type=int, default=VAL_FOLD, help='fold to be used for validation')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (100)')
    parser.add_argument('--shuffle-size', default=SHUFFLE_SIZE, type=int, help='shuffle size (100)')
    parser.add_argument('--verb-step', default=VERB_STEP, type=int, help='# steps between every print message')
    parser.add_argument('--eval-step', default=EVAL_STEP, type=int, help='# steps between evaluations')
    parser.add_argument('--train-step', default=TRAIN_STEP, type=int, help='# steps to train on the dataset')
    parser.add_argument('--learn-rate', default=LEARN_RATE, type=float, help='learning rate')
    parser.add_argument('--decay-step', default=DECAY_STEP, type=int,
    parser.add_argument('--data-dir', default=DATA_DIR, type=str, help='data directory')
                        help='decay learning rate at this number of steps')
    parser.add_argument('--decay-rate', default=DECAY_RATE, type=float,
                        help='decay learn rate by multiplying this factor')
    parser.add_argument('--forward-model-ckpt', default=FORWARDMODEL_CKPT, type=str,
                        help='name of the forward ckpt file')
    parser.add_argument('--force-run', default=FORCE_RUN, type=bool, help='force it to rerun')
    parser.add_argument('--model-name', default=MODEL_NAME, type=str, help='name of the model')
    # parser.add_argument('--train-file', default=TRAIN_FILE, type=str, help='name of the training file')
    # parser.add_argument('--valid-file', default=VALID_FILE, type=str, help='name of the validation file')
    
    flags = parser.parse_args()  #This is for command line version of the code
    #flags = parser.parse_args(args = [])#This is for jupyter notebook version of the code
    return flags



def compare_truth_pred(pred_file, truth_file):
    """
    Read truth and pred from csv files, compute their mean-absolute-error and the mean-squared-error
    :param pred_file: full path to pred file
    :param truth_file: full path to truth file
    :return: mae and mse
    """
    pred = np.loadtxt(pred_file, delimiter=' ')
    truth = np.loadtxt(truth_file, delimiter=' ')

    mae = np.mean(np.abs(pred-truth), axis=1)
    mse = np.mean(np.square(pred-truth), axis=1)

    return mae, mse
def evaluatemain(flags, eval_forward):
    #Clear the default graph first for resolving potential name conflicts
    tf.reset_default_graph()
    
    ckpt_dir = os.path.join(os.path.abspath(''), 'models', flags.model_name)
    clip, forward_fc_filters, tconv_Fnums, tconv_dims, tconv_filters, \
    n_filter, n_branch, reg_scale = network_helper.get_parameters(ckpt_dir)
    print(ckpt_dir)
    # initialize data reader
    if len(tconv_dims) == 0:
        output_size = fc_filters[-1]
    else:
        output_size = tconv_dims[-1]
    features, labels, train_init_op, valid_init_op = data_reader.read_data(input_size=flags.input_size,
                                                               output_size=output_size-2*clip,
                                                               x_range=flags.x_range,
                                                               y_range=flags.y_range,
                                                               cross_val=flags.cross_val,
                                                               val_fold=flags.val_fold,
                                                               batch_size=flags.batch_size,
                                                               shuffle_size=flags.shuffle_size
																															 data_dir = flags.data_dir)
    # make network
    ntwk = Backprop_network_maker.BackPropCnnNetwork(features, labels, model_maker.back_prop_model, flags.batch_size,
                            clip=flags.clip, forward_fc_filters=flags.forward_fc_filters,
                              reg_scale=flags.reg_scale,
                            learn_rate=flags.learn_rate,tconv_Fnums=flags.tconv_Fnums,
                            tconv_dims=flags.tconv_dims,n_branch=flags.n_branch,
                            tconv_filters=flags.tconv_filters, n_filter=flags.n_filter,
                            decay_step=flags.decay_step, decay_rate=flags.decay_rate)

    # evaluate the results if the results do not exist or user force to re-run evaluation
    save_file = os.path.join(os.path.abspath(''), 'data', 'test_pred_{}.csv'.format(flags.model_name))
    if FORCE_RUN or (not os.path.exists(save_file)):
        print('Evaluating the model ...')
        pred_file, truth_file = ntwk.evaluate(valid_init_op, train_init_op,
                                              ckpt_dir=ckpt_dir,back_prop_ephoch = 1500,
                                              model_name=flags.model_name,
                                              write_summary=True,
                                              eval_forward = eval_forward)
    else:
        pred_file = save_file
        truth_file = os.path.join(os.path.abspath(''), 'data', 'test_truth.csv')

    mae, mse = compare_truth_pred(pred_file, truth_file)

    plt.figure(figsize=(12, 6))
    plt.hist(mse, bins=100)
    plt.xlabel('Mean Squared Error')
    plt.ylabel('cnt')
    plt.suptitle('Tandem (Avg MSE={:.4e})'.format(np.mean(mse)))
    plt.savefig(os.path.join(os.path.abspath(''), 'data',
                             'tandem_{}.png'.format(flags.model_name)))
    plt.show()
    print('Tandem (Avg MSE={:.4e})'.format(np.mean(mse)))

if __name__ == '__main__':
	flags = read_flag()
	evaluatemain(flags, eval_forward = False)
	plotsAnalysis.SpectrumComparisonNGeometryComparison(3,2, (13,8), flags.model_name)	

