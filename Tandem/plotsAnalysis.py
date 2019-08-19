import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import evaluate
import seaborn as sns; sns.set()
def RetrieveFeaturePredictionNMse(model_name):
    """
    Retrieve the Feature and Prediciton values and place in a np array
    :param model_name: the name of the model
    return Xtruth, Xpred, Ytruth, Ypred
    """
    ##Retrieve the prediction and truth and prediction first
    feature_file = os.path.join('data', 'test_Xtruth_{}.csv'.format(model_name))
    pred_file = os.path.join('data', 'test_Ypred_{}.csv'.format(model_name))
    truth_file = os.path.join('data', 'test_Ytruth_{}.csv'.format(model_name))
    feat_file = os.path.join('data', 'test_Xpred_{}.csv'.format(model_name))

    #Getting the files from file name
    Xtruth = pd.read_csv(feature_file,header=None, delimiter=' ')
    Xpred = pd.read_csv(feat_file,header=None, delimiter=' ')
    Ytruth = pd.read_csv(truth_file,header=None, delimiter=' ')
    Ypred = pd.read_csv(pred_file,header=None, delimiter=' ')
    
    #retrieve mse, mae
    Ymae, Ymse = evaluate.compare_truth_pred(pred_file, truth_file) #get the maes of y
    
    print(Xtruth.shape)
    return Xtruth.values, Xpred.values, Ytruth.values, Ypred.values, Ymae, Ymse

def ImportColorBarLib():
    """
    Import some libraries that used in a colorbar plot
    """
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib as mpl
    print("import sucessful")
    
    return mpl
  
def UniqueMarkers():
    import itertools
    markers = itertools.cycle(( 'x','1','+', '.', '*','D','v','h'))
    return markers
  
def SpectrumComparisonNGeometryComparison(rownum, colnum, Figsize, model_name, boundary):
    """
    Read the Prediction files and plot the spectra comparison plots
    :param SubplotArray: 2x2 array indicating the arrangement of the subplots
    :param Figsize: the size of the figure
    :param Figname: the name of the figures to save
    :param model_name: model name (typically a list of numebr containing date and time)
    """
    mpl = ImportColorBarLib()    #import lib
    
    Xtruth, Xpred, Ytruth, Ypred, Ymae, Ymse =  RetrieveFeaturePredictionNMse(model_name)  #retrieve features
    print("Ymse shape:",Ymse.shape)
    print("Xpred shape:", Xpred.shape)
    print("Xtrth shape:", Xtruth.shape)
    #Plotting the spectrum comaprison
    f = plt.figure(figsize=Figsize)
    fignum = rownum * colnum
    for i in range(fignum):
      ax = plt.subplot(rownum, colnum, i+1)
      plt.ylabel('Transmission rate')
      plt.xlabel('frequency')
      plt.plot(Ytruth[i], label = 'Truth',linestyle = '--')
      plt.plot(Ypred[i], label = 'Prediction',linestyle = '-')
      plt.legend()
      plt.ylim([0,1])
    f.savefig('Spectrum Comparison_{}'.format(model_name))
    
    """
    Plotting the geometry comparsion, there are fignum points in each plot
    each representing a data point with a unique marker
    8 dimension therefore 4 plots, 2x2 arrangement
    
    """
    #for j in range(fignum):
    pointnum = fignum #change #fig to #points in comparison
    
    f = plt.figure(figsize = Figsize)
    ax0 = plt.gca()
    for i in range(4):
      truthmarkers = UniqueMarkers() #Get some unique markers
      predmarkers = UniqueMarkers() #Get some unique markers
      ax = plt.subplot(2, 2, i+1)
      #plt.xlim([29,56]) #setting the heights limit, abandoned because sometime can't see prediciton
      #plt.ylim([41,53]) #setting the radius limits
      for j in range(pointnum):
        #Since the colored scatter only takes 2+ arguments, plot 2 same points to circumvent this problem
        predArr = [[Xpred[j, i], Xpred[j, i]] ,[Xpred[j, i + 4], Xpred[j, i + 4]]]
        predC = [Ymse[j], Ymse[j]]
        truthplot = plt.scatter(Xtruth[j,i],Xtruth[j,i+4],label = 'Xtruth{}'.format(j),
                                marker = next(truthmarkers),c = 'm',s = 40)
        predplot  = plt.scatter(predArr[0],predArr[1],label = 'Xpred{}'.format(j),
                                c =predC ,cmap = 'jet',marker = next(predmarkers), s = 60)
      
      plt.xlabel('h{}'.format(i))
      plt.ylabel('r{}'.format(i))
      rect = mpl.patches.Rectangle((boundary[0],boundary[2]),boundary[1] - boundary[0], boundary[3] - boundary[2],
																		linewidth=1,edgecolor='r',
                                   facecolor='none',linestyle = '--',label = 'data region')
      ax.add_patch(rect)
      plt.autoscale()
      plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                 mode="expand",ncol = 6, prop={'size': 5})#, bbox_to_anchor=(1,0.5))
    
    cb_ax = f.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = f.colorbar(predplot, cax=cb_ax)
    #f.colorbar(predplot)
    f.savefig('Geometry Comparison_{}'.format(model_name))


class HMpoint(object):
    """
    This is a HeatMap point class where each object is a point in the heat map
    properties:
    1. BV_loss: best_validation_loss of this run
    2. feature_1: feature_1 value
    3. feature_2: feature_2 value, none is there is no feature 2
    """
    def __init__(self, bv_loss, f1, f2 = None, f1_name = 'f1', f2_name = 'f2'):
        self.bv_loss = bv_loss
        self.feature_1 = f1
        self.feature_2 = f2
        print(type(f1))
    def to_dict(self):
        return {
            self.f1_name: self.feature_1,
            self.f2_name: self.feature_2,
            self.bv_loss: self.bv_loss
        }


def HeatMapBVL(plot_x_name, plot_y_name, save_name, HeatMap_dir = 'HeatMap',feature_1_name=None, feature_2_name=None,
                heat_value_name = 'best_validation_loss'):
    """
    Plotting a HeatMap of the Best Validation Loss for a batch of hyperswiping thing
    First, copy those models to a folder called "HeatMap"
    Algorithm: Loop through the directory using os.look and find the parameters.txt files that stores the
    :param HeatMap_dir: The directory where the checkpoint folders containing the parameters.txt files are located
    :param feature_1_name: The name of the first feature that you would like to plot on the feature map
    :param feature_2_name: If you only want to draw the heatmap using 1 single dimension, just leave it as None
    """
    one_dimension_flag = False          #indication flag of whether it is a 1d or 2d plot to plot
    #Check the data integrity 
    if (feature_1_name == None):
        print("Please specify the feature that you want to plot the heatmap");
        return
    if (feature_2_name == None):
        one_dimension_flag = True
        print("You are plotting feature map with only one feature, plotting loss curve instead")

    #Get all the parameters.txt running related data and make HMpoint objects
    HMpoint_list = []
    for subdir, dirs, files in os.walk(HeatMap_dir):
        for file_name in files:
             if (file_name == 'parameters.txt'):
                file_path = os.path.join(subdir, file_name) #Get the file relative path from 
                df = pd.read_csv(file_path, index_col = 0)
                if (one_dimension_flag):
                    HMpoint_list.append(HMpoint(float(df[heat_value_name][0]), eval(df[feature_1_name][0]), f1_name = feature_1_name))
                else:
                    HMpoint_list.append(HMpoint(float(df[heat_value_name][0]),eval(df[feature_1_name][0]),eval(df[feature_2_name][0]),
                                        feature_1_name, feature_2_name))
    #Change the feature if it is a tuple, change to length of it
    for cnt, point in enumerate(HMpoint_list):
        print("For point {} , it has {} loss, {} for feature 1 and {} for feature 2".format(cnt, 
                                                                point.bv_loss, point.feature_1, point.feature_2))
        assert(isinstance(point.bv_loss, float))        #make sure this is a floating number
        if (isinstance(point.feature_1, tuple)):
            point.feature_1 = len(point.feature_1)
        if (isinstance(point.feature_2, tuple)):
            point.feature_2 = len(point.feature_2)

    #After we get the full list of HMpoint object, we can start drawing 
    if (feature_2_name == None):
        print("plotting 1 dimension HeatMap (which is actually a line)")
        HMpoint_list_sorted = sorted(HMpoint_list, key = lambda x: x.feature_1)
        #Get the 2 lists of plot
        bv_loss_list = []
        feature_1_list = []
        for point in HMpoint_list:
            bv_loss_list.append(point.bv_loss)
            feature_1_list.append(point.feature_1)
        print("bv_loss_list:", bv_loss_list)
        print("feature_1_list:",feature_1_list)
        #start plotting
        f = plt.figure()
        plt.plot(feature_1_list, bv_loss_list)
        plt.xlabel(plot_x_name)
        plt.ylabel(plot_y_name)
        plt.savefig(save_name)
    else: #Or this is a 2 dimension HeatMap
        print("plotting 2 dimension HeatMap")
        point_df = pandas.DataFrame.from_records([point.to_dict() for point in HMpoint_list])
        point_df_pivot = point_df.pivot(feature_1_name, feature_2_name, "bv_loss")
        f = plt.figure()
        sns.heatmap(point_df_pivot)
        plt.savefig(save_name)

