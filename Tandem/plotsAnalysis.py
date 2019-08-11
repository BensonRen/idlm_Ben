import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import evaluate
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


