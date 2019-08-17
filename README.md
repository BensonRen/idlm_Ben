# idlm_Ben
This project is for the problem of inverse mapping for meta-material design using deep learning. Special thanks for [Bohao Huang](https://github.com/bohaohuang) and [Christian nadell](https://github.com/chnadell)  for infrastructure of the code. The previous version for the forward mapping learning can be found following this [link](https://github.com/chnadell/dlmCN)

Developer Log:
# Real Dataset

## 2019.06.19

Background: This is modified code from BoHao and the forward model. The backward model is now fully implemented to be tested on the real data set. The original settings of this code was separated files each with different funcitons. Here within this notebook those files would be separated by sections and each sections would be defined before calling them.

## 2019.06.21

Attempts made to integrate the backward model with the forward model checkpoint file provided by Christian. 3 methods were attempted, including direct import (failed because of static graph definition), import from name (failed because of uncertain graph name definition) model extraction method (failed due to incompatibility of tensorflow graph and keras model instaintiation, **which  still comes from an unknown source**)

## 2019.06.24

As dicussed with Jordan on  Friday, the forward model actually costs such little time to train, therefore the new current method would be writing our own forward model and train the related parameters.

**!!However, now lets try the 4th attempt for restoring ckpt files, restore form meta files and run session to import the weights**

## 2019.06.27

After trail runs, the 4th method of building the graph from meta file succeed. Therefore we can spare the time for re-training a forward model.

Module "retrieve forward model" is added to retrieve the forward model. Only the input and output tensor are instaintiated for convenience

**However, this doesn't work for changing the tensors to be trainable and therefore this is abandoned. The tandem structure direct construction method is re-activated(2019.07.31)**

### 2019.06.27-2019.07.28 Vacation Time 
## 2019.07.29

Add collection control on the variables created in the tandem structure so that trainable property can be set on and off by the collection name.

**The algorithm is working now**

Minor problem: 1. The training hook of forward and tandem structure is in the same place (Fixed 08.04)

## 2019.08.04 

Now, the goal is to validate the various methods for dealing the inverse problem. (**Highlight means working now, rest are pending**)

1. Tadem structure (Done)
2. Back-propagation (Done)
3. **Variational Auto-Encoder (VAE)**
4. Generative Adverserial Network (GAN)

Infrastructures added:
1. Spectrum plot for comparing real and predicted structure
2. Evaluate Module added

## 2019.08.05
refining the tandem structure.
Infrastructure Added:
 1. Geometry plot with color bar indicating the error
 2. Loading from existing ckpt and train from there 
 3. Boundary loss implementation Testing

## 2019.08.06

 Boundary loss showing a little wired behavior, continue testing on it.

 Infrastructure added:
 1. Customized summary name 
 2. Symmetric Awareness Distance metric (**Thinking**)

## 2019.08.09

 The tandem training loss is higher than expected. Normalizing the input geomery space for further investigation.

## 2019.08.11

Input normalization done, however during analysis the tandem structure did not learn effective output. 

Bug fixed / added function:
1. partial normalization problem;
2. Plot boundary problem

## 2019.08.12

Bug fixed / added function:
1. Tandem loading pre-training forward model and train from the backward model
2. Early stopping Hook
3. NAN stopping Hook 
4. Backpropagation input normalization
5. Running Hyper-parameter logging (Running flags)
6. Solved the bug of Hyper-parameter logging where only default parameter is recorded. Currently the running flags would be recorded to file called 'parameters.txt'

## 2019.08.13

Problem found:

The tandem structure struggled to learn effective backward representation of the geometry of the material. The test accuracy is not significantly higher than the training one which is signifying it is not over-fitting but. After consulting Evan, hyper-parameter search for a better architecture would be the first thing worth trying and trend of the training shall be analyzed.

Bug fixed / added function:
1. Flag altering training function added to train.py in Tandem structure
2. Hyper parameter search module added called 'hyperswipe.py'
3. Bug fixed for flag logging would change the original data strcuture for coloumn 'y_range'
4. Bug fixed for flag logging would fail under hyper-parameter swiping (Now store once one model is trained)
5. Random Split for the training and testing data points

## 2019.08.14

Bug fixed / added function:
1. Credit to Chrisitan and Bohao added with their personal profile and url linked at README front page
2. VAE structure sub modules: Spectra_encoder, encoder and decoder

## 2019.08.15

Bug fixed / added function:
1. VAE model Connection
2. VAE Model maker
3. waitnrun bug for arithmetic expression
4. VAE Loss
5. VAE Testing

## 2019.08.16

Problem Found:

Both tandem strcuture as well as the VAE model, the testing loss is very high. After swiping through model complexity, VAE model overfitted and Tandem one just couldn't fit given a long training time.

Investigation towards proper convolution that deal with spectra is the next step.

Function Added:
1. Tandem model backward convolution customization
2. Tandem model conv1D swiping test
3. MSE/REG/BDY loss separation summary
4. Hook creation function where you only need to provide the loss names and the loss, it returns a list of hook to you
5. Make trace back to lowest loss Hooks
6. Turn off the conv1D and swipe across other architectures (pl:1)

## 2019.08.17

Functions Added/ bug fixed:
1. Weight summary system added. A new hook has been added to the network_helper.py file and auto added towards the training hooks
2. Bug fixed and weight summary system tested
3. Heat mapping prerequisite done by adding the best_validation_loss into the parameter.txt file

Pending to work: 
1. Heat Map plotting (record the lowest test loss and print to the parameter file? pl : 1)
2. VAE Evaluate 


### !!!HUGE BUG DETECTED, SOLVE BELOW POINTS NOW!!!
Cool you've fixed all bugs found, congrats!


 All possible **heights**:{30, 32, 34, 36, 38, 40, 42.5, 44, 46, 48, 50, 52 ,55}

 All possible **radius**: {42, 42.8, 43.7, 44.5, 45.3, 46.2, 47, 47.8, 48.6, 49.5, 50.4, 51.2, 52}

 h0 h1 h2 h3 r0 r1 r2 r3
