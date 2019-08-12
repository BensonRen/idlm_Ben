# Tandem Structure

## Method Description
The tandem structure is a simple Nueral Network (NN) architecture to solve the data inconsistency problem arise in meta-material inverse design. By simple connecting a backward network to a forward network and setting the loss function to the difference in spectra response, it circumvents the problem by sidestepping. This folder is to explore the effectiveness of this solution using our real data in meta-material design.

## Loss function
Boundary Loss: The Loss that describes our desired range of geometer parameters. If it is out of range, punish it by this loss.
Prediction Loss: Normal MSE loss for evaluating the accuracy of the prediciton. **Error is measured by Mean Squred Error.**

## Citation of this model
Although there has been very similar thoughts long time ago, the latest paper that I can find which used this method on inverse design:
Liu, D., Tan, Y., Khoram, E., & Yu, Z. (2018). Training deep neural networks for the inverse design of nanophotonic structures. ACS Photonics, 5(4), 1365-1369.
