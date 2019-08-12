# Back-propagation method

## Method Description
The back-propagation is a simple method for inverse design generation. It just trains a accurate forward model and fix all weights in that model to train the only input (geometry parameters) to fit the ultimate spectrual response.

This model only trains for 1 stage. However, due to the gradient descend nature of this method, the inference time would be long.

## Loss function
Boundary Loss: The Loss that describes our desired range of geometer parameters. If it is out of range, punish it by this loss.
Prediction Loss: Normal MSE loss for evaluating the accuracy of the prediciton. **Error is measured by Mean Squred Error.**

## Citation of this model
Although there has been very similar thoughts long time ago, the latest paper that I can find which used this method on inverse design:
Peurifoy, J., Shen, Y., Jing, L., Yang, Y., Cano-Renteria, F., DeLacy, B. G., ... & Soljačić, M. (2018). Nanophotonic particle simulation and inverse design using artificial neural networks. Science advances, 4(6), eaar4206.




