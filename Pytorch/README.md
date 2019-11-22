# Pytorch implementation of the whole repo

This is a transition starting from Nov. 20, 2019

# Developer Log:

### RoadMap for this work:
1. Transition the major infra-structure from tensorflow to pytorch
2. Implement a simple forward network for meta-material project
3. Implement the GAN model that was not available in tensorflow version

## 2019.11.20
Background: Start Pytorch transition. GAN model (not in previous tf repo) would be the first model to work on.

Function completed:
1. Construct the network main class wrapper for Forward
2. Handle the data reader transition to pytorch
3. Flag reader to pytorch version

## 2019.11.21

Function completed/Bug Fixed:
1. Forward training module done
2. Forward training tested
3. Bug Fixed for storing the parameters.txt file

## To dos:

1. Add the bounday loss module
2. Add the forward inference module
3. Train on GPU environment for forward model
