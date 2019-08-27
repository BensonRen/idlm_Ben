"""
This is the helper function for retrieving the prediction or truth files in data/
"""
import os

def get_Xpred(path):
    for filename in os.listdir(path):
        if ("Xpred" in filename):
            out_file = filename
            print("Xpred File found", filename)
            break
    return out_file

def get_Ypred(path):
    for filename in os.listdir(path):
        if ("Ypred" in filename):
            out_file = filename
            print("Ypred File found", filename)
            break;
    return out_file

def get_Xtruth(path):
    for filename in os.listdir(path):
        if ("Xtruth" in filename):
            out_file = filename
            print("Xtruth File found", filename)
            break;
    return out_file


def get_Ytruth(path):
    for filename in os.listdir(path):
        if ("Ytruth" in filename):
            out_file = filename
            print("Ytruth File found", filename)
            break;
    return out_file
