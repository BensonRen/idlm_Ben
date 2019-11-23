# Built in
import glob
import os
import shutil

# Torch

# Own
import flag_reader
import data_reader
from class_wrapper import Network
from model_maker import Forward


def put_param_into_folder():  # Put the parameter.txt into folder and take the best validation error as well
    list_of_files = glob.glob('models/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    print("The parameter.txt is put into folder " + latest_file)
    destination = os.path.join(latest_file, "parameters.txt");
    shutil.move("parameters.txt", destination)


if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()

    # Get the data
    train_loader, test_loader = data_reader.read_data(x_range=flags.x_range,
                                                      y_range=flags.y_range,
                                                      geoboundary=flags.geoboundary,
                                                      batch_size=flags.batch_size,
                                                      normalize_input=flags.normalize_input,
                                                      data_dir=flags.data_dir)
    # Reset the boundary is normalized
    if flags.normalize_input:
        flags.geoboundary = [-1, 1, -1, 1]

    print("Boundary is set at:", flags.geoboundary)
    print("Making network now")

    # Make Network
    ntwk = Network(Forward, flags, train_loader, test_loader)

    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder
    flag_reader.write_flags_and_BVE(flags, ntwk.best_validation_loss)
    put_param_into_folder()
    


