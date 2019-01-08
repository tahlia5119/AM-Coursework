# AM-Coursework

This code was implemented on a Windows 10 system using Anaconda. The external libraries that were downloaded include tensorflow, keras, sklearn, and pandas.

The structure of the containing file should be as below:

-Coursework\
--dataset\
--models_scalers\
--npy_files\
--npy_pca_files\
--scripts\
--test_results\
--testing_dataset\
--train_curves\
--train_test_csv\
--attributes_list.csv

To process the training data, these scripts should be run in the following order:

-noise_removal.py
-pca_pixel.py 
-pca_rgb.py

To use the processed training data on the test dataset, run each of the <task_name>_save_model.py scripts. 

Each script has the different data paths that need to be changed according to the users set up of files

To test the different sklearn classifiers for each label, run the sklearn_models.py script.

To test a chosen label and data type of RGB or grayscale images on the CNN model, run model.py and change the label, data_type, and data paths accordingly.

The label_curves.py script calls the run_model.py script to plot learning curves for whichever optimizer, parameters, and data type are selected in the run_model.py script.

The hyperparameters.py script is parameter selection for each label for the classifiers that were identified as the best accuracies in the sklearn_models.py scripts.

The <taskname>_save_model.py scripts fit the chosen model for each task, as well as any scalers or encoders used in processing the data, and saves these to either pickle, json, or HF5D files which are then later implemented in the <taskname>_test.py files and the <taskname>_final.py scripts.
 
The <taskname>_final.py scripts take new input data, converts the data accordingly and makes predictions based on the data and outputs them to a csv file with the inference accuracy.
