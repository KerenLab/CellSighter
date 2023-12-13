
# CellSighter
<img src="./CellSighterLogo.jpg" style="width: 15%; height: 15%">


CellSighter is an ensemble of convolutional neural networks to perform supervised cell classification in multiplexed images. Given a labeled training set, a model can be trained to predict cell classes for new images.

*Run on python 3.8.5*

## Data Preparation
The Data should have the following structure:
* The raw images should be in: {data_path}/CellTypes/data/images

  * Each image should be saved in a format of npz or tiff file as a 3D image shaped: HxWxC, C is the number of proteins in the data

* The segmentation should be in: {data_path}/CellTypes/cells 
  * For each image there should be a segmentation file in a format of npz or tiff file, shaped HxW. The segmentation file is a labeled object matrix whereby all pixels belonging to a cell will have the value of their cell id. The cells should be numbered from 1 to the number of cells in the image.

* The labels should be in: {data_path}/CellTypes/cells2labels 
  * For each image there should be a file in a format of npz (*.npz), such that each row has the label of the cell id as the index of the row. 
  * Another option is to save it as a txt format (*.txt) where each line is separated by \n. Such that the index of each row matches the cell id
  * Note that for cells without labels (eg test), you should set the label to -1, but all cells should appear in the file.

* Channels file, a txt file with the names of proteins ordered according to the order of the proteins in the image file.
the names of the proteins should be separated by \n.

### Notes:
- In the folder "example_experiment" you can find a small example of the data preparation and configuration file for the network.
- The names of the files should be the image id

- The labels of the cells should be integer numbers.

## System requirements
1. Access to GPU
2. See Requirements file for libraries and versions.


## Training a Model

1. Prepare the data in the format above
2. Create a folder with the configuration file named "config.json".
    See "Preparing configuration file"  for more information.
3. Train one model with the following command:
    'python train.py --base_path=/path/to/your/folder/with/the/configuration file'
4. In order to run an ensemble, run the command above more than one time in multiple folders.

### Output files:
1. val_results_{epocNum}.csv - Results on validation set along training.
The file contains the following columns:  
pred - prediction label  
pred_prob - probability of predicting the label  
label - input label to the training  
cell_id - cell_id  
image_id - image_id  
prob_list - list of probabilities per cell type. The index is the cell type.  
2. Weights_{epocNum}_count.pth - The weights of the network.    
3. event.out.### - tensorboard logs

## Evaluating the model

1. Prepare the data in the format above
2. Create a folder with the configuration file named "config.json".
    See "Preparing configuration file"  for more information.
3. Change the "weight_to_eval" field in the config file to be the path to the weights of the model you trained (Weights_{epocNum}_count.pth).
4. Evaluate one model with the following command:
    'python eval.py --base_path=/path/to/your/folder/with/the/configuration file'
5. You should now have a results csv in the folder.
6. In order to run an ensemble just run the command above for each model you trained. Make sure to change the weight paths and work on multiple folders one for each model.
    You should now have multiple results files. You can combine them as you wish, or use the merging scripts supplied.

### Output file:
1. val_results - same format as training
2. event.out.### - tensorboard logs

## Analyze results

- You can merge the results of the ensemble to one unified results file by running the following script:  
analyze_results/unified_ensemble.py  
In the script you will need to fill in the list of paths to all the val_results.csv files that you got from the ensemble.
The output of the script will be a unified results file named "merged_ensemble.csv", with the following columns:  
pred - prediction label  
pred_prob  - probability of predicting the label   
label - input label   
cell_id - cell_id  
image_id - image_id  
- You can visualize a confusion matrix of the input labels and CellSighter labels by running the following script: analyze_results/confusion_matrix.py   
You will need to fill in the path to the csv results file. The script will generate a confusion matrix and save it as a png file.

## Preparing configuration file
The configuration file should be named 'config.json' and should have the following fields:
>   "crop_input_size": 60,  #size of crop that goes into the network. Make sure that it is sufficient to visualize a cell and a fraction of its immediate neighbors.   
    "crop_size": 128,  #size of initial crop before augmentations. This should be ~2-fold the size of the input crop to allow augmentations such as shifts and rotations.  
    "root_dir": "data_path",  #path to the data that you've prepared in previous steps  
    "train_set": ["FOV1", "FOV2", ...],  #List of image ids to use as training set  
    "val_set": ["FOV10", "FOV12", ...],  #List of image ids to use as validation/evaluation set  
    "num_classes": 20,  #Number of classes in the data set  
    "epoch_max": 50, #Number of epochs to train  
    "lr": 0.001, # learning rate value  
    "to_pad": false, #Whether to work on the border of the image or not  
    "blacklist": [],  #channels to not use in the training/validation at all  
    "channels_path": "",  #Path to the protein list that you created during data preparation  
    "weight_to_eval": "",  #Path to weights, relevant only for evaluation  
    "sample_batch": true, #Whether to sample equally from the category in each batch during training   
    "hierarchy_match": {"0": "B cell", "1": "Myeloid",...}  #Dictionary of matching classes to higher category for balancing higher categories during training. The keys should be the label ids and the values the higher categories.   
    "size_data": 1000, #Optional, for each cell type sample size_data samples or less if there aren't enough cells from the cell type  
    "aug": true #Optional, whether to apply augmentations or not
