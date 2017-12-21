# EPFL Machine Learning Project 2: Road extraction from satellite images

## Team: Chronic Machinelearnism

## Code architecture
The code consists of four Python files:
* `run.py` : The ML pipeline. Fits the model and output the predictions for the submission dataset (submission_test.csv).
* `helpers.py` : Definition of all the auxiliary methods (e.g. image manipulation).

## External Dependencies
Keras + TensorFlow backend
OpenCV
imutils


## Running
The user simply needs to *Python3-execute* the run.py file.

*Note: All the above mentioned .py files needs to be in the same folder. This folder needs to contain a subfolder called 'data' with the training and submission folders "extracted as is" from Kaggle.*

*Note: Running the code requires quite some memory. Having (at least) 40GB of RAM is highly recommended.*

## Running time.
The model was trained on a single p2.8xlarge (AWS) instance in around 1 hour. On a laptop we expect the training time to be around 72 hours. We ran our run.py with all training data in multi-gpu mode (disabled on the deliverable). The data augmentation is very memory hungry, taking a considerable amount of memory; at least 128GB of RAM are required to train the model with the full dataset.

## Authors
Aimee Montero, Alfonso Peterssen, Philipp Chervet  
