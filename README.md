# table-detection

All notebooks were run in Google Colaboratory.

Data preparation:
table_recognition_data_preprocessing.ipynb is used to move and consolidate the image files, while reformatting the table data into a text file. It is done for the main 3 datasets, such that the final compiled text file contains data of all 3 datasets.

Image_aug.ipynb is used to create duplicated dilated and smudged images, to be stored separately.


Faster RCNN:
train_baselines.ipynb trains Faster RCNN models with kernel filters of different sizes. The kernel filter size is changed with the parameter side_length in the second cell of the notebook. The model saves as a different file name based on the kernel filter size. After training for 80 epochs, the training procedure will end, and the step parameter needs to be changed to 2 to indicate the last stage of the iterative transfer learning process.

test_v2.ipynb evaluates a model, based on a side_length of the kernel filter given. It infers the model file assuming the same naming convention used in train_baselines.ipynb. Various test sets can be used, stored in the list "validation_files" on the last code cell.

Retinanet:
train.py trains Retinanet models in a similar way to that of Faster RCNN. After training for 80 epochs, training using general tables is stopped and the dataset used for training will be changed to borderless tables instead for the last step of the iterative transfer learning process. 

validation.py calculates the recall and precision of the models trained with varying IoU after importing functions from csv_eval.py, dataloader.py and model.py.
