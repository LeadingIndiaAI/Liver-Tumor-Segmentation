# LiTS
This code is the solution to the Liver Tumor Segmentation Challenge from www.codalab.org
To run this code, the train images and masks should be converted to .bmp files and can be done through Train_bmp_conversion.py and these files can be listed into csv using utils.py
training.py can be used for training the model(VNET) in this case and prediction can be done by prediction.py i.e, converting the test data into .bmp or npy files using the test_npy_conversion.py file.
