# 3D-Printing-Classification
The goal of this project is to detect failed or failing 3D printing projects during the printing process. Detecting and stopping a potential failure can save time, material and cost in a manufacturing environment, or personal hobby projects.

## Files

##### RunModel.py
- Use this file for predictions.
- Imports the `.pkl` function from the model creation.
- Imports the data for prediction. This defaults to video 6.
- Prints statements describing how the model handles the predictions.

##### MakePredFunc.py
- File containing the function that makes the prediction. See file for details. 

##### Model_Creation.py
- Imports the data and exports the model in a `.plk` format.
- Adjustable parameters are in the first few lines. See file for descriptions.

##### Data:
- Contains images and video streams that are used as training, test and predictions in the model.


## Data Source
All images are individually scraped from google images. There are 120 successful print images, and 120 failed/failing print images. 6 video streams have images sampled from them at specified intervals.

#### Good print:
![Good Print](https://github.com/KAValerio/3D-Printing-Classification/blob/main/Figures/good2.jpg?raw=true)

#### Failed print:
![Failed Print](https://github.com/KAValerio/3D-Printing-Classification/blob/main/Figures/fail2.jpg?raw=true)

#### Failing print:
![Failing Print](https://github.com/KAValerio/3D-Printing-Classification/blob/main/Figures/almost2.jpg?raw=true)

## Methodology
I created a binary image classification model using the dataset described above. Once this model was created, a photograph of the user's 3D printing project can be automatically taken at a specified time interval (e.g. every 1, 5 or 10 minutes) and fed into the model to predict the state of the print (i.e. is the current print a success or failure). 

The first step is to compress and process each of the scraped images. The photos taken from the image stream were also used as inputs into this algorithm. These images were downsampled and converted to 400 by 400 matrixes, and the colour was removed. These dimensions were chosen as it provides a balance between processing time and model accuracy. This process utilized the `.convert` and `.resize` functions in the Python`PIL` library. Next, this was converted to a NumPy array.

##### Input photo:
![Input Photo](https://github.com/KAValerio/3D-Printing-Classification/blob/main/data/image_stream/video6/img6stream20.jpg?raw=true)

Next, the Histogram of Oriented Gradients (HOG) was taken for each of the images, using the `hog`[[1]][hog] function in the `skimage.feature` package of `scikit`. The default parameters were used, with the exceptions of `pixels_per_cell = (12,12)`, `cells_per_block = (3,3)` and `block_norm="L2-Hys"`. The block norm function was chosen as it normalized the gradient blocks. This was needed as some parts of the background had very pronounced gradients which overshadowed the gradients within the photos of the prints. The output of this algorithm is a vector of gradients of the image.

![ImageProcessing](https://github.com/KAValerio/3D-Printing-Classification/blob/main/Figures/processingHOG.png?raw=true)

These vectorized images were then split into training and testing groups, split 75% and 25% respectively, using the `model_selection.train_test_split` function in the `sklearn` package.

The classification algorithm that was chosen is the `SGD Classifier`[[2]][SGD Classifier] that is part of the `sklearn.linear_model` package. This function implements regularized linear models with stochastic gradient descent learning, which pairs well with the HOG data. The loss function chosen was "log", which corresponds to a logistic regression probabilistic classifier. This was used as it provides accurate results and an estimate of probabilities. 

## Results
The table below displays the classification evaluation metrics of the SGD classification model using the random seed of 14. These values varied depending on the random seed selected, but by adjusting the image compression size, HOG parameters and model loss function, the values and methods mentioned above gave the highest consistent metrics.

|   | Prediction | Recall | F1 Score | Support|
|--|-------------|-------|-----------|----------|
| **Pass (0)**|  0.82    |  0.97   |   0.89       | 63 |
| **Fail (1)**  |  0.95   |   0.75 |     0.84       | 53|
|**Macro avg**      | 0.89 |     0.86 |      0.87  |     116|
|**Weighted avg**  |      0.88 |     0.87  |     0.87  |     116|
|**Accuracy** | 0.87|

This method provides consistently adequate results. Most of the type 1 errors correspond to images of complex prints, or messy backgrounds. Most of the type 2 errors correspond to "failing" prints that have not started to string, or for failed prints that did not produce stringing.

When tested against a video stream that contained 38 consecutive images and a failure point at image 15, the algorithm misclassified one image before failure (1 out of 15, 6.67% misclassified above failure point) and fifteen images after failure (15 out of 24, 62% misclassified above failure point). The change was detected in image 18. Most of the misclassifications occurred shortly following the failure. Once more stringing occurred, the algorithm classified the figures as errors more consistently. 

These ratios can be tuned by adjusting the HOG parameters, but these were chosen to minimize Type 1 errors as a vast majority of 3D prints are successes and do not result in any stringing and as such, it would be very annoying to receive notifications for failed prints when a print is proceeding as planned.

## Conclusion
This image classification algorithm appears to be quite successful, however, it has many type-2 errors when tested against a video feed. To improve these rates, more printing videos should be used in the training dataset. It would be beneficial to have a training dataset and prediction images that have backgrounds and the printing mechanism removed.

The input data was not perfect, especially in some frames of the video the printer filament extruder was in different locations of the pictures. To the model, this can falsely indicate a change in the images.

## Future Work
The input data appears to be a major factor, so separating the prints from the background in both the training data and predicted model could help improve accuracies. Image detection and isolation could be a useful future addition to this project.

If this model was perfected and it can detect a print failure with high accuracy, then it would also be possible to retain this model onto other machining processes. Similar manufacturing processes include CNC, milling, lathe, laser cutting, welding. Or even other areas of manufacturing such as packaging machines or liquid filling machines.

[hog]: https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog "`hog`"
[SGD Classifier]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html "SGS Classifier"
