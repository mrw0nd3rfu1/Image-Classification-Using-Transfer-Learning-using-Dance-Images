# Prediction-Which-Dance-It-Is-using-CNN
Prediction of 8 type of dances using CNN with minimal amount of data set(350 images only).

The dataset used in this project can be found from the following link-
https://he-s3.s3.amazonaws.com/media/hackathon/hackerearth-deep-learning-challenge-identify-dance-form/identify-the-dance-form-deea77f8/0664343c9a8f11ea.zip?Signature=ZfF6il42KXDVSylaipYkUJnA7%2BE%3D&Expires=1593589777&AWSAccessKeyId=AKIA6I2ISGOYH7WWS3G5

For this project as there was such low amount of training images we have to go for transfer learning.
As we can see from above code the dance.py is me creating a cnn from scratch.
When creating CNN from scratch the accuracy is too low nearly 25% as the data set is too low even doing data augmentation the highest it went was 30%.

Then comes transfer learning-
Using transfer learning I got score above 80% using data augmentation in the process.
I used Xception pre trained weights and build my model around it.
I tried other pre trained model but Xception for me gave the best result.

Some refernce I used for this project-
https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
https://medium.com/@vijayabhaskar96/tutorial-on-keras-imagedatagenerator-with-flow-from-dataframe-8bd5776e45c1
https://towardsdatascience.com/classify-butterfly-images-with-deep-learning-in-keras-b3101fe0f98
https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
