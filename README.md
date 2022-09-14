# asl-recognition
A Computer Vision and Deep Learning approach to [American Sign Language](https://en.wikipedia.org/wiki/American_Sign_Language)(ASL) Detection

## Motivation behind project
I was inspired by the amazing Oscar winning movie [CODA](https://en.wikipedia.org/wiki/CODA_(2021_film)) to pursue this project. By working on such a complex Machine Learning and Computer Vision problem I wanted to understand more about the challenges faced by the CODA community and in general the nuances of American Sign Language.

The main aim is to solve the problem of translating ASL to English alphabets. This is achieved by processing the ASL input data in two different ways and comparing results between 4 common Convolutional Neural Networks - [VGG16](https://www.mygreatlearning.com/blog/introduction-to-vgg16/), [ResNet 50 V2](https://blog.devgenius.io/resnet50-6b42934db431), [Inception V3](https://en.wikipedia.org/wiki/Inceptionv3), [MobileNet V2](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html)

## Data
The data is obtained from a Kaggle dataset called [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). It is a collection of images of alphabets from ASL.

The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING.

I have only included the link to the dataset here as it is too big for this repository.

## Convex Hull Approach
Active contour is a segmentation method that uses energy forces and constraints to separate the pixels of interest from a picture for further processing and analysis. Active contour is defined as an active model for the segmentation process. 

Contours are the boundaries that define the region of interest in an image. A
contour is a collection of points that have been interpolated. The interpolation procedure might be linear, splines, or polynomial, depending on how the curve in the image is described. 

Snakes provide a unified account of a number of visual problems, including detection of edges, lines, and subjective contours;motion tracking; and stereo matching.

Contours can be found using OpenCV’s findContours() function. Before doing this it's necessary to binarize the image for better results. This can be done either by Otsu’s thresholding or adaptive thresholding

Once all the contours are found we find the contour with maximum area and bound it within a rectangle.
Then use convex hull to get the polygon that encompasses the area

This approach is coded in the file convex_hull.py

## CNN Approach
Since Convex Hull processes images in real time for gestures it takes a lot of processing time which can cause the gesture recognition process to be slow or redundant.

Due to the above limitations of the Convex Hull algorithm I decided to use convolutional neural networks (CNN’s) which are quite commonly used and known to perform well. 

I will compare the model performance and accuracies for the particular sign language dataset and gain some insights into the results of model selection and training.

The CNN approach is coded(trained/tested and compared) in the asl_neuralnetworks Google Colab Notebook.

## Results

| Model Name    | No Threholding | Thresholding 1 Accuracy | Thresholding 2 Accuracy |
| ----------- | ----------- | ---------- | ---------|
| VGG16      | 90.62%       | 83.59%          | 90.62% |  
| MobileNet V2   | 98.44%        | 92.19%          | 88.28 |
| ResNet50 V2      | 81.25%       | 69.53%          | 78.91% | 
| Inception V3      | 70.31%       | 76.56%          | 87.5% | 

## Conclusion & Future Work
- We can conclude that VGG16 and MobileNet50 perform both faster and better as compared to other models. The models take more time to train and converge if they are binarized. This was expected as the binarized images could not retain some details.

- The proposed work is for Sign/Alphabet detection only and it currently doesn’t recognize
gestures. For gesture recognition, alogorithms/models like Optical Flow and others are required which would be able to track the hand movement. A large corpus of videos data would also be needed.

- For now worked with only American Sign Language, however there are a vast variety of sign
languages such as British Sign Language(BSL), French Sign Language(LSF), Brazilian Sign
Language(Libras) and more. While a unified model and translation between various languages
would be challenging it has an interesting scope for further research.



