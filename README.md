# Driver Drowsiness Detection
## Table of Content
- [Demo](#demo)
- [Overview](#overview)
- [Motivation](#motivation)
- [Technical Aspect](#technical-aspect)
- [Installation](#installation)
- [Run](#run)
- [Directory Tree](#directory-tree)
- [Technologies and Tools](#technologies-and-tools)
- [To Do](#to-do)
- [Contact](#contact)
## Demo

https://user-images.githubusercontent.com/49152921/118120917-4a0f7180-b40e-11eb-95c9-73fb23c00688.mp4


## Overview
Driver Drowsiness Detection using live video feed from the camera. This project was developed to prevent accidents caused by the drivers due to drowsiness, it will alert the driver when feeling sleepy by ringing an alarm. The CNN Model will be used to identify the whether the human eyes are closed or open with 98% Accuracy. The Haar Cascade Classifiers are used for face, left eye and right eye detection. The dataset used for this project is [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) which contains 80k images encapsulating different features in it.
## Motivation
Currently, transport systems are an essential part of human activities. We all can be victim of drowsiness while driving, simply after too short night sleep, altered physical condition or during long journeys. Which ultimately leads to dangerous situations and increase the probability of occurance of accidents. Driver drowsiness and fatigue are among the important causes of road accidents. Every year, they increase the number of deaths and fatalities injuries globally.
## Technical Aspect
This project is divided into two parts
1. Training a [Convolutional Model](https://github.com/Kirushikesh/Driver_Drowsiness_Detection/blob/main/DrowsinessModel.ipynb) to identify the human eye state.
2. [Using the Haar Cascade Classifiers and the trained model to detect the drowsiness.](https://github.com/Kirushikesh/Driver_Drowsiness_Detection/blob/main/detection.py)
## Installation
The Code is written in Python 3.7. If you don't have Python installed you should install it first. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository:
```python
pip install -r requirements.txt
```
## Run
- Run ```DrowsinessModel.ipynb``` which will train the CNN model and save it as model.h5.
- Move the trained model inside the model directory.
- Run ```detection.py``` which gets the live feed from your camera. The program analyse the feed frame by frame here here is where the cascade model comes into play they used to detect the human eyes. Followed by the CNN model to classify the eyes returned by the Cascades. When the eyes are closed till a particular time the alarm starts ringing.
## Directory Tree
![image](https://user-images.githubusercontent.com/49152921/118118893-2dbe0580-b40b-11eb-9889-8f397a49f49b.png)

## Technologies and Tools
- Python
- Tensorflow
- Keras
- OpenCV
## To Do
- Testing the models performance with different lightings.
- Deploy the model
## Contact
If you found any bug or like to raise a question feel free to contact me through [LinkedIn](https://www.linkedin.com/in/kirushikesh-d-b-10a75a169/).
If you feel this project helped you and like to encourage me for more stuffs like this please endorse my skills in my [LinkedIn](https://www.linkedin.com/in/kirushikesh-d-b-10a75a169/) thanks in advance!!!.
