# SpeechGuild: A Speech Emotion Recognition System

<div text-align = "justify">

A Mini Project performed in Semester 5 in a team of four:<br>
•	Shruti Jain 1902057 <br>
•	Nimit Jhunjhunwala 1902061 <br> 
•	Mitali Juvekar 1902064 <br>
•	Tanay Kamath 1902068 <br>

The project, as the name suggests aids us in analysing an emotion of an audio file, which are distinguished as neutral, calm, happy, sad, angry, fearful, surprise and disgust. The audio file is segregated based on the maximum percentage of emotion it has. The user interface for it is provided via a website which deploys the Convolution Neural Network (CNN) model, made with Django framework.<br>
RAVDESS dataset has been used to train a CNN model. The audio files can either be recorded in real time or by uploading .wav files, they can be analysed by the model. The model then segregates the audiofile and stores them in one of the folders in the “voice-box”. The accuracy of the model is 68% without data augmentation. The audio files are understood with the help of Librosa and the feature extraction is done by MFCC. This is system is basically used to overcome the gap of acoustic characteristics of human voice to distinguish between emotions clearly by the model.<br>

### Dataset Details<br>
[Dataset: RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)<br>
RAVDESS contains 1440 files: 60 trials per actor x 24 actors = 1440. The RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech emotions includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.<br>

#### File naming convention<br>
Each of the 1440 files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav). These identifiers define the stimulus characteristics:<br>

#### Filename identifiers<br>
•	Modality (01 = full-AV, 02 = video-only, 03 = audio-only).<br>
•	Vocal channel (01 = speech, 02 = song).<br>
•	Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).<br>
•	Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.<br>
•	Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").<br>
•	Repetition (01 = 1st repetition, 02 = 2nd repetition).<br>
•	Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).<br>

Filename example: 03-01-06-01-02-01-12.wav<br>
1.	Audio-only (03)
2.	Speech (01)
3.	Fearful (06)
4.	Normal intensity (01)
5.	Statement "dogs" (02)
6.	1st Repetition (01)
7.	12th Actor (12)
Female, as the actor ID number is even.

#### Dataset Preview:
![Dataset Preview](https://github.com/NimitJhunjhunwala/SpeechGuild/blob/master/dataset%20preview.png)

### Processing Details
#### Exploratory Data Analysis (EDA)
The data consists audio files of different emotions for two genders namely, male and female, so, wave plot of 2 emotions (happy and angry) for both the genders has been visualized along with their audio file.
Snippets of the waveplots are shown below:
<div align = "center">

![Wave plot of Male - Happy](https://github.com/NimitJhunjhunwala/SpeechGuild/blob/master/wp%20male%20happy.png)<br>
Wave plot of Male - Happy<br>

![Wave plot of Female - Happy](https://github.com/NimitJhunjhunwala/SpeechGuild/blob/master/wp%20female%20happy.png)<br>
Wave plot of Female - Happy<br>

![Wave plot of Male - Angry](https://github.com/NimitJhunjhunwala/SpeechGuild/blob/master/wp%20male%20angry.png)<br>
Wave plot of Male - Angry<br>

![Wave plot of Female - Angry](https://github.com/NimitJhunjhunwala/SpeechGuild/blob/master/wp%20female%20angry.png)<br>
Wave plot of Female - Angry<br>
</div>

#### Feature Extraction 
Mel Frequency Cepstral Coefficient (MFCC) has been used for feature extraction. MFCC represents speech signal as the short-term power spectrum of sound, based on linear cosine transform of a log power spectrum on a nonlinear Mel scale of frequency. A mel is a unit of measure of perceived pitch or frequency of a signal. MFCC gives better frequency resolution in low frequency region. Hence, it can be applicable for all types of signals and it is not affected by the noise.
So, the mean bands are extracted to its own feature column, NA values are replaced with 0 and thus the data is converted into a workable format.

### Model Details
The model used to train the was Convolution Neural Network (CNN). The data which is preprocessed is kept in a dataset. The dataset is then split into two parts, training and testing data in the ratio of 0.75:0.25 respectively. Data normalization is then performed using mean and std methods of the numpy library which improves accuracy and speeds up the training process. The data is then converted into a numpy array since Keras is to be used, which is followed by one hot encoding of the data. Now, one of the most important steps is to specify a 3rd dimension since CNN is being used to train the model, in our case, the extra dimension is 1 because 1D CNN is performed first along with Adam optimizer. All the functions are then repeated and the data is then fed into 2D CNN Model. The 2D CNN takes in a 2D array of 30 MFCC bands by 216 audio lengths as input data. so just imagine it as a 30 x 216-pixel image.

### Snippets of output
#### Website Landing Page
The home page the user is greeted with after opening the website. This page states features of the website in one line. “Get started” navigates to the Record Page<br>
 <div align = "center">
 
 ![Website Landing Page](https://github.com/NimitJhunjhunwala/SpeechGuild/blob/master/website%20landing%20page.png) 
 </div>

#### Record Page 
The record page is the feature page. This is where the audio is analyzed to check for the emotions. The audio can be either recorded in real time or uploaded from the device which is then analyzed and sent to the voice box page w.r.t. their emotions.<br>
 <div align = "center"> 
 
 ![Record Page](https://github.com/NimitJhunjhunwala/SpeechGuild/blob/master/record%20page.png) 
 </div>

#### Voice Box Page 
The voice box page consists folders of all the 8 emotions which would contain 29 the segregated audio files as a result.<br>
<div align = "center"> 
 
![Voice Box Page](https://github.com/NimitJhunjhunwala/SpeechGuild/blob/master/voice%20box%20page.png)
</div><br>

Working of the Website The working involves 3 activities i.e. the two features in the Record Page – Record Audio and Upload a Clip, and Post Analysis and Segregation - Audio file in a folder in the Voice Box Page w.r.t. the maximum emotion.

#### a. Record Audio 
After clicking on the “Record a Clip” icon<br>
<div align = "center"> 

![Record Audio](https://github.com/NimitJhunjhunwala/SpeechGuild/blob/master/record%20audio.png)
</div>

After successfully recording an audio clip<br>
<div align = "center">

![Analyse Pop Up](https://github.com/NimitJhunjhunwala/SpeechGuild/blob/master/analyse.png)
</div>

Click on “Analyse” to move on to the Post Analysis and Segregation stage<br>

#### b. Upload an Audio file 
After clicking on the “Upload a Clip” icon<br>
<div align = "center"> 

![Upload an Audio file ](https://github.com/NimitJhunjhunwala/SpeechGuild/blob/master/upload%20audio.png)
</div>

After successfully uploading an audio clip<br>
<div align = "center"> 

![Analyse Pop Up](https://github.com/NimitJhunjhunwala/SpeechGuild/blob/master/analyse.png)
</div>

Click on “Analyse” to move on to the Post Analysis and Segregation stage<br>

#### Post Analysis and Segregation 
Audio file in a folder in the Voice Box Page w.r.t. the maximum emotion.<br>
<div align = "center"> 

![Post Analysis and Segregation](https://github.com/NimitJhunjhunwala/SpeechGuild/blob/master/post%20analysis%20and%20segregation.png)
</div><br>

</div>
