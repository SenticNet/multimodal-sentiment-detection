# Multi-Modal-Sentiment-Detection
Multi Modal Sentiment Detection
1.	Videos are in video folder
2.	CLM2 contains face outline from CLM software
3.	Transcripts and time intervals for each video are in folder ‘transcriptions’
4.	Perl time.pl 
5.	Extract start and end of time segments into folder ‘transcriptions3’
6.	Matlab –r readdata.m
7.	This will divide each video using segment information into folder ‘transcription2’
8.	Check the width and height of the video 
9.	Matlab –r crop_jul14.m
10.	This will crop each video and save in folder ‘cnninput’
11.	There are 10 folders for 50 videos each
12.	Matlab –r resizeall.m will reduce the resolution and duplicate the time series video
13.	Matlab –r maketrain.m divide each of the 10 files into train, val and test
14.	We can change the test files here with new dataset.
15.	Python pack_Data_vid_cv.py will pack files for deep cnn
16.	Cnninput/pack_Data_foldb.py this will pack same validation and test data
17.	cnn/convolutional_mlp_amazon.py will run deep cnn on all 10 files
18.	perl cnn_features.pl convert output to text
19.	matlab –r format.m
20.	matlab –r moud_rnn.m
21.	Output for each file is in classfmea.txt and allfmea.txt
