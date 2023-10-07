#!/usr/bin/env python
# coding: utf-8

#Print Python Version
import sys
print(sys.version)

#Import Packages
#For AI
import tensorflow
import transformers as t   #import transformers library for pre-trained models

#For Data
import os
import googleapiclient.discovery

#For General Use
import pandas as pd


#Get API Key (From Local File For Security)
key = open(r"C:\Users\pc\Desktop\Work\AI\YouTube Comments Sentiment Analysis\key.txt", "r").readlines()[0]

#Get Channel ID (From Local File For Security)
channel = open(r"C:\Users\pc\Desktop\Work\AI\YouTube Comments Sentiment Analysis\channel.txt", "r").readlines()[0]

#Get Youtube Data
#Set Parameters
api_service_name = "youtube" #Use YT API
api_version = "v3"

#Initialize
youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey = key)

#Get YT Comments For Specified Channel
request = youtube.commentThreads().list(
    part="snippet,replies",
    allThreadsRelatedToChannelId=channel,   #Set channel ID
    access_token=key   #API Key
)
response = request.execute()  #Get API Output (As Dictionary and Subdictionaries)

#Convert api result to Pandas DataFrame
data = pd.DataFrame()  #Create a blank DataFrame
for x in range(len(values)):  #Loop through the data (0 --> length of the data)
    value = values[x]['snippet']['topLevelComment']['snippet']      #Get the comment
    data = pd.concat([data, pd.DataFrame(values[x]['snippet']['topLevelComment']['snippet'])],ignore_index=True)  #Append to df

del data['channelId']   #delete extra columns to keep it clean
del data ['textDisplay']
del data['authorChannelId']

model = t.pipeline(task="sentiment-analysis") #get sentiment analysis pre-trained model

labels = []  #empty list
scores = []  #empty list
for x in data['textOriginal']: #for each youtube comment
    result = model(x)[0]            #evaluate it with the pre-trained model
    labels.append(result['label'])  #append the label (postive or negative)
    scores.append(result['score'])  #append the score (0 to 1)

data['label']=labels
data['score']=scores

summary = data[['textOriginal','label','score','publishedAt','likeCount']]

negative = 0
positive = 0
for x in summary['label']:
    if x == "NEGATIVE":
        negative = negative + 1
    elif x == "POSITIVE":
        positive = positive + 1
print("Negative Comments: "+str(negative)+" ("+str(negative/(negative+postive)*100)+"%)")
print("Positive Comments: "+str(positive)+" ("+str(positive/(negative+postive)*100)+"%)")
