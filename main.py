#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Print Python Version
import sys
print(sys.version)


# In[ ]:


#Import Packages
#For AI
import transformers as t   #import transformers library for pre-trained models
import tensorflow

#For Data
import os
import googleapiclient.discovery

#For General Use
import pandas as pd


# In[ ]:


#Get Youtube Data
#Set Parameters
def callAPI(key,channel,pageToken=""):
    api_service_name = "youtube" #Use YT API
    api_version = "v3"

    #Initialize
    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey = key)

    #If no page token provided (first api call run)
    if pageToken=="":
        #Get YT Comments For Specified Channel
        request = youtube.commentThreads().list(
            part="snippet,replies",
            maxResults=10,
            allThreadsRelatedToChannelId=channel,   #Set channel ID
            access_token=key   #API Key
        )
    else: #If page token provided
        #Get YT Comments For Specified Channel
        request = youtube.commentThreads().list(
            part="snippet,replies",
            maxResults=10,
            pageToken=pageToken,
            allThreadsRelatedToChannelId=channel,   #Set channel ID
            access_token=key   #API Key
        )
    response = request.execute()  #Get API Output (As Dictionary and Subdictionaries)
    return response


# In[ ]:


#Convert api result to Pandas DataFrame
def convertToDF(values):
    data = pd.DataFrame()  #Create a blank DataFrame
    for x in range(len(values)):  #Loop through the data (0 --> length of the data)
        value = values[x]['snippet']['topLevelComment']['snippet']      #Get the comment
        data = pd.concat([data, pd.DataFrame(values[x]['snippet']['topLevelComment']['snippet'])],ignore_index=True)  #Append to df
    return data


# In[ ]:


#Get YouTube Comments Data (Abstracts details)
def getData(key,channel):
    data = pd.DataFrame()
    response = callAPI(key,channel)  #Call API for the first time
    while (True):
        try:                                       #Try
            nextPage = response['nextPageToken']      #to get the next page token (for the next page of data)
        except KeyError:                           #If there is no next page token (last page)
            break                                     #then break out of the loop
        response = callAPI(key,channel,nextPage)   #Get the data for the next page
        newData = convertToDF(response['items'])   #Convert the new data to a data frame
        data = pd.concat([data, newData],ignore_index=True) #Append it to our result
    return data


# In[ ]:


#GET DATA
#Get API Key (From Local File For Security)
key = open(r"C:\Users\pc\Desktop\Work\AI\YouTube Comments Sentiment Analysis\key.txt", "r").readlines()[0]

#Get Channel ID (From Local File For Security)
channel = open(r"C:\Users\pc\Desktop\Work\AI\YouTube Comments Sentiment Analysis\channel.txt", "r").readlines()[0]

#Get Data
data = getData(key, channel)  #Get a list of all youtube comments from the channel specified


# In[ ]:


#CLEAN DATA
del data['channelId']   #delete extra columns to keep it clean
del data ['textDisplay']
del data['authorChannelId']


# In[ ]:


data = data.rename(columns={'textOriginal':'comment'})


# In[ ]:


#LOAD NLP MODEL
model = t.pipeline(task="sentiment-analysis") #get sentiment analysis pre-trained model


# In[ ]:


#EVALUATE SENTIMENT
labels = []  #empty list
scores = []  #empty list
for x in data['comment']: #for each youtube comment
    result = model(x)[0]            #evaluate it with the pre-trained model
    labels.append(result['label'])  #append the label (postive or negative)
    scores.append(result['score'])  #append the score (0 to 1)


# In[ ]:


#ADD SENTIMENT TO DATA
data['sentiment']=labels  #Add label (sentiment) column
data['score']=scores  #Add numeric score from the model


# In[ ]:


summary = data[['comment','sentiment','score','publishedAt','likeCount']]


# In[ ]:


#SUMMARIZE RESULTS
#Initialize Counts
negative = 0       #Negative comments
positive = 0       #Positive comments
likedNegative = 0  #Negative comments that were liked
likedPositive = 0  #Positive comments that were liked

for x in range(len(summary)):  #calculate result then track the result
    row = summary.loc[x]
    if row['sentiment'] == "NEGATIVE":            #If negative
        negative +=1                                 #Increment
        if row['likeCount'] > 0:                     #If liked and negative
            likedNegative +=1                          #Increment 
    elif row['sentiment'] == "POSITIVE":          #If positive
        positive +=1                                 #Increment
        if row['likeCount'] > 0:                     #If liked and positive
            likedPositive +=1                          #Increment
        
#Print Results
print("Total Comments: "+str(len(summary)))
print()
print("Negative Comments: "+str(negative)+" ("+str(negative/(negative+positive)*100)+"%)")
print("Negative and Liked Comments: "+str(likedNegative)+" ("+str(likedNegative/(negative)*100)+"%)")
print()
print("Positive Comments: "+str(positive)+" ("+str(positive/(negative+positive)*100)+"%)")
print("Positive and Liked Comments: "+str(likedPositive)+" ("+str(likedPositive/(positive)*100)+"%)")
