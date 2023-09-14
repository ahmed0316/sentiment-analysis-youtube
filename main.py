#For AI
import transformers
import tensorflow

#For Data
import os
import googleapiclient.discovery

#For General Use
import pandas as pd

#Get Youtube Data
def getYoutubeData(channelID, apiKey):
  api_service_name = "youtube"  #Use the youtube google api
  api_version = "v3"            
  DEVELOPER_KEY = apiKey  #API key
  
  youtube = googleapiclient.discovery.build(
      api_service_name, api_version, developerKey = DEVELOPER_KEY)
  
  request = youtube.commentThreads().list(     #Get list of all comments on the specified yt channel
      part="snippet,replies",
      allThreadsRelatedToChannelId=channelID,     #Channel id of the youtube channel
      access_token=apiKey                         #API key
  )
  response = request.execute()

rawData = getYouTubeData("x")  #Returns a list of all youtube comments for the channel

#Clean the raw data
#(Test):#
import json
json.loads(str(response))
f = open("C:/Users/pc/Desktop/youtube_extract.json", "w")
f.write(request)
f.close()
x = pd.DataFrame.from_dict(response['items'])
pd.DataFrame.read_json(x['snippet'])
pd.concat([x, x['snippet'].str.split(',', expand=True)], axis=1)
