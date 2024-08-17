import googleapiclient.discovery

def get_youtube_video_url(search_keyword):
    api_key = ''

    youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)
    search_response = youtube.search().list(
        q=search_keyword,
        type='video',
        part='id,snippet',
        maxResults=1  
    ).execute()

  
    video_id = search_response['items'][0]['id']['videoId']

 
    video_url = f'https://www.youtube.com/watch?v={video_id}'

    return video_url
