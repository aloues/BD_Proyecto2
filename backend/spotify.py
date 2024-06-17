import base64
import json
import os

from dotenv import load_dotenv
from requests import post, get

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")


def get_token():
  auth_string = f"{client_id}:{client_secret}"
  auth_bytes = auth_string.encode('utf-8')
  auth_base64 = str(base64.b64encode(auth_bytes), 'utf-8')

  url = "https://accounts.spotify.com/api/token"
  headers = {
    "Authorization": f"Basic {auth_base64}",
    "Content-Type": "application/x-www-form-urlencoded"
  }

  data = {
    "grant_type": "client_credentials"
  }

  result = post(url, headers=headers, data=data)
  json_result = json.loads(result.content)
  token = json_result['access_token']
  return token

def get_auth_header(token):
  return {
    "Authorization": f"Bearer {token}"
  }

def get_track_info(token, track_id):
  headers = get_auth_header(token)
  url = f"https://api.spotify.com/v1/tracks/{track_id}"
  result = get(url, headers=headers)
  return json.loads(result.content)

def simplify_track_info(track_info):
  return {
    "name": track_info['name'],
    "artists": [artist['name'] for artist in track_info['artists']],
    "album": track_info['album']['name'],
    "preview_url": track_info['preview_url'] if track_info['preview_url'] else "",
    "album_image": track_info['album']['images'][0]['url'] if track_info['album']['images'] else ""
  }
