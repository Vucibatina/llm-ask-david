import llama_cpp
import ctypes

# Addition for model not to dump all the verbose things during the run
LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p)
@LOG_CALLBACK
def silent_logger(user_data, level, text):
    pass  # swallow all logs
llama_cpp.llama_log_set(silent_logger, None)

import os 
import sys
import time
import argparse
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import re
import requests
import json
# import spacy
from typing import List

from llama_cpp import Llama


SAVE_SUMMARY_FILE = True

class SmartTranscriptFormatter:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.max_sentence_length = 30
        self.split_points = {
            'strong': [
                r'(?<=\w)(,\s*(?:and|but|or|because|so|which|while|when|where|if|although))',
                r'(?<=\w)(;\s*)',
                r'(?<=\w)(\.\s+)(?=[A-Z])',
            ],
            'medium': [
                r'(?<=\w)(,\s*(?:then|after|before|however|moreover|furthermore|therefore))',
                r'(?<=\w)(\s+(?:as|since|until|unless|though))',
            ],
            'weak': [
                r'(?<=\w)(,\s*(?:in|with|by|for|to))',
                r'(?<=\w)(\s+(?:through|during|among|between))',
            ]
        }

    def find_best_split_point(self, text: str) -> int:
        # Try strong split points first
        for strength in ['strong', 'medium', 'weak']:
            for pattern in self.split_points[strength]:
                matches = list(re.finditer(pattern, text))
                if matches:
                    # Find the match closest to the middle
                    middle = len(text) / 2
                    best_match = min(matches, key=lambda x: abs(x.start() - middle))
                    return best_match.start()
        
        # If no good split points found, use spaCy
        doc = self.nlp(text)
        best_split = None
        middle = len(text) / 2
        
        for token in doc:
            if token.dep_ in ['cc', 'prep'] or token.pos_ in ['VERB', 'CCONJ']:
                if best_split is None or abs(token.idx - middle) < abs(best_split - middle):
                    best_split = token.idx
        
        return best_split if best_split is not None else len(text) // 2

    def split_sentence(self, text: str) -> List[str]:
        if len(text.split()) <= self.max_sentence_length:
            return [text]
        
        split_point = self.find_best_split_point(text)
        first_half = text[:split_point].strip()
        second_half = text[split_point:].strip()
        
        # Ensure proper capitalization and punctuation
        if not first_half.endswith(('.', '!', '?')):
            first_half += '.'
        if second_half.startswith((',', ';')):
            second_half = second_half[1:].strip()
        second_half = second_half[0].upper() + second_half[1:]
        
        # Recursively split if still too long
        result = []
        result.extend(self.split_sentence(first_half))
        result.extend(self.split_sentence(second_half))
        return result

    def format_transcript(self, text: str) -> str:
        # Basic cleanup
        text = re.sub(r'\b([a-zA-Z])\s+(?:\1\s+)+', '', text)
        text = ' '.join(text.split())
        
        # Split into sentences using spaCy
        doc = self.nlp(text)
        formatted_sentences = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text.split()) > self.max_sentence_length:
                formatted_sentences.extend(self.split_sentence(sent_text))
            else:
                formatted_sentences.append(sent_text)
        
        # Format paragraphs and final cleanup
        text = ' '.join(formatted_sentences)
        text = re.sub(r'(?<=[.!?])\s+(?=[A-Z][^.])', '\n\n', text)
        return text.strip()

# Usage
def format_text(input_text: str) -> str:
    formatter = SmartTranscriptFormatter()
    return formatter.format_transcript(input_text)

class TranscriptFormatter:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.max_sentence_length = 30
        self.filler_words = r'\b(um|uh|like|you know|I mean|sort of|kind of|basically)\b'
        self.repeated_words = r'\b(\w+)(\s+\1)+\b'

    def split_long_sentence(self, sentence: str) -> List[str]:
        doc = self.nlp(sentence)
        words = len(doc)
        
        if words <= self.max_sentence_length:
            return [sentence]
            
        sentences = []
        current = []
        word_count = 0
        
        for token in doc:
            current.append(token.text)
            word_count += 1
            
            # Check for natural break points
            if (word_count >= self.max_sentence_length and 
                (token.dep_ in ['cc', 'prep'] or token.pos_ in ['VERB', 'CCONJ'])):
                sentences.append(' '.join(current))
                current = []
                word_count = 0
        
        if current:
            sentences.append(' '.join(current))
            
        # Add proper punctuation
        return [s.strip() + '.' if not s.strip().endswith('.') else s.strip() 
                for s in sentences]

    def format_transcript(self, text: str) -> str:
        # Basic cleanup
        text = re.sub(r'\b([a-zA-Z])\s+(?:\1\s+)+', '', text)
        text = re.sub(self.repeated_words, r'\1', text)
        text = re.sub(self.filler_words, '', text, flags=re.IGNORECASE)
        text = ' '.join(text.split())
        
        # Split into sentences
        doc = self.nlp(text)
        formatted_sentences = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text.split()) > self.max_sentence_length:
                formatted_sentences.extend(self.split_long_sentence(sent_text))
            else:
                formatted_sentences.append(sent_text)
        
        # Join sentences and format paragraphs
        text = ' '.join(formatted_sentences)
        text = re.sub(r'(?<=[.!?])\s+(?=[A-Z][^.])', '\n\n', text)
        
        return text.strip()

def format_text_preserving(text):
    """Format text while preserving every word exactly"""
    
    # Store original words
    original_words = text.split()
    word_index = 0
    formatted_sentences = []
    current_sentence = []
    
    while word_index < len(original_words):
        word = original_words[word_index]
        current_sentence.append(word)
        
        # Check for sentence boundaries
        if (word_index + 1 < len(original_words) and 
            (original_words[word_index + 1][0].isupper() or  # Next word starts with capital
             any(word.lower().endswith(end) for end in ['ok', 'okay', 'right', 'yes', 'no']) or  # Common endings
             any(original_words[word_index + 1].lower() == starter for starter in 
                 ['then', 'but', 'however', 'therefore', 'thus', 'hence', 'so', 'anyway', 
                  'finally', 'later', 'next', 'first', 'second', 'third', 'last'])  # Transition words
            )):
            sentence = ' '.join(current_sentence)
            # Capitalize first word
            sentence = sentence[0].upper() + sentence[1:]
            # Add period if no ending punctuation
            if not sentence[-1] in '.!?':
                sentence += '.'
            formatted_sentences.append(sentence)
            current_sentence = []
        
        word_index += 1
    
    # Handle last sentence
    if current_sentence:
        sentence = ' '.join(current_sentence)
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence[-1] in '.!?':
            sentence += '.'
        formatted_sentences.append(sentence)
    
    # Group into paragraphs (3-4 sentences per paragraph)
    paragraphs = []
    current_para = []
    
    for i, sentence in enumerate(formatted_sentences):
        current_para.append(sentence)
        if len(current_para) >= 4 or i == len(formatted_sentences) - 1:
            paragraphs.append(' '.join(current_para))
            current_para = []
    
    return '\n\n'.join(paragraphs)


# Usage example
def format_file(input_path: str, output_path: str):
    formatter = TranscriptFormatter()
    
    with open(input_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    formatted = formatter.format_transcript(text)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(formatted)
    
    return formatted


youtube_api_key = "AIzaSyBLhrvOhdxYc5hApA1SaAK7cPamY0S1uSA"
data_store_directory ="/Users/vuk/projects/data"

# Create a YouTube Data API client
youtube = build('youtube', 'v3', developerKey=youtube_api_key)


def get_youtube_channel_id(api_key, username):
    # Removing the '@' sign
    clean_username = username.lstrip('@')
    
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&type=channel&q={clean_username}&key={api_key}"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to retrieve the channel info. Status Code: {response.status_code}")
        return None, None
    
    data = response.json()
    
    # Check if we found any channels in the response
    if "items" in data and len(data["items"]) > 0:
        channel_id = data["items"][0]["snippet"]["channelId"]
        channel_name = data["items"][0]["snippet"]["title"]  # Correcting the field for the channel name
        return channel_id, channel_name
    else:
        print(f"No channels found for username: {username}")
        return None, None

def save_transcript_as_file(directory, video_title, file_name, video_id, video_url, transcript, username, channel_id, channel_name, video_publish_date):
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    filename = file_name
    
    # Create the full file path
    file_path = os.path.join(directory, filename)
    
    # Format the date as YYYY-MM-DD
    formatted_date = video_publish_date.split('T')[0] if 'T' in video_publish_date else video_publish_date
    
    # Open the file for writing
    with open(file_path, "w", encoding='utf-8') as file:
        # Write each string to the file with the new metadata fields
        file.write("Channel ID: " + username + "\n")
        file.write("Channel Internal youtube ID: " + channel_id + "\n")
        file.write("Channel Name: " + channel_name + "\n")
        file.write("Date Produced: " + formatted_date + "\n")
        file.write("Title: " + video_title + "\n")
        file.write("File Name: " + file_name + "\n")
        file.write("Video Id: " + video_id + "\n")
        file.write("Video URL: " + video_url + "\n\n")
        file.write(transcript + "\n")
        
    print(f"Transcript written to file {filename} successfully.")

def get_transcript(video_id):
    full_transcript = ''
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = ' '.join([segment['text'] for segment in transcript])

        # ✅ Also save raw JSON with timings
        raw_json_path = os.path.join(data_store_directory, f"{video_id}_raw.json")
        with open(raw_json_path, "w", encoding="utf-8") as raw_json_file:
            json.dump(transcript, raw_json_file, indent=2)

        return full_transcript
    
    except Exception as e:
        print(f"Cannot get transcript for {video_id} - an error occurred: {str(e)}")

    return full_transcript

def get_channel_videos(channel_id, api_key):
    # Create a YouTube Data API client
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Initialize an empty list to store the video data
    videos = []

    # Make the initial request to retrieve the first page of videos
    next_page_token = None
    while True:
        # Make the API request
        search_response = youtube.search().list(
            channelId=channel_id,
            type='video',
            part='id,snippet',
            maxResults=50,
            order='date',
            pageToken=next_page_token
        ).execute()

        # Extract the videos from the search response
        videos.extend(search_response.get('items', []))

        # Check if there is a next page token
        next_page_token = search_response.get('nextPageToken')

        # If there are no more pages, break out of the loop
        if not next_page_token:
            break

    return videos


def get_transcript_summary(full_transcript):
    max_context_tokens = 2048
    max_response_tokens = 800
    max_prompt_tokens = max_context_tokens - max_response_tokens

    # Fixed system instruction
    prompt_intro = (
        "This is a transcript from a video. I need you to summarize with maximum 300 words what this video is about. "
        "Primary focus is practicality of what it suggests to be done, second priority is the theory and science:\n\n"
    )

    # Safe word-based truncation
    intro_words = prompt_intro.split()
    transcript_words = full_transcript.split()
    allowed_transcript_words = max_prompt_tokens - len(intro_words)
    safe_transcript = " ".join(transcript_words[:allowed_transcript_words])
    full_text = prompt_intro + safe_transcript
    full_prompt = prompt_intro + safe_transcript

    llm = Llama(
        model_path="llm_models/Yi-1.5-9B-Chat-Q4_K_M.gguf",
        n_ctx=max_context_tokens,
        n_threads=8,
        temperature=0.7,
        chat_format="chatml"  # Use the correct chat template
    )

    output = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": full_text}
        ],
        max_tokens=max_response_tokens
    )

    return output["choices"][0]["message"]["content"].strip()


#########################  MAIN FUNCTION  ###################################
#
# Takes paramers and it can be run like this
#    python transcript_loader.py --path "/Users/vuk/projects/data" --username "DavidSnyderNLP"
#
#############################################################################
#########################  MAIN FUNCTION  ###################################
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Download YouTube transcripts from a channel')
    parser.add_argument('--path', '-p', type=str, required=True, help='Directory path where transcripts will be saved')
    parser.add_argument('--username', '-u', type=str, required=True, help='YouTube channel username')
    
    SLEEP_TIME = 1
    # Parse arguments
    args = parser.parse_args()
    
    # Use the arguments
    file_path = args.path
    username = args.username
    name_option = 2  # This remains hardcoded as it's not a parameter
    
    # Fetch the channel ID and name
    channel_id, channel_name = get_youtube_channel_id(youtube_api_key, username)
    if channel_id:
        print(f"Channel ID for {username} is: {channel_id}")
        print(f"Channel Name: {channel_name}")
    else:
        print(f"Could not find channel ID for {username}")
        sys.exit(1)  # Exit if channel not found
    
    # Use the provided file_path directly without creating a subdirectory
    transcript_folder = file_path
    
    # Create the directory if it doesn't exist
    if not os.path.exists(transcript_folder):
        os.makedirs(transcript_folder)
    
    videos = get_channel_videos(channel_id, youtube_api_key)
    # Print the total number of videos
    print(f"Total videos: {len(videos)}")
    
    master_transcript = ''
    transcribeable = 0

    # Before starting to loop over videos, load already transcribed video IDs
    already_transcribed_ids = set()
    for filename in os.listdir(transcript_folder):
        if filename.endswith('.txt'):
            video_id = filename[:-4]  # Remove the '.txt' extension
            already_transcribed_ids.add(video_id)

    # Print the video titles and URLs
    for video in videos:
        video_title = video['snippet']['title']
        video_url = f"https://www.youtube.com/watch?v={video['id']['videoId']}"
        video_id = video['id']['videoId']
        
        # Check if already transcribed
        if video_id in already_transcribed_ids:
            print(f"Video {video_id} already transcribed and saved. Skipping.")
            continue  # Skip this video

        print(f"Title: {video_title}")
        print(f"ID: {video_id}")
        print(f"URL: {video_url}")
        
        # full_transcript_raw = get_transcript(video_id)
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            # Save raw JSON file
            raw_json_path = os.path.join(transcript_folder, f"{video_id}_raw.json")
            with open(raw_json_path, "w", encoding="utf-8") as raw_json_file:
                json.dump(transcript, raw_json_file, indent=2)
            full_transcript_raw = ' '.join([segment['text'] for segment in transcript])
        except Exception as e:
            print(f"Cannot get transcript for {video_id} - an error occurred: {str(e)}")
            continue  # Skip this video and move on

        time.sleep(SLEEP_TIME)
        full_transcript = format_text_preserving(full_transcript_raw)
        video_summary = get_transcript_summary(full_transcript)
        if SAVE_SUMMARY_FILE:
            # Save summary to a JSON file
            summary_json_path = os.path.join(transcript_folder, f"{video_id}_summary.json")
            summary_data = {
                "video_id": video_id,
                "video_title": video_title,
                "video_url": video_url,
                "channel_id": channel_id,
                "channel_name": channel_name,
                "username": username,
                "publish_date": video['snippet']['publishedAt'],
                "summary": video_summary
            }
            
            with open(summary_json_path, "w", encoding="utf-8") as summary_json_file:
                json.dump(summary_data, summary_json_file, indent=2)
            
            print(f"Summary saved to {summary_json_path}")

        print("\nVIDEO SUMMARY: ")
        print(video_summary)
        
        # Using video ID for filename
        file_name = f"{video_id}.txt"
        
        # Save transcript to the folder
        video_publish_date = video['snippet']['publishedAt']
        save_transcript_as_file(
            transcript_folder, 
            video_title, 
            file_name, 
            video_id, 
            video_url, 
            full_transcript,
            username,
            channel_id,
            channel_name,
            video_publish_date
        )

        print(f"Transcript available: {len(full_transcript)>0}")
        words = re.findall(r'\b\w+\b', full_transcript)
        num_words = len(words)
        print(f"Word count: {num_words}")
        print("---")
        
        # Add to master transcript
        if len(full_transcript) > 0:
            master_transcript += '\n\n' + video_title + '\n' + full_transcript 
            transcribeable += 1
    
    print(f"Total videos: {len(videos)}")
    print(f" - Transcribeable: {transcribeable}")
    print(f"MASTER TRANSCRIPT LENGTH: {len(master_transcript)}")
    
if __name__ == "__main__":
    main()