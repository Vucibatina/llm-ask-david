# transcript_loader.py

import ctypes
import os
import sys
import time
import argparse
import re
import json
import requests
from typing import List

# Silence llama.cpp logs
import llama_cpp
LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p)
@LOG_CALLBACK
def silent_logger(user_data, level, text):
    pass
llama_cpp.llama_log_set(silent_logger, None)

# YouTube Transcript API (>= 1.2.x instance API)
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)

# YouTube Data API
from googleapiclient.discovery import build

# LLM summarization
from llama_cpp import Llama


# ------------------------------- Config ------------------------------------

SAVE_SUMMARY_FILE = True
DEFAULT_SLEEP_BETWEEN_VIDEOS_S = 1

# Prefer env var if set, else fallback to your existing key
youtube_api_key = os.getenv("YT_API_KEY", "AIzaSyBLhrvOhdxYc5hApA1SaAK7cPamY0S1uSA")

# Model used for summarization
SUMMARY_MODEL_PATH = os.getenv("SUMMARY_MODEL_PATH", "llm_models/Yi-1.5-9B-Chat-Q4_K_M.gguf")
SUMMARY_MAX_CONTEXT = 2048
SUMMARY_MAX_RESPONSE = 800

# Initialize shared objects
ytt = YouTubeTranscriptApi()
youtube = build("youtube", "v3", developerKey=youtube_api_key)


# ----------------------------- Utilities -----------------------------------

def get_youtube_channel_id(api_key: str, username: str):
    """Return (channel_id, channel_name) for a given username or handle."""
    clean_username = username.lstrip('@')
    url = (
        "https://www.googleapis.com/youtube/v3/search"
        f"?part=snippet&type=channel&q={clean_username}&key={api_key}"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Failed to retrieve the channel info. Status Code: {resp.status_code}")
        return None, None

    data = resp.json()
    if "items" in data and len(data["items"]) > 0:
        channel_id = data["items"][0]["snippet"]["channelId"]
        channel_name = data["items"][0]["snippet"]["title"]
        return channel_id, channel_name

    print(f"No channels found for username: {username}")
    return None, None


def get_channel_videos(channel_id: str, api_key: str):
    """Return all videos for a channel (newest first)."""
    vids = []
    next_page_token = None
    while True:
        sr = youtube.search().list(
            channelId=channel_id,
            type="video",
            part="id,snippet",
            maxResults=50,
            order="date",
            pageToken=next_page_token
        ).execute()

        vids.extend(sr.get("items", []))
        next_page_token = sr.get("nextPageToken")
        if not next_page_token:
            break
        # tiny delay to be nice to the API
        time.sleep(0.1)

    return vids


def save_transcript_as_file(
    directory: str,
    video_title: str,
    file_name: str,
    video_id: str,
    video_url: str,
    transcript_text: str,
    username: str,
    channel_id: str,
    channel_name: str,
    video_publish_date: str,
):
    """Save a human-readable .txt with header metadata + transcript text."""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, file_name)

    # Format the date as YYYY-MM-DD
    formatted_date = video_publish_date.split("T")[0] if "T" in video_publish_date else video_publish_date

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Channel Username: " + username + "\n")
        f.write("Channel ID: " + channel_id + "\n")
        f.write("Channel Name: " + channel_name + "\n")
        f.write("Date Produced: " + formatted_date + "\n")
        f.write("Title: " + video_title + "\n")
        f.write("File Name: " + file_name + "\n")
        f.write("Video Id: " + video_id + "\n")
        f.write("Video URL: " + video_url + "\n\n")
        f.write(transcript_text + "\n")

    print(f"Transcript written to file {file_name} successfully.")


def fetch_transcript_text(video_id: str, save_dir: str = None, preferred=('en', 'en-US', 'en-GB')) -> str:
    """
    Using youtube-transcript-api >= 1.2.x:
      - Prefer manually created transcripts in preferred langs
      - Fallback to auto-generated
      - Translate to English if possible
      - Save raw JSON (if save_dir provided)
      - Return flattened text string
    """
    try:
        tl = ytt.list(video_id)  # TranscriptList

        # Prefer manual
        try:
            t = tl.find_manually_created_transcript(preferred)
        except NoTranscriptFound:
            # Fallback to any matching language (often auto-generated)
            t = tl.find_transcript(preferred)

        # Translate to English if we can and it's not already in preferred
        try:
            if getattr(t, "language_code", None) not in preferred and getattr(t, "is_translatable", False):
                t = t.translate('en')
        except Exception:
            pass

        fetched = t.fetch()  # iterable of segment objects
        segments = [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            raw_json_path = os.path.join(save_dir, f"{video_id}_raw.json")
            with open(raw_json_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, indent=2)

        return " ".join(seg["text"] for seg in segments)

    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript, VideoUnavailable) as e:
        print(f"Cannot get transcript for {video_id} - {type(e).__name__}: {e}")
    except Exception as e:
        print(f"Cannot get transcript for {video_id} - Unexpected error: {e}")

    return ""


def get_transcript_summary(full_transcript: str) -> str:
    """Summarize transcript text using a local GGUF model via llama-cpp-python."""
    max_context_tokens = SUMMARY_MAX_CONTEXT
    max_response_tokens = SUMMARY_MAX_RESPONSE
    max_prompt_tokens = max_context_tokens - max_response_tokens

    prompt_intro = (
        "This is a transcript from a video. I need you to summarize with maximum 300 words what this video is about. "
        "Primary focus is practicality of what it suggests to be done, second priority is the theory and science:\n\n"
    )

    intro_words = prompt_intro.split()
    transcript_words = full_transcript.split()
    allowed_transcript_words = max_prompt_tokens - len(intro_words)
    safe_transcript = " ".join(transcript_words[:max(0, allowed_transcript_words)])

    full_text = prompt_intro + safe_transcript

    llm = Llama(
        model_path=SUMMARY_MODEL_PATH,
        n_ctx=max_context_tokens,
        n_threads=max(1, os.cpu_count() // 2),
        temperature=0.7,
        chat_format="chatml",
        verbose=False,
        # Make sure you pass n_gpu_layers > 0 in your environment if you want Metal acceleration.
        # Example: set env var N_GPU_LAYERS=40 and read it here if desired.
    )

    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": full_text}],
        max_tokens=max_response_tokens
    )

    return out["choices"][0]["message"]["content"].strip()


# ----------------------------- Main script ---------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download YouTube transcripts from a channel")
    parser.add_argument("--path", "-p", type=str, required=True,
                        help="Directory path where transcripts will be saved")
    parser.add_argument("--username", "-u", type=str, required=True,
                        help="YouTube channel handle/username (e.g. DavidSnyderNLP or @DavidSnyderNLP)")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_BETWEEN_VIDEOS_S,
                        help="Sleep seconds between videos (to be polite)")
    args = parser.parse_args()

    transcript_folder = args.path
    username = args.username

    # Resolve channel
    channel_id, channel_name = get_youtube_channel_id(youtube_api_key, username)
    if not channel_id:
        print(f"Could not find channel ID for {username}")
        sys.exit(1)

    print(f"Channel ID for {username} is: {channel_id}")
    print(f"Channel Name: {channel_name}")

    os.makedirs(transcript_folder, exist_ok=True)

    # Load already-transcribed IDs to skip
    already_transcribed_ids = {
        fn[:-4] for fn in os.listdir(transcript_folder) if fn.endswith(".txt")
    }

    videos = get_channel_videos(channel_id, youtube_api_key)
    print(f"Total videos: {len(videos)}")

    master_transcript = ""
    transcribeable = 0

    for video in videos:
        video_id = video["id"]["videoId"]
        video_title = video["snippet"]["title"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        if video_id in already_transcribed_ids:
            print(f"Video {video_id} already transcribed and saved. Skipping.")
            continue

        print(f"Title: {video_title}")
        print(f"ID: {video_id}")
        print(f"URL: {video_url}")

        # Fetch transcript text (saves raw JSON next to outputs)
        full_transcript_raw = fetch_transcript_text(video_id, save_dir=transcript_folder)
        if not full_transcript_raw:
            # No transcript available or error — skip
            continue

        # Light formatting that preserves original words (your existing function)
        full_transcript = format_text_preserving(full_transcript_raw)

        # Optional summary
        video_summary = get_transcript_summary(full_transcript)

        if SAVE_SUMMARY_FILE:
            summary_json_path = os.path.join(transcript_folder, f"{video_id}_summary.json")
            summary_data = {
                "video_id": video_id,
                "video_title": video_title,
                "video_url": video_url,
                "channel_id": channel_id,
                "channel_name": channel_name,
                "username": username,
                "publish_date": video["snippet"]["publishedAt"],
                "summary": video_summary,
            }
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)
            print(f"Summary saved to {summary_json_path}")

        print("\nVIDEO SUMMARY: ")
        print(video_summary)

        # Save final .txt
        file_name = f"{video_id}.txt"
        video_publish_date = video["snippet"]["publishedAt"]
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
            video_publish_date,
        )

        print(f"Transcript available: {len(full_transcript) > 0}")
        words = re.findall(r"\b\w+\b", full_transcript)
        print(f"Word count: {len(words)}")
        print("---")

        if full_transcript:
            master_transcript += "\n\n" + video_title + "\n" + full_transcript
            transcribeable += 1

        time.sleep(args.sleep)

    print(f"Total videos: {len(videos)}")
    print(f" - Transcribeable: {transcribeable}")
    print(f"MASTER TRANSCRIPT LENGTH: {len(master_transcript)}")


# ------------------------ Minimal formatter used ---------------------------

def format_text_preserving(text: str) -> str:
    """Format text while preserving every word exactly (no spacy dependency)."""
    original_words = text.split()
    idx = 0
    formatted_sentences: List[str] = []
    current_sentence: List[str] = []

    while idx < len(original_words):
        word = original_words[idx]
        current_sentence.append(word)

        if (
            idx + 1 < len(original_words) and
            (
                original_words[idx + 1][0].isupper() or
                any(word.lower().endswith(end) for end in ["ok", "okay", "right", "yes", "no"]) or
                original_words[idx + 1].lower() in {
                    "then","but","however","therefore","thus","hence","so","anyway",
                    "finally","later","next","first","second","third","last"
                }
            )
        ):
            sentence = " ".join(current_sentence)
            sentence = sentence[0].upper() + sentence[1:]
            if sentence and sentence[-1] not in ".!?":
                sentence += "."
            formatted_sentences.append(sentence)
            current_sentence = []

        idx += 1

    if current_sentence:
        sentence = " ".join(current_sentence)
        sentence = sentence[0].upper() + sentence[1:]
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        formatted_sentences.append(sentence)

    # Group into short paragraphs
    paragraphs: List[str] = []
    para: List[str] = []
    for i, s in enumerate(formatted_sentences):
        para.append(s)
        if len(para) >= 4 or i == len(formatted_sentences) - 1:
            paragraphs.append(" ".join(para))
            para = []
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
