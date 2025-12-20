"""
Transcription Processor Lambda

Entry point for the sermon processing pipeline.
1. Downloads audio from S3 (to /tmp)
2. Transcribes with OpenAI Whisper API (~$0.24/episode for 40 min)
3. Invokes Sermon Lambda with { messageId, transcript, title, passageRef }
4. Invokes Podcast Lambda with { messageId, transcript, title, speaker, audioUrl, ... }

Environment Variables:
- MONGODB_SECRET_ARN: Secrets Manager ARN for MongoDB URI (to fetch message metadata)
- OPENAI_SECRET_ARN: Secrets Manager ARN for OpenAI API key
- SERMON_LAMBDA_NAME: Name of the sermon processor Lambda
- PODCAST_LAMBDA_NAME: Name of the podcast RSS generator Lambda
- S3_BUCKET: S3 bucket for audio files (default: thrive-audio)
"""

import boto3
import json
import pymongo
import os
import tempfile
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from bson import ObjectId

# Configuration
S3_BUCKET = os.environ.get('S3_BUCKET', 'thrive-audio')
SERMON_LAMBDA_NAME = os.environ.get('SERMON_LAMBDA_NAME', 'sermon-processor-prod')
PODCAST_LAMBDA_NAME = os.environ.get('PODCAST_LAMBDA_NAME', 'podcast-rss-generator-prod')
DB_NAME = 'SermonSeries'

# AWS clients
s3 = boto3.client('s3')
lambda_client = boto3.client('lambda', region_name='us-east-2')
secrets_client = boto3.client('secretsmanager', region_name='us-east-2')

# Cache for secrets (Lambda container reuse)
_secrets_cache = {}


def get_secret(secret_arn: str, secret_key: str = None) -> str:
    """Fetch secret from Secrets Manager with caching."""
    cache_key = f"{secret_arn}:{secret_key}" if secret_key else secret_arn

    if cache_key not in _secrets_cache:
        response = secrets_client.get_secret_value(SecretId=secret_arn)
        secret_value = response['SecretString']

        try:
            parsed = json.loads(secret_value)
            if isinstance(parsed, dict):
                if secret_key:
                    secret_value = parsed.get(secret_key, '')
                elif len(parsed) == 1:
                    secret_value = list(parsed.values())[0]
        except json.JSONDecodeError:
            pass

        _secrets_cache[cache_key] = secret_value
    return _secrets_cache[cache_key]


def get_mongodb_uri() -> str:
    """Get MongoDB URI from Secrets Manager."""
    secret_key = os.environ.get('MONGODB_SECRET_KEY')
    return get_secret(os.environ['MONGODB_SECRET_ARN'], secret_key)


def get_openai_api_key() -> str:
    """Get OpenAI API key from Secrets Manager."""
    secret_key = os.environ.get('OPENAI_SECRET_KEY')
    return get_secret(os.environ['OPENAI_SECRET_ARN'], secret_key)


def get_mongodb_client():
    """Create MongoDB client with connection pooling settings for Lambda."""
    return pymongo.MongoClient(
        get_mongodb_uri(),
        maxPoolSize=1,
        serverSelectionTimeoutMS=5000
    )


def get_message_metadata(db, message_id: str) -> Optional[Dict[str, Any]]:
    """Fetch message metadata from Messages collection."""
    try:
        message = db['Messages'].find_one({'_id': ObjectId(message_id)})
        if not message:
            print(f"Message not found: {message_id}")
            return None

        # Get series info for artwork
        series = db['SermonSeries'].find_one({'_id': ObjectId(message.get('SeriesId'))})
        artwork_url = series.get('ArtUrl', '') if series else ''

        return {
            'messageId': message_id,
            'title': message.get('Title', ''),
            'speaker': message.get('Speaker', ''),
            'passageRef': message.get('PassageRef', ''),
            'audioUrl': message.get('AudioUrl', ''),
            'audioFileSize': message.get('AudioFileSize', 0),
            'audioDuration': message.get('AudioDuration', 0),
            'date': message.get('Date'),
            'artworkUrl': artwork_url,
            'seriesId': message.get('SeriesId', '')
        }
    except Exception as e:
        print(f"Error fetching message metadata: {e}")
        return None


def get_existing_transcript(db, message_id: str) -> Optional[str]:
    """Fetch existing transcript from PodcastEpisodes collection."""
    try:
        episode = db['PodcastEpisodes'].find_one({'messageId': message_id})
        if episode and episode.get('transcript'):
            print(f"Found existing transcript for message {message_id}: {len(episode['transcript'])} chars")
            return episode['transcript']
        return None
    except Exception as e:
        print(f"Error fetching existing transcript: {e}")
        return None


def download_audio_from_s3(audio_url: str) -> Optional[str]:
    """Download audio file from S3 to /tmp directory."""
    try:
        from urllib.parse import urlparse

        # Parse S3 URL format: https://bucket.s3.us-east-2.amazonaws.com/key
        # Example: https://thrive-audio.s3.us-east-2.amazonaws.com/2025/2025-11-30-Recording.mp3
        parsed = urlparse(audio_url)
        bucket = parsed.netloc.split('.')[0]  # thrive-audio
        key = parsed.path.lstrip('/')  # 2025/2025-11-30-Recording.mp3

        print(f"Parsed S3 URL - Bucket: {bucket}, Key: {key}")

        # Download to temp file
        tmp_path = tempfile.mktemp(suffix='.mp3', dir='/tmp')
        s3.download_file(bucket, key, tmp_path)
        print(f"Downloaded audio to {tmp_path}")
        return tmp_path

    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None


def transcribe_audio(audio_path: str) -> Optional[str]:
    """Transcribe audio file using OpenAI Whisper API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=get_openai_api_key())

        with open(audio_path, 'rb') as audio_file:
            print("Starting Whisper transcription...")
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )

        print(f"Transcription complete: {len(response)} characters")
        return response

    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None


def invoke_sermon_lambda(payload: Dict[str, Any]) -> bool:
    """Invoke the Sermon Processor Lambda asynchronously."""
    try:
        response = lambda_client.invoke(
            FunctionName=SERMON_LAMBDA_NAME,
            InvocationType='Event',  # Fire-and-forget
            Payload=json.dumps(payload)
        )
        print(f"Invoked Sermon Lambda: {response['StatusCode']}")
        return response['StatusCode'] == 202
    except Exception as e:
        print(f"Error invoking Sermon Lambda: {e}")
        return False


def invoke_podcast_lambda(payload: Dict[str, Any]) -> bool:
    """Invoke the Podcast RSS Generator Lambda asynchronously."""
    try:
        response = lambda_client.invoke(
            FunctionName=PODCAST_LAMBDA_NAME,
            InvocationType='Event',  # Fire-and-forget
            Payload=json.dumps(payload)
        )
        print(f"Invoked Podcast Lambda: {response['StatusCode']}")
        return response['StatusCode'] == 202
    except Exception as e:
        print(f"Error invoking Podcast Lambda: {e}")
        return False


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def lambda_handler(event, context):
    """
    Main Lambda handler.

    Event structure:
    {
        "messageId": "abc123",
        "skipTranscription": false  // Optional: skip Whisper and reuse existing transcript
    }

    Or with pre-provided metadata (skips MongoDB lookup):
    {
        "messageId": "abc123",
        "skipTranscription": false,
        "audioUrl": "https://thrive-audio.s3.amazonaws.com/2025/file.mp3",
        "title": "Sermon Title",
        "speaker": "Pastor Name",
        "passageRef": "John 3:16",
        "audioFileSize": 38248061,
        "audioDuration": 2389,
        "date": "2025-06-15T10:00:00Z",
        "artworkUrl": "https://..."
    }
    """
    message_id = event.get('messageId')
    skip_transcription = event.get('skipTranscription', False)

    if not message_id:
        return {
            'statusCode': 400,
            'body': 'Missing messageId'
        }

    client = None
    tmp_audio_path = None

    try:
        # Connect to MongoDB
        client = get_mongodb_client()
        db = client[DB_NAME]

        # Get message metadata (from event or MongoDB)
        if event.get('audioUrl'):
            # Use metadata from event
            metadata = {
                'messageId': message_id,
                'title': event.get('title', ''),
                'speaker': event.get('speaker', ''),
                'passageRef': event.get('passageRef', ''),
                'audioUrl': event.get('audioUrl'),
                'audioFileSize': event.get('audioFileSize', 0),
                'audioDuration': event.get('audioDuration', 0),
                'date': event.get('date'),
                'artworkUrl': event.get('artworkUrl', ''),
                'seriesId': event.get('seriesId', '')
            }
        else:
            # Fetch from MongoDB
            metadata = get_message_metadata(db, message_id)

            if not metadata:
                return {
                    'statusCode': 404,
                    'body': f'Message not found: {message_id}'
                }

        audio_url = metadata.get('audioUrl')
        if not audio_url:
            return {
                'statusCode': 400,
                'body': f'No audio URL for message: {message_id}'
            }

        print(f"Processing message: {message_id} - {metadata.get('title')} (skipTranscription={skip_transcription})")

        # Get transcript - either from existing PodcastEpisode or by transcribing
        transcript = None

        if skip_transcription:
            # Try to reuse existing transcript from PodcastEpisodes
            transcript = get_existing_transcript(db, message_id)
            if transcript:
                print(f"Reusing existing transcript ({len(transcript)} chars)")
            else:
                print("No existing transcript found, will transcribe")

        if not transcript:
            # Step 1: Download audio from S3
            tmp_audio_path = download_audio_from_s3(audio_url)

            if not tmp_audio_path:
                return {
                    'statusCode': 500,
                    'body': 'Failed to download audio file'
                }

            # Step 2: Transcribe with Whisper
            transcript = transcribe_audio(tmp_audio_path)

        if not transcript:
            return {
                'statusCode': 500,
                'body': 'Failed to transcribe audio'
            }

        # Step 3: Invoke Sermon Lambda
        sermon_payload = {
            'messageId': message_id,
            'transcript': transcript,
            'title': metadata.get('title', ''),
            'passageRef': metadata.get('passageRef', ''),
            'audioUrl': audio_url,
            'generateWaveform': True  # Generate waveform from audio
        }
        invoke_sermon_lambda(sermon_payload)

        # Step 4: Invoke Podcast Lambda
        pub_date = metadata.get('date')
        if isinstance(pub_date, datetime):
            pub_date = pub_date.isoformat()

        podcast_payload = {
            'action': 'upsert',
            'episode': {
                'messageId': message_id,
                'title': metadata.get('title', ''),
                'audioUrl': audio_url,
                'audioFileSize': metadata.get('audioFileSize', 0),
                'audioDuration': metadata.get('audioDuration', 0),
                'speaker': metadata.get('speaker', ''),
                'pubDate': pub_date,
                'artworkUrl': metadata.get('artworkUrl', ''),
                'guid': message_id,
                'transcript': transcript  # For description generation
            }
        }
        invoke_podcast_lambda(podcast_payload)

        return {
            'statusCode': 200,
            'body': {
                'messageId': message_id,
                'transcriptLength': len(transcript),
                'transcriptionSkipped': skip_transcription and tmp_audio_path is None,
                'sermonLambdaInvoked': True,
                'podcastLambdaInvoked': True,
                'status': 'success'
            }
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': f'Error: {str(e)}'
        }
    finally:
        # Cleanup
        if client:
            client.close()
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.unlink(tmp_audio_path)


# =============================================================================
# LOCAL TESTING
# =============================================================================

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    # Test event
    test_event = {
        'messageId': '507f1f77bcf86cd799439011'  # Replace with real ID
    }
    result = lambda_handler(test_event, None)
    print(result)

