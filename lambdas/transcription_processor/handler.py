"""
Transcription Processor Lambda

Entry point for the sermon processing pipeline.
1. Downloads audio from S3 (to /tmp)
2. Transcribes with Azure OpenAI Whisper API
3. Uploads transcript to Azure Blob Storage (transcripts container)
4. Invokes Sermon Lambda with { messageId, transcript, transcriptUrl, title, passageRef }
5. Invokes Podcast Lambda with { messageId, transcript, title, speaker, audioUrl, ... }

Environment Variables:
- MONGODB_SECRET_ARN: Secrets Manager ARN for MongoDB URI (to fetch message metadata)
- OPENAI_SECRET_ARN: Secrets Manager ARN for Azure OpenAI API key (Azure_OpenAI_ApiKey)
- AZURE_STORAGE_SECRET_ARN: Secrets Manager ARN for Azure Storage credentials
- WHISPER_DEPLOYMENT_NAME: Azure OpenAI Whisper deployment name (default: whisper)
- SERMON_LAMBDA_NAME: Name of the sermon processor Lambda
- PODCAST_LAMBDA_NAME: Name of the podcast RSS generator Lambda
- S3_BUCKET: S3 bucket for audio files (default: thrive-audio)
- AZURE_STORAGE_ACCOUNT: Azure Storage account name (default: thrivefl)
- AZURE_STORAGE_CONTAINER: Azure Blob container for transcripts (default: transcripts)
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

# Azure Blob Storage configuration
AZURE_STORAGE_ACCOUNT = os.environ.get('AZURE_STORAGE_ACCOUNT', 'thrivefl')
AZURE_STORAGE_CONTAINER = os.environ.get('AZURE_STORAGE_CONTAINER', 'transcripts')

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
    """Get OpenAI API key from Secrets Manager based on provider."""
    provider = os.environ.get('OPENAI_PROVIDER', 'azure')
    if provider == 'azure':
        secret_key = os.environ.get('OPENAI_SECRET_KEY', 'Azure_OpenAI_ApiKey')
    else:
        secret_key = os.environ.get('OPENAI_SECRET_KEY', 'OpenAI_ChatCompletions_ApiKey')
    return get_secret(os.environ['OPENAI_SECRET_ARN'], secret_key)


def get_openai_client():
    """Create OpenAI client based on OPENAI_PROVIDER environment variable."""
    provider = os.environ.get('OPENAI_PROVIDER', 'azure')
    api_key = get_openai_api_key()

    if provider == 'azure':
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT', 'https://thrive-fl.openai.azure.com/'),
            api_key=api_key,
            api_version=os.environ.get('AZURE_OPENAI_API_VERSION', '2024-10-21')
        )
    else:
        from openai import OpenAI
        return OpenAI(api_key=api_key)


def get_transcription_model_name() -> str:
    """Get the transcription model/deployment name based on provider."""
    provider = os.environ.get('OPENAI_PROVIDER', 'azure')
    if provider == 'azure':
        # Azure deployment name (whisper has no duration limit, gpt-4o-transcribe has 25min limit)
        return os.environ.get('AZURE_TRANSCRIPTION_DEPLOYMENT', 'whisper')
    else:
        # Public OpenAI uses whisper-1
        return os.environ.get('OPENAI_TRANSCRIPTION_MODEL', 'whisper-1')


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

        # Get artwork URL - prefer message-level PodcastImageUrl, fall back to series ArtUrl
        artwork_url = message.get('PodcastImageUrl', '')
        if not artwork_url:
            series = db['SermonSeries'].find_one({'_id': ObjectId(message.get('SeriesId'))})
            artwork_url = series.get('ArtUrl', '') if series else ''

        # Get podcast title - prefer PodcastTitle, fall back to Title
        podcast_title = message.get('PodcastTitle', '') or message.get('Title', '')

        return {
            'messageId': message_id,
            'title': message.get('Title', ''),
            'podcastTitle': podcast_title,
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


def compress_audio_for_whisper(audio_path: str) -> Optional[str]:
    """
    Compress audio to under 25MB for Whisper API if needed.
    Uses 32kbps mono MP3 - plenty for speech recognition.
    """
    import subprocess

    # Whisper API limit is 25MB (26,214,400 bytes)
    WHISPER_MAX_SIZE = 25 * 1024 * 1024

    file_size = os.path.getsize(audio_path)
    print(f"Original audio file size: {file_size / (1024*1024):.2f} MB")

    if file_size <= WHISPER_MAX_SIZE:
        print("File is under 25MB, no compression needed")
        return audio_path

    # Compress using FFmpeg: 32kbps mono MP3 (great for speech)
    compressed_path = tempfile.mktemp(suffix='_compressed.mp3', dir='/tmp')
    ffmpeg_path = '/var/task/bin/ffmpeg'

    cmd = [
        ffmpeg_path,
        '-i', audio_path,
        '-ac', '1',           # Mono
        '-ab', '32k',         # 32kbps bitrate
        '-ar', '16000',       # 16kHz sample rate (Whisper's native rate)
        '-y',                 # Overwrite output
        '-loglevel', 'error',
        compressed_path
    ]

    print(f"Compressing audio: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return None

        compressed_size = os.path.getsize(compressed_path)
        print(f"Compressed audio size: {compressed_size / (1024*1024):.2f} MB "
              f"(reduced by {(1 - compressed_size/file_size)*100:.1f}%)")

        return compressed_path

    except subprocess.TimeoutExpired:
        print("FFmpeg compression timed out")
        return None
    except Exception as e:
        print(f"Error compressing audio: {e}")
        return None


def transcribe_audio(audio_path: str) -> Optional[str]:
    """Transcribe audio file using OpenAI or Azure OpenAI transcription API."""
    compressed_path = None
    try:
        client = get_openai_client()
        model = get_transcription_model_name()

        # Compress if over 25MB
        transcribe_path = compress_audio_for_whisper(audio_path)
        if not transcribe_path:
            print("Failed to compress audio for transcription")
            return None

        # Track if we created a compressed file (for cleanup)
        if transcribe_path != audio_path:
            compressed_path = transcribe_path

        with open(transcribe_path, 'rb') as audio_file:
            print(f"Starting transcription with {model}...")
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="text"
            )

        print(f"Transcription complete: {len(response)} characters")
        return response

    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None
    finally:
        # Clean up compressed file if we created one
        if compressed_path and os.path.exists(compressed_path):
            os.unlink(compressed_path)


# =============================================================================
# AZURE BLOB STORAGE
# =============================================================================

# Cache for Azure Blob client
_blob_service_client = None


def get_azure_storage_connection_string() -> str:
    """Get Azure Storage connection string from Secrets Manager."""
    secret_arn = os.environ.get('AZURE_STORAGE_SECRET_ARN')
    if not secret_arn:
        raise ValueError("AZURE_STORAGE_SECRET_ARN environment variable not set")

    response = secrets_client.get_secret_value(SecretId=secret_arn)
    secret_value = json.loads(response['SecretString'])

    return secret_value.get('Azure_Storage_ConnectionString')


def get_blob_service_client():
    """Get Azure Blob Service client using connection string."""
    global _blob_service_client

    if _blob_service_client is None:
        from azure.storage.blob import BlobServiceClient

        connection_string = get_azure_storage_connection_string()
        _blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    return _blob_service_client


def upload_transcript_to_azure(message_id: str, transcript: str, metadata: Dict[str, Any]) -> Optional[str]:
    """
    Upload transcript JSON to Azure Blob Storage.

    Returns the blob URL on success, None on failure.
    Format: https://thrivefl.blob.core.windows.net/transcripts/{messageId}.json
    """
    try:
        blob_service = get_blob_service_client()
        container_client = blob_service.get_container_client(AZURE_STORAGE_CONTAINER)

        blob_name = f"{message_id}.json"

        # Create transcript document - keep it simple for querying
        transcript_doc = {
            'messageId': message_id,
            'title': metadata.get('title', ''),
            'speaker': metadata.get('speaker', ''),
            'transcript': transcript,
            'wordCount': len(transcript.split()),
            'uploadedAt': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        }

        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(
            json.dumps(transcript_doc, indent=2),
            overwrite=True,
            content_settings={'content_type': 'application/json'}
        )

        transcript_url = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/{AZURE_STORAGE_CONTAINER}/{blob_name}"
        print(f"Uploaded transcript to Azure Blob: {transcript_url}")
        return transcript_url

    except Exception as e:
        print(f"Error uploading transcript to Azure Blob: {e}")
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
            # Use metadata from event - podcastTitle falls back to title if not provided
            title = event.get('title', '')
            podcast_title = event.get('podcastTitle', '') or title
            metadata = {
                'messageId': message_id,
                'title': title,
                'podcastTitle': podcast_title,
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

        # Step 3: Upload transcript to Azure Blob Storage
        transcript_url = None
        try:
            transcript_url = upload_transcript_to_azure(message_id, transcript, metadata)
            if not transcript_url:
                print("Warning: Failed to upload transcript to Azure, continuing without URL")
        except Exception as e:
            print(f"Warning: Azure upload error: {e}, continuing without URL")

        # Step 4: Invoke Sermon Lambda
        sermon_payload = {
            'messageId': message_id,
            'transcript': transcript,
            'transcriptUrl': transcript_url,  # Azure Blob URL (or None if upload failed)
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
                'podcastTitle': metadata.get('podcastTitle', ''),
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
                'transcriptUrl': transcript_url,
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

