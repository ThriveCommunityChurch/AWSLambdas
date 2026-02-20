"""
Transcription Processor Lambda

Entry point for the sermon processing pipeline.
1. Downloads audio from S3 (to /tmp)
2. Transcribes with Azure Speech-to-Text Fast API (azure) or OpenAI Whisper API (openai)
3. Uploads transcript to Azure Blob Storage (transcripts container)
4. Invokes Sermon Lambda with { messageId, transcript, transcriptUrl, title, passageRef }
5. Invokes Podcast Lambda with { messageId, transcript, title, speaker, audioUrl, ... }

Environment Variables:
- MONGODB_SECRET_ARN: Secrets Manager ARN for MongoDB URI (to fetch message metadata)
- OPENAI_SECRET_ARN: Secrets Manager ARN for API keys (Azure_OpenAI_ApiKey or OpenAI_ChatCompletions_ApiKey)
- AZURE_STORAGE_SECRET_ARN: Secrets Manager ARN for Azure Storage credentials
- OPENAI_PROVIDER: Provider for AI services - 'azure' or 'openai' (default: azure)
  - azure: Uses Azure Speech-to-Text Fast Transcription API (synchronous, up to 2hrs/200MB)
  - openai: Uses OpenAI Whisper API for transcription (requires compression to <25MB)
- AZURE_SPEECH_ENDPOINT: Azure Speech Services endpoint (default: https://thrive-fl.cognitiveservices.azure.com)
- AZURE_TRANSCRIPTION_DEPLOYMENT: Azure OpenAI Whisper deployment name (only used when OPENAI_PROVIDER=openai)
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
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Callable, Tuple
from bson import ObjectId

# Configuration
S3_BUCKET = os.environ.get('S3_BUCKET', 'thrive-audio')
SERMON_LAMBDA_NAME = os.environ.get('SERMON_LAMBDA_NAME', 'sermon-processor-prod')
PODCAST_LAMBDA_NAME = os.environ.get('PODCAST_LAMBDA_NAME', 'podcast-rss-generator-prod')
DB_NAME = 'SermonSeries'

# Azure Blob Storage configuration
AZURE_STORAGE_ACCOUNT = os.environ.get('AZURE_STORAGE_ACCOUNT', 'thrivefl')
AZURE_STORAGE_CONTAINER = os.environ.get('AZURE_STORAGE_CONTAINER', 'transcripts')

# Azure Speech-to-Text configuration (Fast Transcription API)
AZURE_SPEECH_ENDPOINT = os.environ.get('AZURE_SPEECH_ENDPOINT', 'https://thrive-fl.cognitiveservices.azure.com')

# Sermon Notes & Study Guide configuration (using gpt-4o via Azure OpenAI)

# Prompt file paths (relative to handler.py)
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')
DEVOTIONAL_PROMPT_FILE = os.path.join(PROMPTS_DIR, 'devotional_prompt.txt')
STUDY_GUIDE_PROMPT_FILE = os.path.join(PROMPTS_DIR, 'study_guide_prompt.txt')
NOTES_PROMPT_FILE = os.path.join(PROMPTS_DIR, 'notes_prompt.txt')

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


def configure_langfuse():
    """Configure Langfuse by fetching secret key from AWS Secrets Manager.

    Must be called before any Langfuse/OpenAI imports to ensure the SDK
    picks up the LANGFUSE_SECRET_KEY environment variable.
    """
    if 'LANGFUSE_SECRET_KEY' not in os.environ and 'LANGFUSE_SECRET_ARN' in os.environ:
        try:
            secret_key_name = os.environ.get('LANGFUSE_SECRET_KEY_NAME', 'Langfuse_SecretKey')
            secret = get_secret(os.environ['LANGFUSE_SECRET_ARN'], secret_key_name)
            os.environ['LANGFUSE_SECRET_KEY'] = secret
        except Exception as e:
            print(f"Warning: Could not configure Langfuse: {e}")




def get_openai_client():
    """Create OpenAI client based on OPENAI_PROVIDER environment variable."""
    # Configure Langfuse before importing the wrapped OpenAI client
    configure_langfuse()

    provider = os.environ.get('OPENAI_PROVIDER', 'azure')
    api_key = get_openai_api_key()

    if provider == 'azure':
        from langfuse.openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT', 'https://thrive-fl.openai.azure.com/'),
            api_key=api_key,
            api_version=os.environ.get('AZURE_OPENAI_API_VERSION', '2024-10-21')
        )
    else:
        from langfuse.openai import OpenAI
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


def get_chat_model_name() -> str:
    """Get the chat model/deployment name for sermon notes/study guide generation."""
    provider = os.environ.get('OPENAI_PROVIDER', 'azure')
    if provider == 'azure':
        # Azure deployment name - gpt-5-mini for high-quality generation
        return os.environ.get('AZURE_CHAT_DEPLOYMENT', 'gpt-5-mini')
    else:
        # Public OpenAI model
        return os.environ.get('OPENAI_CHAT_MODEL', 'gpt-5-mini')


def is_gpt5_model(model_name: str) -> bool:
    """Check if the model is a GPT-5 series model (which has different API parameters)."""
    gpt5_prefixes = ('gpt-5', 'o1', 'o3', 'o4')
    return any(model_name.startswith(prefix) for prefix in gpt5_prefixes)


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
            'seriesId': message.get('SeriesId', ''),
            'blobUrl': message.get('BlobUrl', '')
        }
    except Exception as e:
        print(f"Error fetching message metadata: {e}")
        return None


def get_existing_transcript_from_blob(message_id: str, blob_url: Optional[str] = None) -> Optional[str]:
    """
    Fetch existing transcript from Azure Blob Storage.

    If blob_url is provided (from message's BlobUrl field), use it directly.
    Otherwise, fall back to constructing the blob name from message_id.
    """
    try:
        blob_service = get_blob_service_client()
        container_client = blob_service.get_container_client(AZURE_STORAGE_CONTAINER)

        # Determine blob name from URL or message_id
        if blob_url:
            # Extract blob name from URL
            # URL format: https://account.blob.core.windows.net/container/messageId.json
            from urllib.parse import urlparse
            parsed = urlparse(blob_url)
            blob_name = parsed.path.split('/')[-1]  # Get filename from path
            print(f"Using blob name from BlobUrl: {blob_name}")
        else:
            blob_name = f"{message_id}.json"
            print(f"Using default blob name: {blob_name}")

        blob_client = container_client.get_blob_client(blob_name)

        # Check if blob exists
        if not blob_client.exists():
            print(f"No existing transcript blob for message {message_id}")
            return None

        # Download and parse the transcript JSON
        blob_data = blob_client.download_blob().readall()
        transcript_doc = json.loads(blob_data)
        transcript = transcript_doc.get('transcript')

        if transcript:
            print(f"Found existing transcript in blob storage for message {message_id}: {len(transcript)} chars")
            return transcript
        return None
    except Exception as e:
        print(f"Error fetching existing transcript from blob: {e}")
        return None


def download_audio_from_s3(audio_url: str) -> Optional[str]:
    """Download audio file from S3 to /tmp directory."""
    try:
        from urllib.parse import urlparse

        # Parse S3 URL format: https://bucket.s3.us-east-2.amazonaws.com/key
        # Example: https://podcast.thrive-fl.org/2025/2025-11-30-Recording.mp3
        parsed = urlparse(audio_url)
        bucket = S3_BUCKET  # Use environment variable instead of parsing from URL
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


# =============================================================================
# AZURE SPEECH-TO-TEXT TRANSCRIPTION (Fast Transcription API)
# =============================================================================

def transcribe_audio_azure_stt(audio_path: str) -> Optional[str]:
    """
    Transcribe audio using Azure Speech-to-Text Fast Transcription API.

    This is a synchronous API that accepts audio directly via multipart form data
    and returns the transcript immediately. Much faster than batch transcription.

    Supports audio files up to 2 hours and 200MB.
    """
    try:
        # Get API key (same as Azure OpenAI key)
        api_key = get_openai_api_key()

        # Fast transcription endpoint
        transcription_url = f"{AZURE_SPEECH_ENDPOINT}/speechtotext/transcriptions:transcribe?api-version=2024-11-15"

        headers = {
            'Ocp-Apim-Subscription-Key': api_key,
            'Accept': 'application/json'
        }

        # Definition for the transcription request
        definition = json.dumps({
            'locales': ['en-US'],
            'profanityFilterMode': 'None'  # Don't mask profanity for sermon transcripts
        })

        file_size = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"Starting Azure STT fast transcription ({file_size:.1f} MB)...")

        # Send audio as multipart form data
        with open(audio_path, 'rb') as audio_file:
            files = {
                'audio': (os.path.basename(audio_path), audio_file, 'audio/mpeg'),
                'definition': (None, definition, 'application/json')
            }

            response = requests.post(
                transcription_url,
                headers=headers,
                files=files,
                timeout=600  # 10 minute timeout for long audio
            )

        if response.status_code != 200:
            print(f"Azure STT failed: {response.status_code} - {response.text}")
            return None

        result = response.json()

        # Extract transcript from combinedPhrases
        combined_phrases = result.get('combinedPhrases', [])
        if combined_phrases:
            transcript_text = ' '.join(
                phrase.get('text', '') for phrase in combined_phrases
            )
        else:
            # Fallback: try to get from phrases
            phrases = result.get('phrases', [])
            if phrases:
                transcript_text = ' '.join(
                    phrase.get('text', '') for phrase in phrases
                )
            else:
                print("No transcript text found in response")
                print(f"Response keys: {result.keys()}")
                return None

        print(f"Azure STT transcription complete: {len(transcript_text)} characters")
        return transcript_text

    except requests.exceptions.Timeout:
        print("Azure STT request timed out")
        return None
    except Exception as e:
        print(f"Error in Azure STT transcription: {e}")
        return None


def transcribe_audio_whisper(audio_path: str) -> Optional[str]:
    """Transcribe audio file using OpenAI Whisper API (public OpenAI or Azure OpenAI Whisper deployment)."""
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
            print(f"Starting transcription with Whisper ({model})...")
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="text"
            )

        print(f"Whisper transcription complete: {len(response)} characters")
        return response

    except Exception as e:
        print(f"Error in Whisper transcription: {e}")
        return None
    finally:
        # Clean up compressed file if we created one
        if compressed_path and os.path.exists(compressed_path):
            os.unlink(compressed_path)


def transcribe_audio(audio_path: str) -> Optional[str]:
    """
    Transcribe audio file using the configured provider.

    - azure provider: Uses Azure Speech-to-Text Fast API (no compression needed, supports 200MB/2hrs)
    - openai provider: Uses OpenAI Whisper API (requires compression to <25MB)
    """
    provider = os.environ.get('OPENAI_PROVIDER', 'azure')
    print(f"Transcription provider: {provider}")

    if provider == 'azure':
        # Use Azure Speech-to-Text Fast API - no compression needed (200MB limit)
        return transcribe_audio_azure_stt(audio_path)
    else:
        # Use OpenAI Whisper API - requires compression for files >25MB
        return transcribe_audio_whisper(audio_path)


# =============================================================================
# SERMON NOTES & STUDY GUIDE GENERATION
# =============================================================================

# =============================================================================
# PROMPT FILE LOADING
# =============================================================================

def load_prompt_template(prompt_file: str) -> str:
    """Load a prompt template from file."""
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Prompt file not found: {prompt_file}")
        raise


def build_devotional_prompt(transcript: str, metadata: Dict[str, Any]) -> str:
    """Build the devotional generation prompt from template file."""
    template = load_prompt_template(DEVOTIONAL_PROMPT_FILE)
    return template.replace(
        '{{title}}', metadata.get('title', 'Unknown')
    ).replace(
        '{{speaker}}', metadata.get('speaker', 'Unknown')
    ).replace(
        '{{date}}', metadata.get('date', 'Unknown')
    ).replace(
        '{{transcript}}', transcript
    )


def build_study_guide_prompt_from_file(transcript: str, metadata: Dict[str, Any], devotional_text: str) -> str:
    """Build the study guide generation prompt from template file with pre-generated devotional."""
    template = load_prompt_template(STUDY_GUIDE_PROMPT_FILE)
    prompt = template.replace(
        '{{title}}', metadata.get('title', 'Unknown')
    ).replace(
        '{{speaker}}', metadata.get('speaker', 'Unknown')
    ).replace(
        '{{date}}', metadata.get('date', 'Unknown')
    ).replace(
        '{{transcript}}', transcript
    )
    # Append the pre-generated devotional as context
    prompt += f"\n\nPRE-GENERATED DEVOTIONAL (use this exactly as the \"devotional\" field in your JSON output):\n{devotional_text}"
    return prompt


def build_notes_prompt_from_file(transcript: str, metadata: Dict[str, Any]) -> str:
    """Build the sermon notes generation prompt from template file."""
    template = load_prompt_template(NOTES_PROMPT_FILE)
    return template.replace(
        '{{title}}', metadata.get('title', 'Unknown')
    ).replace(
        '{{speaker}}', metadata.get('speaker', 'Unknown')
    ).replace(
        '{{date}}', metadata.get('date', 'Unknown')
    ).replace(
        '{{transcript}}', transcript
    )


def validate_notes(notes: Dict[str, Any], transcript: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate generated notes structure (structural validation only).
    Content validation is handled by promptfoo evals before deployment.
    Returns (is_valid, message, notes_with_validation).
    """
    # Structural validation (required fields)
    required_fields = ['mainScripture', 'summary', 'keyPoints', 'quotes', 'applicationPoints']
    for field in required_fields:
        if field not in notes:
            return False, f"Missing required field: {field}", notes

    # Validate keyPoints structure
    key_points = notes.get('keyPoints', [])
    if not isinstance(key_points, list) or len(key_points) < 2:
        return False, "keyPoints must be an array with at least 2 items", notes

    for i, kp in enumerate(key_points):
        if not isinstance(kp, dict) or 'point' not in kp:
            return False, f"keyPoints[{i}] must have 'point' field", notes

    # Validate quotes structure
    quotes = notes.get('quotes', [])
    if not isinstance(quotes, list):
        return False, "quotes must be an array", notes

    for i, q in enumerate(quotes):
        if not isinstance(q, dict) or 'text' not in q:
            return False, f"quotes[{i}] must have 'text' field", notes

    # Validate applicationPoints
    app_points = notes.get('applicationPoints', [])
    if not isinstance(app_points, list) or len(app_points) < 1:
        return False, "applicationPoints must be an array with at least 1 item", notes

    # Store validation metadata
    notes['_validation'] = {'passed': True}
    return True, "Valid", notes


def validate_study_guide(guide: Dict[str, Any], transcript: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate study guide structure (structural validation only).
    Content validation is handled by promptfoo evals before deployment.
    Returns (is_valid, message, guide_with_validation).
    """
    # Structural validation (required fields)
    required_fields = ['mainScripture', 'summary', 'keyPoints', 'scriptureReferences',
                       'discussionQuestions', 'prayerPrompts', 'takeHomeChallenges', 'devotional']
    for field in required_fields:
        if field not in guide:
            return False, f"Missing required field: {field}", guide

    # Validate discussionQuestions structure (no minimum counts - style guidance handles that)
    dq = guide.get('discussionQuestions', {})
    if not isinstance(dq, dict):
        return False, "discussionQuestions must be an object", guide

    for category in ['icebreaker', 'reflection', 'application']:
        if category not in dq or not isinstance(dq[category], list):
            return False, f"discussionQuestions.{category} must be an array", guide

    # Validate keyPoints
    key_points = guide.get('keyPoints', [])
    if not isinstance(key_points, list) or len(key_points) < 2:
        return False, "keyPoints must be an array with at least 2 items", guide

    # Validate prayerPrompts minimum count
    prayer_prompts = guide.get('prayerPrompts', [])
    if not isinstance(prayer_prompts, list) or len(prayer_prompts) < 2:
        return False, "prayerPrompts must be an array with at least 2 items", guide

    # Validate takeHomeChallenges minimum count
    challenges = guide.get('takeHomeChallenges', [])
    if not isinstance(challenges, list) or len(challenges) < 2:
        return False, "takeHomeChallenges must be an array with at least 2 items", guide

    # Validate devotional (required, non-empty string)
    devotional = guide.get('devotional', '')
    if not isinstance(devotional, str) or len(devotional.strip()) < 100:
        return False, "devotional must be a non-empty string with at least 100 characters", guide

    # Store validation metadata
    guide['_validation'] = {'passed': True}
    return True, "Valid", guide


def assess_confidence(guide: Dict[str, Any], transcript: str) -> Dict[str, str]:
    """
    Assess confidence in generated content by checking scripture references against transcript.

    Returns confidence levels:
    - high: >80% of scriptures found in transcript
    - medium: 50-80% of scriptures found
    - low: <50% of scriptures found (indicates potential hallucination)
    """
    transcript_lower = transcript.lower()

    # Check how many scriptures are actually in the transcript
    scriptures = guide.get('scriptureReferences', [])
    found_count = 0
    for ref in scriptures:
        # Simple check - see if the book name appears in transcript
        book_name = ref.get('reference', '').split()[0].lower() if ref.get('reference') else ''
        if book_name and book_name in transcript_lower:
            found_count += 1

    # Calculate scripture accuracy with three levels
    if len(scriptures) == 0:
        scripture_accuracy = "high"
    else:
        ratio = found_count / len(scriptures)
        if ratio > 0.8:
            scripture_accuracy = "high"
        elif ratio >= 0.5:
            scripture_accuracy = "medium"
        else:
            scripture_accuracy = "low"
            print(f"WARNING: Low scripture accuracy ({found_count}/{len(scriptures)} found)")

    # Content coverage - default to high
    content_coverage = "high"

    return {
        "scriptureAccuracy": scripture_accuracy,
        "contentCoverage": content_coverage
    }


def generate_content(
    prompt: str,
    validator: Callable[[Dict[str, Any], str], Tuple[bool, str, Dict[str, Any]]],
    transcript: str,
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Generate content with LLM (gpt-4o or gpt-5-mini) with validation and retry logic.
    """
    client = get_openai_client()
    model = get_chat_model_name()

    for attempt in range(max_retries):
        try:
            print(f"Generating content using {model} (attempt {attempt + 1}/{max_retries})")

            # Build the system/developer message
            system_content = (
                "You create sermon notes and study guides strictly from the provided "
                "transcript and metadata. You must follow the user's JSON schema "
                "exactly, avoid generic or fabricated content, and return a single "
                "well-formed JSON object that could be traced back to this specific "
                "sermon recording."
            )

            # GPT-5 models use different parameters than GPT-4o
            if is_gpt5_model(model):
                # GPT-5 models: no temperature, use max_completion_tokens, developer role
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "developer", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=16000,
                    reasoning_effort="low",  # low/medium/high - low is faster and cheaper
                    response_format={"type": "json_object"}
                )
            else:
                # GPT-4o models: use temperature, system role
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )

            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Parse JSON from response
            result = parse_json_response(content)
            if not result:
                print(f"Failed to parse JSON on attempt {attempt + 1}")
                print(f"Finish reason: {finish_reason}")
                print(f"Response length: {len(content) if content else 0} chars")
                if content:
                    print(f"First 200 chars: {content[:200]}")
                    print(f"Last 200 chars: {content[-200:]}")
                continue

            # Validate the result
            is_valid, message, validated = validator(result, transcript)
            if is_valid:
                return validated

            print(f"Validation failed on attempt {attempt + 1}: {message}")

        except Exception as e:
            print(f"Generation error on attempt {attempt + 1}: {e}")

    return None


def parse_json_response(content: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    import re
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in content
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def generate_sermon_notes(transcript: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate sermon notes from transcript."""
    print("Generating sermon notes...")
    prompt = build_notes_prompt_from_file(transcript, metadata)
    notes = generate_content(prompt, validate_notes, transcript)

    if notes:
        # Remove internal validation metadata before returning
        notes.pop('_validation', None)
        print("Sermon notes generated successfully")
    else:
        print("Failed to generate sermon notes after retries")

    return notes


def generate_devotional(transcript: str, metadata: Dict[str, Any], max_retries: int = 3) -> Optional[str]:
    """
    Generate devotional text from transcript (first pass of two-pass architecture).

    Returns plain text devotional, not JSON.
    Validates minimum word count and retries if too short.
    """
    client = get_openai_client()
    model = get_chat_model_name()
    min_word_count = 600

    base_prompt = build_devotional_prompt(transcript, metadata)
    previous_word_count = 0  # Track word count from previous attempt for retry message

    for attempt in range(max_retries):
        try:
            prompt = base_prompt

            # Add retry instruction if previous attempt was too short
            if attempt > 0:
                prompt += f"\n\nIMPORTANT: Your previous response was too short ({previous_word_count} words). Please write a COMPLETE devotional with at least 700 words and 5-6 substantial paragraphs. Do not cut your response short."

            print(f"Generating devotional using {model} (attempt {attempt + 1}/{max_retries})")

            # Build the system/developer message for devotional
            system_content = (
                "You are a devotional writer creating contemplative reflections based on sermons. "
                "Write in third-person authoritative voice like a study Bible note. "
                "Return ONLY the devotional text, no JSON formatting."
            )

            # GPT-5 models use different parameters than GPT-4o
            if is_gpt5_model(model):
                # GPT-5 models: no temperature, use max_completion_tokens, developer role
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "developer", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=4000,  # Devotional is shorter than full study guide
                    reasoning_effort="low"
                )
            else:
                # GPT-4o models: use temperature, system role
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.7
                )

            devotional_text = response.choices[0].message.content.strip()

            # Validate word count
            word_count = len(devotional_text.split())
            previous_word_count = word_count  # Save for retry message
            print(f"Devotional word count: {word_count}")

            if word_count >= min_word_count:
                print(f"Devotional generated successfully ({word_count} words)")
                return devotional_text
            else:
                print(f"Devotional too short ({word_count} words < {min_word_count}), retrying...")

        except Exception as e:
            print(f"Error generating devotional (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise

    print("Failed to generate devotional after all retries")
    return None


def generate_study_guide(transcript: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generate study guide from transcript using two-pass architecture.

    Pass 1: Generate devotional with higher quality/reasoning
    Pass 2: Generate full study guide with pre-generated devotional as context
    """
    print("Generating study guide (two-pass architecture)...")

    # PASS 1: Generate devotional
    print("Pass 1: Generating devotional...")
    devotional_text = generate_devotional(transcript, metadata)

    if not devotional_text:
        print("Failed to generate devotional after retries")
        return None

    # PASS 2: Generate study guide with devotional as context
    print("Pass 2: Generating study guide with devotional context...")
    prompt = build_study_guide_prompt_from_file(transcript, metadata, devotional_text)
    guide = generate_content(prompt, validate_study_guide, transcript)

    if guide:
        # Add confidence assessment
        guide['confidenceAssessment'] = assess_confidence(guide, transcript)
        # Remove internal validation metadata
        guide.pop('_validation', None)
        print("Study guide generated successfully (two-pass)")
    else:
        print("Failed to generate study guide after retries")

    return guide


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

    return secret_value.get('AzureBlobsConnectionString')


def get_blob_service_client():
    """Get Azure Blob Service client using connection string."""
    global _blob_service_client

    if _blob_service_client is None:
        from azure.storage.blob import BlobServiceClient

        connection_string = get_azure_storage_connection_string()
        _blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    return _blob_service_client


def upload_transcript_to_azure(
    message_id: str,
    transcript: str,
    metadata: Dict[str, Any],
    sermon_notes: Optional[Dict[str, Any]] = None,
    study_guide: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Upload transcript JSON to Azure Blob Storage.

    Returns the blob URL on success, None on failure.
    Format: https://thrivefl.blob.core.windows.net/transcripts/{messageId}.json

    Output format matches TranscriptBlob.cs and backfill_notes.py:
    {
      "messageId": "...",
      "title": "...",
      "speaker": "...",
      "transcript": "...",
      "wordCount": 5648,
      "uploadedAt": "2026-01-01T03:23:50.562Z",
      "notes": { SermonNotesBlob with metadata },
      "studyGuide": { StudyGuideBlob with metadata }
    }
    """
    from azure.storage.blob import ContentSettings

    try:
        blob_service = get_blob_service_client()
        container_client = blob_service.get_container_client(AZURE_STORAGE_CONTAINER)

        blob_name = f"{message_id}.json"
        title = metadata.get('title', '')
        speaker = metadata.get('speaker', '')
        word_count = len(transcript.split())
        generated_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

        # Create transcript document with all generated content
        transcript_doc = {
            'messageId': message_id,
            'title': title,
            'speaker': speaker,
            'transcript': transcript,
            'wordCount': word_count,
            'uploadedAt': generated_at
        }

        # Include notes with metadata (matches backfill_notes.py format)
        if sermon_notes:
            sermon_notes['generatedAt'] = generated_at
            sermon_notes['modelUsed'] = 'gpt-5-mini'
            sermon_notes['title'] = title
            sermon_notes['speaker'] = speaker
            sermon_notes['wordCount'] = word_count
            transcript_doc['notes'] = sermon_notes

        # Include study guide with metadata (matches backfill_notes.py format)
        if study_guide:
            study_guide['generatedAt'] = generated_at
            study_guide['modelUsed'] = 'gpt-5-mini'
            study_guide['title'] = title
            study_guide['speaker'] = speaker
            transcript_doc['studyGuide'] = study_guide

        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(
            json.dumps(transcript_doc, indent=2),
            overwrite=True,
            content_settings=ContentSettings(content_type='application/json')
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
            # Try to reuse existing transcript from Azure Blob Storage
            # Use the message's BlobUrl if available for accurate lookup
            blob_url = metadata.get('blobUrl')
            transcript = get_existing_transcript_from_blob(message_id, blob_url)
            if transcript:
                print(f"Reusing existing transcript ({len(transcript)} chars)")
            else:
                print("No existing transcript found in blob storage, will transcribe")

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

        # Step 3: Generate sermon notes and study guide
        # Wrap LLM calls with Lambda function name tag for Langfuse tracing
        lambda_name = os.environ.get('AWS_LAMBDA_FUNCTION_NAME', 'local')
        sermon_notes = None
        study_guide = None
        try:
            from langfuse import propagate_attributes

            # Convert date to string for prompt templates (replace() expects strings)
            date_value = metadata.get('date', '')
            if isinstance(date_value, datetime):
                date_value = date_value.strftime('%Y-%m-%d')
            generation_metadata = {
                'title': metadata.get('title', ''),
                'speaker': metadata.get('speaker', ''),
                'date': date_value
            }

            # Tag all LLM calls with Lambda function name
            with propagate_attributes(tags=[lambda_name]):
                sermon_notes = generate_sermon_notes(transcript, generation_metadata)
                study_guide = generate_study_guide(transcript, generation_metadata)
        except Exception as e:
            print(f"Warning: Notes/study guide generation error: {e}, continuing without them")

        # Step 4: Upload transcript to Azure Blob Storage (includes notes and study guide)
        transcript_url = None
        try:
            transcript_url = upload_transcript_to_azure(
                message_id, transcript, metadata,
                sermon_notes=sermon_notes,
                study_guide=study_guide
            )
            if not transcript_url:
                print("Warning: Failed to upload transcript to Azure, continuing without URL")
        except Exception as e:
            print(f"Warning: Azure upload error: {e}, continuing without URL")

        # Step 5: Invoke Sermon Lambda
        # Build list of available transcript features
        # Note: Feature names must match TRANSCRIPT_FEATURE_TO_INT in sermon_processor
        available_features = ['Transcript']  # Always has transcript - Maps to TranscriptFeature.Transcript (0)
        if sermon_notes:
            available_features.append('Notes')  # Maps to TranscriptFeature.Notes (1)
        if study_guide:
            available_features.append('StudyGuide')  # Maps to TranscriptFeature.StudyGuide (2)

        # Convert date to ISO string for JSON serialization
        sermon_date = metadata.get('date', '')
        if isinstance(sermon_date, datetime):
            sermon_date = sermon_date.isoformat()

        sermon_payload = {
            'messageId': message_id,
            'transcript': transcript,
            'blobUrl': transcript_url,  # Azure Blob URL (or None if upload failed)
            'title': metadata.get('title', ''),
            'speaker': metadata.get('speaker', ''),
            'date': sermon_date,
            'passageRef': metadata.get('passageRef', ''),
            'audioUrl': audio_url,
            'generateWaveform': True,  # Generate waveform from audio
            'availableTranscriptFeatures': available_features  # List of generated features
        }
        invoke_sermon_lambda(sermon_payload)

        # Step 6: Invoke Podcast Lambda
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
                'blobUrl': transcript_url,
                'transcriptionSkipped': skip_transcription and tmp_audio_path is None,
                'availableTranscriptFeatures': available_features,
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
        # Flush Langfuse traces before Lambda exits (required for short-lived processes)
        try:
            import langfuse
            langfuse.flush()
        except Exception:
            pass  # Langfuse not configured or error flushing
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

