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

def build_notes_prompt(transcript: str, metadata: Dict[str, Any]) -> str:
    """Build the sermon notes generation prompt (aligned with TranscriptBlob schema)."""
    return f'''You are a ministry assistant helping create sermon notes for church members. Your job is to distill a sermon transcript into shareable, practical notes that sound like this specific message, not a generic sermon.

CRITICAL RULES:
1. ONLY include scripture references that the speaker explicitly mentioned or read in the transcript
2. Quotes must be ACTUAL phrases from the sermon (you may clean up filler words, but the core wording must appear in the transcript)
3. Key points should use the speaker's own framework/outline and language when apparent
4. The summary should speak directly to the reader in the PRESENT tense, capturing the sermon's heart and unique angle (for example, "Words are powerful, shaping your life and relationships"), not just list topics or say "This sermon was about..."
5. Application points must be specific actions, not vague encouragements
6. If you are unsure whether something was said or which scripture was used, LEAVE IT OUT rather than guessing
7. Do not mention the name of the speaker, the church, or any other organization - focus on the content of the sermon
8. Write everything as if you are talking to the reader right now (using "you" and "we"), not reporting on what the speaker did (avoid phrases like "the speaker said" or "in this sermon they talked about...") in summaries, contexts, or details

SERMON METADATA:
- Title: {metadata.get('title', 'Unknown')}
- Speaker: {metadata.get('speaker', 'Unknown')}
- Date: {metadata.get('date', 'Unknown')}

TRANSCRIPT:
{transcript}

Generate sermon notes in this exact JSON structure. Do NOT include title, speaker, date, generatedAt, modelUsed, or wordCount fields – the system will add those. Every field must be grounded in the transcript above:

{{
  "mainScripture": "The primary passage the sermon is based on (e.g., 'Galatians 4:1-7'). Use an empty string if no clear primary passage is given.",
  "summary": "2-3 paragraphs (approximately 250 words) that speak directly to the reader in the present tense, capturing the sermon's core message and why it matters right now. IMPORTANT: Separate each paragraph with two newline characters (\\n\\n) so the text is not a wall of text. Write it so someone who missed Sunday would still feel personally invited into what God is saying.",
  "keyPoints": [
    {{
      "point": "A clear, memorable statement of the idea using the speaker's own language when possible.",
      "scripture": "Book Chapter:Verse ONLY if explicitly referenced in the transcript (otherwise an empty string).",
      "detail": "One sentence of elaboration, written to the reader in the present tense, using the speaker's own explanation from the sermon (for example, 'Our words can shape our lives and the lives of others...')."
    }}
  ],
  "quotes": [
    {{
      "text": "An actual memorable line from the sermon (lightly cleaned of filler words only).",
      "context": "Brief, reader-facing context written in the present tense (for example, how this line helps you see God, yourself, or others differently), or an empty string if not needed. Avoid meta-phrases like 'in this part of the sermon' or 'the speaker was talking about...'."
    }}
  ],
  "applicationPoints": [
    "Specific, actionable takeaway – what should someone DO this week in response to this sermon?"
  ]
}}

MINIMUM REQUIREMENTS (the validator will reject responses that don't meet these):
- keyPoints: at least 2 items (every sermon has at least 2 main ideas)
- quotes: may be empty if no truly quotable lines are present
- applicationPoints: at least 1 item

STYLE GUIDANCE:
- Aim for 3–5 keyPoints, 2–4 quotes, and 2–4 applicationPoints when the transcript supports them
- Write keyPoints as statements someone would remember, not academic summaries
- Make quotes truly quotable – the kind of line someone might share or recall later
- ApplicationPoints must be concrete actions (e.g., "Have a conversation with...", "Set aside time to..."), not vague ideas (e.g., "Think about...", "Try to be better...")
- Do NOT introduce new scriptures, stories, or ideas that are not clearly present in the transcript

Return ONLY a single valid JSON object that matches the structure above.'''


def build_study_guide_prompt(transcript: str, metadata: Dict[str, Any]) -> str:
    """Build the study guide generation prompt (aligned with TranscriptBlob schema)."""
    return f'''You are a thoughtful small group leader and devotional writer preparing a study guide from a sermon.

Your goal is to create a short, Scripture-rooted devotional and discussion guide that helps people
personally encounter Jesus through THIS specific message. The tone should feel like a modern Bible
reading plan: warm, conversational, reflective, and accessible—not academic, preachy, or assuming
that people already "know all the church words".

CRITICAL RULES:
1. Scripture accuracy is paramount:
   - Set "directlyQuoted": true ONLY for passages the speaker actually read aloud in the transcript
   - Set "directlyQuoted": false for passages the speaker mentioned but did not read
   - Use the "additionalStudy" section ONLY for related passages NOT mentioned in the sermon but helpful for deeper study
2. Every question, summary line, and application must clearly flow from THIS sermon (stories, phrases,
   arguments, scriptures) – not generic Christian ideas
3. Do not introduce scriptures, stories, or big ideas that are not clearly present in the transcript
4. Use the sermon's own illustrations and language in your questions and points whenever possible
5. If you are unsure whether a scripture or idea appeared in the sermon, LEAVE IT OUT rather than guessing
6. Do not mention the name of the speaker, the church, or any other organization – focus on the content of the sermon
7. The tone of the devotional should be:
   - Casual but honoring (you can say "you" and "we")
   - Thoughtful and honest, never cheesy or overly hyped
   - Gentle and invitational, not shaming or assuming people are at a certain level of spiritual maturity
8. Do NOT explicitly mention any app, website, or brand name.
9. Write in the present tense as if you are talking directly to the reader right now, not merely reporting on what the speaker did (avoid phrases like "the speaker said" or "this passage was used to show" in summaries, contexts, and illustrations).

SERMON METADATA:
- Title: {metadata.get('title', 'Unknown')}
- Speaker: {metadata.get('speaker', 'Unknown')}
- Date: {metadata.get('date', 'Unknown')}

TRANSCRIPT:
{transcript}

Generate a study guide in this exact JSON structure. Do NOT include title, speaker, date, generatedAt,
modelUsed, or confidence fields – the system will add those.

MINIMUM REQUIREMENTS (the validator will reject responses that don't meet these):
- keyPoints: at least 2 items (every sermon has at least 2 main ideas)
- prayerPrompts: at least 2 items
- takeHomeChallenges: at least 2 items
- devotional: required (3–5 paragraphs, approximately 600–800 words)

STYLE GUIDANCE (aim for these counts when the sermon content supports them):
- keyPoints: 3–5 items
- discussionQuestions.icebreaker: 2–3 questions
- discussionQuestions.reflection: 2–3 questions
- discussionQuestions.application: 2–3 questions
- prayerPrompts: 3–4 prompts
- takeHomeChallenges: 2–3 challenges

Other arrays (illustrations, additionalStudy, scriptureReferences) may be empty if the content isn't clearly present:

{{
  "mainScripture": "Primary passage for the sermon (or an empty string if unclear).",
  "summary": "A paragraph (4–6 sentences) that speaks directly to the reader in the present tense, capturing the sermon's message, main argument, and significance in a devotional tone—grounded in Scripture, reflective, and written so someone who missed Sunday still feels invited into what God is saying right now.",

  "keyPoints": [
    {{
      "point": "Core idea in simple, memorable language, using the speaker's own phrasing when possible.",
      "theologicalContext": "1–2 sentences of gentle background that deepen understanding, drawn from the sermon itself and rooted in the passage(s) used.",
      "scripture": "Reference if applicable (only if the scripture was mentioned).",
      "directlyQuoted": true
    }}
  ],

  "scriptureReferences": [
    {{
      "reference": "Book Chapter:Verse (only scriptures that were actually mentioned).",
      "context": "A short, devotional explanation of what this passage is saying to the reader now (for example, how it calls you to trust, surrender, forgive, or obey), written in the present tense and avoiding meta-phrases like 'used to show' or 'the speaker used this passage to...'.",
      "directlyQuoted": true
    }}
  ],

  "discussionQuestions": {{
    "icebreaker": [
      "An easy, relatable question connected to the sermon's theme that everyone can answer without needing Bible knowledge.",
      "Another accessible opening question that draws people in using everyday experiences related to the topic."
    ],
    "reflection": [
      "A question that invites honest self-examination using specific content from this sermon, phrased gently and without shame.",
      "Another reflection question that helps people connect the sermon's message to their own story or struggles.",
      "A deeper question that explores a specific point, illustration, or scripture from the sermon."
    ],
    "application": [
      "A question that moves toward concrete, grace-filled next steps (what this could look like in everyday life), clearly tied to this sermon.",
      "Another practical question that helps people identify one specific action they can take this week."
    ]
  }},

  "illustrations": [
    {{
      "summary": "Brief description of a story or example from the sermon, written so the reader can picture it without saying 'the speaker shared...'.",
      "point": "What that story or example illustrates for the reader's life right now (for example, what it reveals about God, the heart, faith, or relationships), stated in clear, present-tense language."
    }}
  ],

  "prayerPrompts": [
    "Short, specific prayer prompts written in a natural, conversational tone (for example, lines someone could pray after reading this devotional)."
  ],

  "takeHomeChallenges": [
    "A concrete, realistic action for the coming week that applies the sermon in everyday life (not just a vague idea)."
  ],

  "devotional": "A 4–5 paragraph personal devotional. This is the longest and most substantial section of the study guide—do not cut it short. Each paragraph should be 130–160 words. IMPORTANT: Separate each paragraph with two newline characters (\\n\\n) so the text is not a wall of text. Write it like a daily devotional reading—warm, reflective, and conversational. Do not refer to it as 'this guide' or 'this study' or 'this devotional'—instead, Start by grounding the reader in the main Scripture passage, then walk through the sermon's key insight or tension, and close with an invitation to respond personally. You may incorporate external Scripture references or theologically sound illustrations to support the sermon's main theme and learning outcomes, but ensure everything aligns with orthodox Christian teaching. Avoid churchy clichés; write as if you're sitting across the table from a friend who genuinely wants to grow but doesn't have all the answers. This should feel like something someone would read with their morning coffee or huddled in a group setting for a Bible study, ready to be moved by the Spirit.",

  "additionalStudy": [
    {{
      "topic": "A theme from the sermon worth exploring further.",
      "scriptures": ["Related passages for deeper study (not necessarily mentioned in the sermon)."],
      "note": "Briefly explain how these passages connect back to the sermon's core Scripture and themes."
    }}
  ],

  "estimatedStudyTime": "Approximate time to complete the guide (e.g., '30–45 minutes')."
}}

QUESTION QUALITY EXAMPLES:

❌ BAD (generic, could be any sermon):
- "What stood out to you from this message?"
- "How can you apply this to your life?"
- "What is God saying to you?"

✅ GOOD (specific to THIS sermon, devotional in tone):
- "The speaker said 'every saint has a past and every sinner has a future.' Which part of that statement is harder for you to believe about yourself right now, and why?"
- "[Replace with a concrete quote or image from THIS sermon] – ask a question that directly builds on it in a gentle, reflective way."

AUTHENTICITY CHECKLIST (for your own internal use):
- Would a real small group leader or devotional writer who heard THIS sermon ask these questions?
- Do the questions clearly reference specific content from THIS sermon (not just general ideas)?
- Do application questions lead to concrete next steps, not vague feelings or pressure?
- Does the overall guide feel like something a person might read in a daily devotional—rooted in Scripture, honest about real life, and full of grace?

Return ONLY a single valid JSON object that matches the structure above.'''


def validate_notes(notes: Dict[str, Any], transcript: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate generated notes structure and content against transcript.
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

    # Content validation (check against transcript)
    issues = []
    transcript_lower = transcript.lower()

    # Check quotes exist in transcript
    for quote in quotes:
        quote_text = quote.get('text', '').lower()
        # Allow some flexibility for cleaned-up quotes - check first 5 words
        words = quote_text.split()[:5]
        search_text = ' '.join(words)
        if search_text and search_text not in transcript_lower:
            issues.append(f"Quote may not be verbatim: '{quote_text[:50]}...'")

    # Check scripture references exist in transcript
    for point in key_points:
        scripture = point.get('scripture', '')
        if scripture:
            book = scripture.split()[0].lower() if scripture else ''
            if book and book not in transcript_lower:
                issues.append(f"Scripture '{scripture}' not found in transcript")

    notes['_validation'] = {
        'passed': True,
        'issues': issues,
        'validated': len(issues) == 0
    }

    if issues:
        print(f"Notes validation warnings: {issues}")

    return True, "Valid", notes


def validate_study_guide(guide: Dict[str, Any], transcript: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate study guide structure and content against transcript.
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

    # Content validation (check against transcript)
    issues = []
    transcript_lower = transcript.lower()

    # Check directly quoted scriptures
    for ref in guide.get('scriptureReferences', []):
        if ref.get('directlyQuoted'):
            reference = ref.get('reference', '')
            book = reference.split()[0].lower() if reference else ''
            if book and book not in transcript_lower:
                issues.append(f"Marked as quoted but not found: '{reference}'")

    guide['_validation'] = {
        'passed': True,
        'issues': issues,
        'validated': len(issues) == 0
    }

    if issues:
        print(f"Study guide validation warnings: {issues}")

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
    prompt = build_notes_prompt(transcript, metadata)
    notes = generate_content(prompt, validate_notes, transcript)

    if notes:
        # Remove internal validation metadata before returning
        notes.pop('_validation', None)
        print("Sermon notes generated successfully")
    else:
        print("Failed to generate sermon notes after retries")

    return notes


def generate_study_guide(transcript: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate study guide from transcript."""
    print("Generating study guide...")
    prompt = build_study_guide_prompt(transcript, metadata)
    guide = generate_content(prompt, validate_study_guide, transcript)

    if guide:
        # Add confidence assessment
        guide['confidenceAssessment'] = assess_confidence(guide, transcript)
        # Remove internal validation metadata
        guide.pop('_validation', None)
        print("Study guide generated successfully")
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
        sermon_notes = None
        study_guide = None
        try:
            generation_metadata = {
                'title': metadata.get('title', ''),
                'speaker': metadata.get('speaker', ''),
                'date': metadata.get('date', '')
            }
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

        sermon_payload = {
            'messageId': message_id,
            'transcript': transcript,
            'blobUrl': transcript_url,  # Azure Blob URL (or None if upload failed)
            'title': metadata.get('title', ''),
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

