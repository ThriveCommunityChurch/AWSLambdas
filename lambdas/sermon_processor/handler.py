"""
Sermon Processor Lambda

Receives transcript and messageId, then:
1. Generates sermon summary (Azure OpenAI gpt-5-mini)
2. Generates waveform data (FFmpeg + pure Python RMS) - if audio URL provided
3. Generates tags (Azure OpenAI gpt-5-mini)
4. Updates SermonMessages collection in MongoDB
5. If series has EndDate, invokes Series Summary Lambda (eliminates race condition)

The series summary trigger happens AFTER the message summary is saved to MongoDB,
ensuring all message summaries are available before generating the series summary.

Environment Variables:
- MONGODB_SECRET_ARN: Secrets Manager ARN for MongoDB URI
- OPENAI_SECRET_ARN: Secrets Manager ARN for Azure OpenAI API key (Azure_OpenAI_ApiKey)
- SERIES_SUMMARY_LAMBDA_NAME: Name of the Series Summary Processor Lambda

Note: Waveform generation requires FFmpeg Lambda Layer. Uses pure Python RMS
calculation to avoid heavy ML dependencies (librosa, numpy, scipy) that would
exceed Lambda's deployment size limits.
"""

import boto3
import json
import pymongo
import os
from datetime import datetime, timezone
from typing import Optional, List
from bson import ObjectId

# Configuration
DB_NAME = 'SermonSeries'
COLLECTION_NAME = 'Messages'
SERIES_SUMMARY_LAMBDA_NAME = os.environ.get('SERIES_SUMMARY_LAMBDA_NAME', 'series-summary-processor-prod')

# Prompt file paths
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')
SERMON_SUMMARY_PROMPT_FILE = os.path.join(PROMPTS_DIR, 'sermon_summary_prompt.txt')

# AWS clients
secrets_client = boto3.client('secretsmanager', region_name='us-east-2')
lambda_client = boto3.client('lambda', region_name='us-east-2')

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


def get_chat_model_name() -> str:
    """Get the chat model/deployment name based on provider."""
    provider = os.environ.get('OPENAI_PROVIDER', 'azure')
    if provider == 'azure':
        return os.environ.get('AZURE_CHAT_DEPLOYMENT', 'gpt-5-mini')
    else:
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


# =============================================================================
# TAG CONFIGURATION - Full list from Sermon_Summarization_Agent
# =============================================================================

# Relationships & Family
RELATIONSHIPS_FAMILY = [
    "Marriage", "Family", "Friendship", "Singleness"
]

# Financial & Stewardship
FINANCIAL = [
    "FinancialStewardship", "Generosity"
]

# Theological Foundations
THEOLOGICAL = [
    "NatureOfGod", "Trinity", "Salvation", "Resurrection", "HolySpirit",
    "Church", "EndTimes", "SinAndRepentance", "Faith", "Sanctification", "Covenant",
    "Apologetics"
]

# Spiritual Disciplines
SPIRITUAL_DISCIPLINES = [
    "Prayer", "Fasting", "Worship", "BibleStudy", "Meditation", "Service", "Praise"
]

# Sacraments & Ordinances
SACRAMENTS = [
    "Baptism", "Communion"
]

# Life Stages & Transitions
LIFE_STAGES = [
    "Youth", "Aging", "GriefAndLoss", "LifeTransitions"
]

# Social Issues & Justice
SOCIAL_ISSUES = [
    "SocialJustice", "RacialReconciliation", "Poverty", "Creation", "Politics"
]

# Personal Growth & Character
PERSONAL_GROWTH = [
    "Identity", "Purpose", "Courage", "Hope", "Love", "Joy", "Peace", "Patience",
    "Humility", "Wisdom", "Integrity", "Forgiveness", "Gratitude", "Trust",
    "Obedience", "Contentment", "Pride", "Fear", "Anger"
]

# Challenges & Struggles
CHALLENGES = [
    "Suffering", "Doubt", "Anxiety", "Depression", "Addiction", "Temptation",
    "SpiritualWarfare", "Persecution"
]

# Eternal & Supernatural
ETERNAL = [
    "Heaven", "Hell"
]

# Mission & Evangelism
MISSION = [
    "Evangelism", "Missions", "Discipleship", "Leadership", "Witnessing"
]

# Biblical Studies
BIBLICAL_STUDIES = [
    "Parables", "SermonOnTheMount", "FruitOfTheSpirit", "ArmorOfGod", "Prophets"
]

# Biblical Book Studies
BOOK_STUDIES = [
    "Genesis", "Exodus", "Psalms", "Proverbs", "Gospels", "Acts", "Romans",
    "PaulineEpistles", "Revelation", "OldTestament", "NewTestament"
]

# Seasonal & Liturgical
SEASONAL = [
    "Advent", "Christmas", "Lent", "Easter", "Pentecost"
]

# Work & Vocation
WORK = [
    "Work", "Rest"
]

# Gender & Relationships
GENDER = [
    "BiblicalManhood", "BiblicalWomanhood", "SexualPurity"
]

# Other
OTHER = [
    "Miracles", "Prophecy", "Healing", "Community", "Culture", "Technology"
]

# Master list of all tags (90+ tags for comprehensive categorization)
VALID_TAGS = (
    RELATIONSHIPS_FAMILY +
    FINANCIAL +
    THEOLOGICAL +
    SPIRITUAL_DISCIPLINES +
    SACRAMENTS +
    LIFE_STAGES +
    SOCIAL_ISSUES +
    PERSONAL_GROWTH +
    CHALLENGES +
    ETERNAL +
    MISSION +
    BIBLICAL_STUDIES +
    BOOK_STUDIES +
    SEASONAL +
    WORK +
    GENDER +
    OTHER
)

# =============================================================================
# TAG TO INTEGER MAPPING - Matches C# MessageTag enum values
# The API uses C# enums which serialize to integers in MongoDB
# =============================================================================
TAG_TO_INT = {
    # Relationships & Family (0-3)
    "Marriage": 0, "Family": 1, "Friendship": 2, "Singleness": 3,
    # Financial & Stewardship (4-5)
    "FinancialStewardship": 4, "Generosity": 5,
    # Theological Foundations (6-17) - Note: Faith=14 inserted after SinAndRepentance
    "NatureOfGod": 6, "Trinity": 7, "Salvation": 8, "Resurrection": 9, "HolySpirit": 10,
    "Church": 11, "EndTimes": 12, "SinAndRepentance": 13, "Faith": 14, "Sanctification": 15,
    "Covenant": 16, "Apologetics": 17,
    # Spiritual Disciplines (18-24)
    "Prayer": 18, "Fasting": 19, "Worship": 20, "BibleStudy": 21, "Meditation": 22,
    "Service": 23, "Praise": 24,
    # Sacraments & Ordinances (25-26)
    "Baptism": 25, "Communion": 26,
    # Life Stages & Transitions (27-30)
    "Youth": 27, "Aging": 28, "GriefAndLoss": 29, "LifeTransitions": 30,
    # Social Issues & Justice (31-35)
    "SocialJustice": 31, "RacialReconciliation": 32, "Poverty": 33, "Creation": 34, "Politics": 35,
    # Personal Growth & Character (36-54)
    "Identity": 36, "Purpose": 37, "Courage": 38, "Hope": 39, "Love": 40, "Joy": 41, "Peace": 42,
    "Patience": 43, "Humility": 44, "Wisdom": 45, "Integrity": 46, "Forgiveness": 47, "Gratitude": 48,
    "Trust": 49, "Obedience": 50, "Contentment": 51, "Pride": 52, "Fear": 53, "Anger": 54,
    # Challenges & Struggles (55-62)
    "Suffering": 55, "Doubt": 56, "Anxiety": 57, "Depression": 58, "Addiction": 59, "Temptation": 60,
    "SpiritualWarfare": 61, "Persecution": 62,
    # Eternal & Supernatural (63-64)
    "Heaven": 63, "Hell": 64,
    # Mission & Evangelism (65-69)
    "Evangelism": 65, "Missions": 66, "Discipleship": 67, "Leadership": 68, "Witnessing": 69,
    # Biblical Studies (70-74)
    "Parables": 70, "SermonOnTheMount": 71, "FruitOfTheSpirit": 72, "ArmorOfGod": 73, "Prophets": 74,
    # Biblical Book Studies (75-85)
    "Genesis": 75, "Exodus": 76, "Psalms": 77, "Proverbs": 78, "Gospels": 79, "Acts": 80,
    "Romans": 81, "PaulineEpistles": 82, "Revelation": 83, "OldTestament": 84, "NewTestament": 85,
    # Seasonal & Liturgical (86-90)
    "Advent": 86, "Christmas": 87, "Lent": 88, "Easter": 89, "Pentecost": 90,
    # Work & Vocation (91-92)
    "Work": 91, "Rest": 92,
    # Gender & Relationships (93-95)
    "BiblicalManhood": 93, "BiblicalWomanhood": 94, "SexualPurity": 95,
    # Other (96-101)
    "Miracles": 96, "Prophecy": 97, "Healing": 98, "Community": 99, "Culture": 100, "Technology": 101,
}

# TranscriptFeature enum values (must match ThriveChurchOfficialAPI.Core.Enums.TranscriptFeature)
TRANSCRIPT_FEATURE_TO_INT = {
    "Transcript": 0,
    "Notes": 1,
    "StudyGuide": 2,
}


def convert_tags_to_ints(tags: List[str]) -> List[int]:
    """Convert string tag names to their integer enum values for MongoDB storage."""
    int_tags = []
    for tag in tags:
        if tag in TAG_TO_INT:
            int_tags.append(TAG_TO_INT[tag])
        else:
            print(f"Warning: Unknown tag '{tag}' - skipping")
    return int_tags


def convert_transcript_features_to_ints(features: List[str]) -> List[int]:
    """Convert TranscriptFeature string names to their integer enum values for MongoDB storage."""
    int_features = []
    for feature in features:
        if feature in TRANSCRIPT_FEATURE_TO_INT:
            int_features.append(TRANSCRIPT_FEATURE_TO_INT[feature])
        else:
            print(f"Warning: Unknown transcript feature '{feature}' - skipping")
    return int_features


# =============================================================================
# PROMPT FILE LOADING
# =============================================================================

def load_prompt_template(prompt_file: str) -> str:
    """Load a prompt template from file."""
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found: {prompt_file}")
        raise


def build_sermon_summary_prompt_from_file(transcript: str, title: str, speaker: str = "", date: str = "") -> str:
    """Build the sermon summary generation prompt from template file."""
    template = load_prompt_template(SERMON_SUMMARY_PROMPT_FILE)
    return template.replace(
        '{{title}}', title
    ).replace(
        '{{speaker}}', speaker
    ).replace(
        '{{date}}', date
    ).replace(
        '{{transcript}}', transcript
    )


# =============================================================================
# CONTENT GENERATION
# =============================================================================

def generate_sermon_summary(transcript: str, title: str, speaker: str = "", date: str = "") -> str:
    """
    Generate a single-paragraph, end-user-friendly summary of the sermon.
    Uses file-based prompt for maintainability and consistency with promptfoo testing.
    """
    if not transcript:
        return ""

    # Build prompt from file template
    prompt = build_sermon_summary_prompt_from_file(transcript, title, speaker, date)

    try:
        client = get_openai_client()
        model = get_chat_model_name()

        print(f"Generating sermon summary with {model}...")

        # GPT-5 models use different parameters than GPT-4o
        if is_gpt5_model(model):
            # GPT-5 models: no temperature, use max_completion_tokens, developer role
            # Reasoning models need more tokens for internal thinking + output
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "developer", "content": prompt}
                ],
                max_completion_tokens=1500,
                reasoning_effort="low"
            )
        else:
            # GPT-4o models: use temperature, user role for combined prompt
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.45
            )

        summary_text = response.choices[0].message.content
        if not summary_text:
            print(f"Warning: LLM returned empty summary response")
            print(f"Full response: {response}")
            print(f"Finish reason: {response.choices[0].finish_reason}")
            if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                print(f"Refusal: {response.choices[0].message.refusal}")
            return ""
        summary_text = summary_text.strip()

        # Ensure it's a single paragraph (remove any internal line breaks)
        summary_text = " ".join(summary_text.split("\n")).replace("'", "'")

        print(f"Summary generated: {len(summary_text.split())} words")
        return summary_text

    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""


def format_tags_for_prompt(tags: List[str]) -> str:
    """Format tags list for inclusion in the prompt."""
    return ", ".join(tags)


def generate_tags(summary_text: str, transcript: str, title: str) -> List[str]:
    """
    Generate semantic tags for the sermon.
    Uses a hybrid approach: analyzes both the summary (for main themes) and
    transcript excerpt (for comprehensive coverage) to ensure accurate tag selection.
    """
    if not transcript and not summary_text:
        return []

    # Format tags for the prompt
    tags_str = format_tags_for_prompt(VALID_TAGS)

    # Create the system prompt with hybrid approach instructions (from LangGraph agent)
    system_prompt = (
        "You are an expert at analyzing sermon content and applying relevant topical tags. "
        "Your task is to analyze BOTH the sermon summary and transcript excerpt to select "
        "the most relevant tags from a provided list.\n\n"
        "Analysis Strategy:\n"
        "- Use the SUMMARY to identify the main themes and primary focus of the sermon\n"
        "- Use the TRANSCRIPT EXCERPT to identify secondary themes and ensure comprehensive coverage\n"
        "- Having both sources provides better context for accurate tag selection\n"
        "- The transcript may reveal important themes that didn't make it into the summary\n\n"
        "Selection Guidelines:\n"
        "- Select no more than 5 tags that best represent the sermon's themes and topics\n"
        "- Prioritize main themes from the summary, but include important secondary themes from the transcript\n"
        "- Focus on the primary themes, not every minor topic mentioned\n"
        "- Only select tags from the provided list - do not create new tags\n"
        "- Consider both theological themes and practical life applications\n"
        "- If the sermon is part of a book study, include the relevant book tag\n"
        "- Return ONLY the tag names as a JSON array, nothing else\n"
        "- Tag names are case-sensitive and must match exactly\n\n"
        f"Available tags:\n{tags_str}\n\n"
        "Respond with a JSON array of selected tags, for example:\n"
        '[\"Prayer\", \"Suffering\", \"Hope\"]'
    )

    # Limit transcription to ~15000 characters for comprehensive context while managing tokens
    transcript_excerpt = transcript[:15000] if transcript else ""
    if len(transcript) > 15000:
        transcript_excerpt += "..."

    # Build the user prompt with both sources clearly labeled
    user_prompt = (
        "=== SERMON SUMMARY (Main Themes) ===\n"
        f"{summary_text or 'No summary available'}\n\n"
        "=== TRANSCRIPT EXCERPT (Comprehensive Context) ===\n"
        f"{transcript_excerpt}\n\n"
        "=== INSTRUCTIONS ===\n"
        "Please analyze BOTH the summary and transcript excerpt above, then return the most "
        "relevant tags as a JSON array. Use the summary to identify main themes and the "
        "transcript to ensure comprehensive coverage of all important topics discussed."
    )

    try:
        client = get_openai_client()
        model = get_chat_model_name()

        print(f"Analyzing sermon content (summary + transcript) and applying tags with {model}...")

        # GPT-5 models use different parameters than GPT-4o
        if is_gpt5_model(model):
            # GPT-5 models: no temperature, use max_completion_tokens, developer role
            # Reasoning models need more tokens for internal thinking + output
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=1000,
                reasoning_effort="low"
            )
        else:
            # GPT-4o models: use temperature, system role
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=100,
                temperature=0.2
            )

        response_text = response.choices[0].message.content
        if not response_text:
            print(f"Warning: LLM returned empty response")
            print(f"Full response: {response}")
            print(f"Finish reason: {response.choices[0].finish_reason}")
            if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                print(f"Refusal: {response.choices[0].message.refusal}")
            return []
        response_text = response_text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines).strip()

        result = json.loads(response_text)

        if not isinstance(result, list):
            print(f"Warning: LLM returned non-list response: {response_text}")
            return []

        # Validate tags are in the allowed list
        valid_tags = [tag for tag in result if tag in VALID_TAGS]

        if len(valid_tags) != len(result):
            invalid_tags = [tag for tag in result if tag not in VALID_TAGS]
            print(f"Warning: LLM returned invalid tags (ignored): {invalid_tags}")

        print(f"Applied {len(valid_tags)} tags: {', '.join(valid_tags)}")
        return valid_tags

    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse LLM response as JSON: {e}")
        return []
    except Exception as e:
        print(f"Error generating tags: {e}")
        return []


def generate_waveform_from_s3(audio_url: str, num_points: int = 480) -> Optional[List[float]]:
    """
    Download audio from S3 and generate waveform data using FFmpeg + pure Python RMS.
    Returns list of normalized RMS values (0.15-1.0 range).

    This approach avoids heavy ML dependencies (librosa, numpy, scipy) that exceed
    Lambda's deployment size limits. FFmpeg is provided via Lambda Layer.
    """
    if not audio_url:
        return None

    tmp_path = None
    try:
        import tempfile
        import subprocess
        import struct
        import math

        s3 = boto3.client('s3')

        # Parse S3 URL to get bucket and key
        # Format: https://thrive-audio.s3.us-east-2.amazonaws.com/2025/file.mp3
        # or s3://thrive-audio/2025/file.mp3
        if audio_url.startswith('s3://'):
            parts = audio_url[5:].split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
        elif 's3.' in audio_url and 'amazonaws.com' in audio_url:
            # https://bucket.s3.region.amazonaws.com/key
            from urllib.parse import urlparse
            parsed = urlparse(audio_url)
            bucket = parsed.netloc.split('.')[0]
            key = parsed.path.lstrip('/')
        else:
            print(f"Unsupported audio URL format: {audio_url}")
            return None

        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            s3.download_fileobj(bucket, key, tmp)
            tmp_path = tmp.name

        print(f"Downloaded audio to {tmp_path}, generating waveform...")

        # Use FFmpeg to decode audio to raw 16-bit signed PCM
        # -f s16le: 16-bit signed little-endian
        # -ac 1: mono (single channel)
        # -ar 8000: 8kHz sample rate (low rate is fine for waveform visualization)
        # Note: FFmpeg binary is bundled at /var/task/bin/ffmpeg in Lambda
        cmd = [
            '/var/task/bin/ffmpeg',
            '-i', tmp_path,
            '-f', 's16le',
            '-ac', '1',
            '-ar', '8000',  # Low sample rate saves memory - waveform viz doesn't need high fidelity
            '-loglevel', 'error',
            '-'
        ]

        result = subprocess.run(cmd, capture_output=True)

        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr.decode()}")
            return None

        raw_audio = result.stdout
        print(f"Decoded {len(raw_audio):,} bytes of raw PCM audio")

        # Process in chunks directly from bytes to avoid loading all samples into memory
        # Each sample is 2 bytes (16-bit signed)
        num_samples = len(raw_audio) // 2
        chunk_size = num_samples // num_points
        if chunk_size < 1:
            chunk_size = 1

        waveform = []
        for i in range(num_points):
            start_byte = i * chunk_size * 2
            end_byte = min(start_byte + chunk_size * 2, len(raw_audio))
            chunk_bytes = raw_audio[start_byte:end_byte]

            if len(chunk_bytes) < 2:
                waveform.append(0.0)
                continue

            # Unpack only this chunk's samples (not entire file)
            chunk_samples = struct.unpack(f'<{len(chunk_bytes)//2}h', chunk_bytes)

            # RMS = sqrt(mean(samples^2)) / max_amplitude
            # 32768 is max value for 16-bit signed int
            sum_squares = sum(s * s for s in chunk_samples)
            rms = math.sqrt(sum_squares / len(chunk_samples)) / 32768.0
            waveform.append(rms)

        # Normalize to 0.15-1.0 range (matches existing waveform format)
        min_val = min(waveform)
        max_val = max(waveform)

        if max_val - min_val < 0.0001:
            print("Audio appears uniform/silent, returning mid-level waveform")
            return [0.7] * num_points

        # Scale: 0.15 + (value - min) / (max - min) * 0.85
        normalized = [round(0.15 + (v - min_val) / (max_val - min_val) * 0.85, 3) for v in waveform]

        print(f"Generated {len(normalized)} waveform points")
        return normalized

    except Exception as e:
        print(f"Error generating waveform: {e}")
        return None
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def update_sermon_message(db, message_id: str, summary: str, tags: List[str],
                          waveform_data: Optional[List[float]] = None,
                          blob_url: Optional[str] = None,
                          available_transcript_features: Optional[List[str]] = None) -> bool:
    """Update SermonMessage document with generated content."""
    try:
        collection = db[COLLECTION_NAME]

        # Convert string tags to integer enum values for MongoDB storage
        # The API uses C# enums which serialize to integers
        int_tags = convert_tags_to_ints(tags)

        update_doc = {
            'Summary': summary,
            'Tags': int_tags,
            'LastUpdated': datetime.now(timezone.utc)
        }

        if waveform_data:
            update_doc['WaveformData'] = waveform_data

        if blob_url:
            update_doc['BlobUrl'] = blob_url

        if available_transcript_features:
            # Convert string feature names to integer enum values for MongoDB storage
            # The API uses C# enums (TranscriptFeature) which serialize to integers
            int_features = convert_transcript_features_to_ints(available_transcript_features)
            update_doc['AvailableTranscriptFeatures'] = int_features

        result = collection.update_one(
            {'_id': ObjectId(message_id)},
            {'$set': update_doc}
        )

        if result.matched_count == 0:
            print(f"No document found with _id: {message_id}")
            return False

        int_features = convert_transcript_features_to_ints(available_transcript_features) if available_transcript_features else []
        print(f"Updated SermonMessage {message_id}: summary={len(summary)} chars, tags={tags} -> {int_tags}, blobUrl={blob_url is not None}, availableFeatures={available_transcript_features} -> {int_features}")
        return True

    except Exception as e:
        print(f"Error updating SermonMessage: {e}")
        return False


# =============================================================================
# SERIES SUMMARY TRIGGER
# =============================================================================

def get_series_id_for_message(db, message_id: str) -> Optional[str]:
    """Look up the SeriesId for a message from MongoDB."""
    try:
        message = db[COLLECTION_NAME].find_one(
            {'_id': ObjectId(message_id)},
            {'SeriesId': 1}
        )
        if message:
            series_id = message.get('SeriesId')
            # Convert ObjectId to string if necessary for JSON serialization
            if series_id and isinstance(series_id, ObjectId):
                return str(series_id)
            return series_id
        return None
    except Exception as e:
        print(f"Error looking up SeriesId for message {message_id}: {e}")
        return None


def check_series_complete(db, series_id: str) -> bool:
    """
    Check if a series has an EndDate set (meaning it's complete/concluded).
    Used to trigger series summary generation after message processing.
    """
    if not series_id:
        return False

    try:
        series = db['Series'].find_one(
            {'_id': ObjectId(series_id)},
            {'EndDate': 1}
        )
        if series and series.get('EndDate'):
            print(f"Series {series_id} has EndDate, will trigger summary generation")
            return True
        return False
    except Exception as e:
        print(f"Error checking series EndDate: {e}")
        return False


def invoke_series_summary_lambda(series_id: str) -> bool:
    """Invoke the Series Summary Processor Lambda asynchronously."""
    try:
        response = lambda_client.invoke(
            FunctionName=SERIES_SUMMARY_LAMBDA_NAME,
            InvocationType='Event',  # Fire-and-forget
            Payload=json.dumps({'seriesId': series_id})
        )
        print(f"Invoked Series Summary Lambda: {response['StatusCode']}")
        return response['StatusCode'] == 202
    except Exception as e:
        print(f"Error invoking Series Summary Lambda: {e}")
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
        "transcript": "Full transcript text...",
        "blobUrl": "https://thrivefl.blob.core.windows.net/transcripts/abc123.json",  // Optional - Azure Blob URL
        "title": "Sermon Title",
        "speaker": "Pastor Name",
        "date": "2025-01-19T00:00:00",  // ISO format date string
        "passageRef": "John 3:16",
        "audioUrl": "https://thrive-audio.s3.amazonaws.com/2025/file.mp3",  // Optional
        "generateWaveform": true,  // Optional, defaults to false
        "availableTranscriptFeatures": ["SermonNotes", "StudyGuide"]  // Optional - list of generated features
    }
    """
    message_id = event.get('messageId')
    transcript = event.get('transcript', '')
    blob_url = event.get('blobUrl')  # Azure Blob URL
    title = event.get('title', '')
    speaker = event.get('speaker', '')
    date = event.get('date', '')
    passage_ref = event.get('passageRef', '')
    audio_url = event.get('audioUrl')
    generate_waveform = event.get('generateWaveform', False)
    available_transcript_features = event.get('availableTranscriptFeatures', [])  # List of generated features

    if not message_id:
        return {
            'statusCode': 400,
            'body': 'Missing messageId'
        }

    if not transcript:
        return {
            'statusCode': 400,
            'body': 'Missing transcript'
        }

    client = None
    try:
        # Generate content using hybrid approach
        print(f"Processing sermon: {message_id} - {title}")

        # Step 1: Generate summary first
        summary = generate_sermon_summary(transcript, title, speaker, date)

        # Step 2: Generate tags using BOTH summary and transcript (hybrid approach)
        # This provides better context - summary for main themes, transcript for comprehensive coverage
        tags = generate_tags(summary, transcript, title)

        waveform_data = None
        if generate_waveform and audio_url:
            waveform_data = generate_waveform_from_s3(audio_url)

        # Update MongoDB
        client = get_mongodb_client()
        db = client[DB_NAME]

        success = update_sermon_message(
            db, message_id, summary, tags, waveform_data, blob_url,
            available_transcript_features
        )

        if not success:
            return {
                'statusCode': 404,
                'body': f'Message not found: {message_id}'
            }

        # After successfully updating the message, check if we should trigger series summary
        # This happens AFTER message summary is saved, eliminating race conditions
        series_summary_triggered = False
        series_id = get_series_id_for_message(db, message_id)
        if series_id and check_series_complete(db, series_id):
            series_summary_triggered = invoke_series_summary_lambda(series_id)

        return {
            'statusCode': 200,
            'body': {
                'messageId': message_id,
                'summary': summary,
                'tags': tags,
                'hasWaveform': waveform_data is not None,
                'blobUrl': blob_url,
                'availableTranscriptFeatures': available_transcript_features,
                'seriesSummaryTriggered': series_summary_triggered,
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
        if client:
            client.close()


# =============================================================================
# LOCAL TESTING
# =============================================================================

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    # Test event
    test_event = {
        'messageId': '507f1f77bcf86cd799439011',  # Replace with real ID
        'transcript': 'This is a test transcript about faith and grace...',
        'title': 'Test Sermon',
        'passageRef': 'John 3:16'
    }
    result = lambda_handler(test_event, None)
    print(result)

