"""
Podcast RSS Generator Lambda

Handles two actions:
- upsert: Add/update a single episode in MongoDB, then rebuild entire RSS feed
- rebuild: Regenerate entire RSS feed from PodcastEpisodes collection

Design: MongoDB (PodcastEpisodes collection) is the source of truth. The RSS feed
is always rebuilt from the database to avoid fragile XML manipulation. This ensures
data integrity and prevents bugs where updating one episode could corrupt the feed.

Environment Variables:
- MONGODB_SECRET_ARN: Secrets Manager ARN for MongoDB URI
- OPENAI_SECRET_ARN: Secrets Manager ARN for Azure OpenAI API key (Azure_OpenAI_ApiKey)
- S3_BUCKET: S3 bucket name (default: thrive-audio)
- S3_FEED_KEY: S3 key for RSS feed (default: feed/rss.xml)
"""

import boto3
import json
import pymongo
import os
from datetime import datetime, timezone
from email.utils import formatdate
from typing import Optional
from html import escape

# Configuration
S3_BUCKET = os.environ.get('S3_BUCKET', 'thrive-audio')
S3_FEED_KEY = os.environ.get('S3_FEED_KEY', 'feed/rss.xml')
DB_NAME = 'SermonSeries'

# Prompt file paths
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')
PODCAST_DESCRIPTION_PROMPT_FILE = os.path.join(PROMPTS_DIR, 'podcast_description_prompt.txt')

# AWS clients
s3 = boto3.client('s3')
secrets_client = boto3.client('secretsmanager', region_name='us-east-2')

# Cache for secrets (Lambda container reuse)
_secrets_cache = {}


def get_secret(secret_arn: str, secret_key: str = None) -> str:
    """Fetch secret from Secrets Manager with caching.

    Args:
        secret_arn: The ARN of the secret
        secret_key: Optional key if secret is JSON with multiple keys
    """
    cache_key = f"{secret_arn}:{secret_key}" if secret_key else secret_arn

    if cache_key not in _secrets_cache:
        response = secrets_client.get_secret_value(SecretId=secret_arn)
        secret_value = response['SecretString']

        # Handle JSON format
        try:
            parsed = json.loads(secret_value)
            if isinstance(parsed, dict):
                if secret_key:
                    # Get specific key from JSON
                    secret_value = parsed.get(secret_key, '')
                elif len(parsed) == 1:
                    # Single key JSON, get that value
                    secret_value = list(parsed.values())[0]
        except json.JSONDecodeError:
            pass  # Plain string, use as-is

        _secrets_cache[cache_key] = secret_value
    return _secrets_cache[cache_key]


def get_mongodb_uri() -> str:
    """Get MongoDB URI from Secrets Manager."""
    secret_key = os.environ.get('MONGODB_SECRET_KEY')  # e.g., 'MongoConnectionString'
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


# =============================================================================
# STATIC CHANNEL HEADER (only lastBuildDate changes)
# =============================================================================

CHANNEL_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
     xmlns:content="http://purl.org/rss/1.0/modules/content/"
     xmlns:atom="http://www.w3.org/2005/Atom">
<channel>
    <title>Thrive Community Church</title>
    <link>https://thrive-fl.org</link>
    <language>en-us</language>
    <copyright>© {year} Thrive Community Church</copyright>
    <description>Weekly sermons from Thrive Community Church in Estero, Florida. Join us as we explore God's Word and grow together in faith.</description>
    <itunes:author>Thrive Community Church</itunes:author>
    <itunes:owner>
        <itunes:name>Thrive Community Church</itunes:name>
        <itunes:email>info@thrive-fl.org</itunes:email>
    </itunes:owner>
    <itunes:image href="https://d2v6hk6f64og35.cloudfront.net/podcast_img.jpg"/>
    <itunes:category text="Religion &amp; Spirituality">
        <itunes:category text="Christianity"/>
    </itunes:category>
    <itunes:explicit>false</itunes:explicit>
    <atom:link href="https://podcast.thrive-fl.org/feed/rss.xml" rel="self" type="application/rss+xml"/>
    <lastBuildDate>{last_build_date}</lastBuildDate>
"""

CHANNEL_FOOTER = """</channel>
</rss>"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_mongodb_client():
    """Create MongoDB client with connection pooling settings for Lambda."""
    return pymongo.MongoClient(
        get_mongodb_uri(),
        maxPoolSize=1,
        serverSelectionTimeoutMS=5000
    )


def format_duration(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format for iTunes."""
    # Handle None or invalid values
    if seconds is None:
        return "00:00:00"
    # Convert to int to handle float durations from MongoDB
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_rfc2822_date(dt: datetime) -> str:
    """Format datetime as RFC 2822 for RSS pubDate."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return formatdate(dt.timestamp(), usegmt=True)


def escape_xml(text: str) -> str:
    """Escape special XML characters."""
    if not text:
        return ""
    return escape(text)


def ensure_datetime(value) -> Optional[datetime]:
    """
    Ensure a value is a proper datetime object for MongoDB storage.

    MongoDB sorts datetime objects correctly, but string dates sort lexicographically,
    which breaks chronological ordering (e.g., "2025-12-14" sorts after "2025-12-07"
    as strings, but we need proper date comparison).

    Args:
        value: Can be datetime, string (ISO format), or None

    Returns:
        datetime object with UTC timezone, or None if invalid
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        # Already a datetime - ensure it has timezone info
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    if isinstance(value, str):
        try:
            # Parse ISO format string (handles both with and without 'Z' suffix)
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            pass

    return None


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


def build_podcast_description_prompt(transcript: str, title: str, speaker: str) -> str:
    """Build the user prompt for podcast description generation."""
    return (
        f"SERMON METADATA:\n"
        f"- Title: {title}\n"
        f"- Speaker: {speaker}\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"Generate a podcast description: two paragraphs (130-180 words total). "
        f"First paragraph names the tension. Second previews the approach and ends with curiosity, not a question. "
        f"Paragraphs separated by a blank line. Return ONLY the description text."
    )


def generate_podcast_description(transcript: str, title: str, speaker: str, passage_ref: str = "") -> str:
    """
    Generate a podcast-friendly description.

    This description is optimized for podcast apps (Apple Podcasts, Spotify, etc.) where
    listeners may be discovering the church for the first time. The tone is welcoming and
    accessible to seekers, newcomers, and those exploring faith - not just existing members.
    """
    if not transcript:
        # Fallback description if no transcript
        return f"Join {speaker} as they share a powerful message titled '{title}'."

    # Load system prompt from file
    system_prompt = load_prompt_template(PODCAST_DESCRIPTION_PROMPT_FILE)

    # Build user prompt with sermon metadata
    user_prompt = build_podcast_description_prompt(transcript, title, speaker)

    try:
        client = get_openai_client()
        model = get_chat_model_name()

        print(f"Generating podcast description with {model}...")

        # GPT-5 models use different parameters than GPT-4o
        if is_gpt5_model(model):
            # GPT-5 models: no temperature, use max_completion_tokens, developer role
            # NOTE: GPT-5 uses reasoning tokens that consume the budget, need high max_completion_tokens
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=20000,
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
                max_tokens=500,
                temperature=0.6
            )

        description = response.choices[0].message.content.strip()
        print(f"Podcast description generated: {len(description.split())} words")
        return description

    except Exception as e:
        print(f"Error generating description: {e}")
        return f"Join {speaker} as they share a powerful message titled '{title}'."



def build_podcast_title(episode: dict) -> str:
    """
    Build the podcast episode title.

    Priority:
    1. Use podcastTitle if explicitly set
    2. Otherwise, build from: "Series Name – Week # | Message Title"
    3. Fall back to just the message title if no series info

    Format: "Series Name – Week # | Message Title"
    Example: "The Big Relief – Week 3 | Finding Peace"
    """
    # If podcastTitle is explicitly set, use it
    podcast_title = episode.get('podcastTitle', '')
    if podcast_title:
        return podcast_title

    # Build title from series info + message title
    message_title = episode.get('title', '')
    series_name = episode.get('seriesName', '')
    week_num = episode.get('weekNum')

    if series_name and week_num is not None:
        # Full format: "Series Name – Week # | Message Title"
        return f"{series_name} – Week {week_num} | {message_title}"
    elif series_name:
        # No week number: "Series Name | Message Title"
        return f"{series_name} | {message_title}"
    else:
        # No series info, just use message title
        return message_title


def build_item_xml(episode: dict) -> str:
    """Build an RSS <item> element from episode data."""
    # Build title using the proper format
    title = escape_xml(build_podcast_title(episode))
    description = escape_xml(episode.get('description', ''))
    audio_url = episode.get('audioUrl') or ''
    audio_size_raw = episode.get('audioFileSize') or 0
    duration_seconds = episode.get('audioDuration') or 0

    # Convert audio size to bytes (integer) for RSS enclosure length
    # MongoDB stores as MB (float) or bytes (int) - normalize to bytes
    if audio_size_raw and audio_size_raw < 10000:
        # Value is in MB (e.g., 24.68), convert to bytes
        audio_size = int(audio_size_raw * 1024 * 1024)
    else:
        # Value is already in bytes
        audio_size = int(audio_size_raw) if audio_size_raw else 0
    speaker = escape_xml(episode.get('speaker', ''))
    pub_date = episode.get('pubDate')
    # Default episode artwork - same as legacy Anchor.fm feed
    default_artwork = 'https://d3t3ozftmdmh3i.cloudfront.net/staging/podcast_uploaded_episode/45021512/91d19d6fa1414965.png'
    artwork_url = episode.get('artworkUrl') or default_artwork
    guid = episode.get('guid', episode.get('messageId', ''))

    # Format pubDate
    if isinstance(pub_date, datetime):
        pub_date_str = format_rfc2822_date(pub_date)
    elif isinstance(pub_date, str):
        try:
            dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            pub_date_str = format_rfc2822_date(dt)
        except:
            pub_date_str = pub_date
    else:
        pub_date_str = format_rfc2822_date(datetime.now(timezone.utc))

    duration_str = format_duration(duration_seconds)

    item = f"""    <item>
        <title>{title}</title>
        <description><![CDATA[{description}]]></description>
        <enclosure url="{audio_url}" length="{audio_size}" type="audio/mpeg"/>
        <guid isPermaLink="false">{guid}</guid>
        <pubDate>{pub_date_str}</pubDate>
        <itunes:author>{speaker}</itunes:author>
        <itunes:duration>{duration_str}</itunes:duration>
        <itunes:explicit>false</itunes:explicit>
        <itunes:image href="{artwork_url}"/>
    </item>"""

    return item


# =============================================================================
# S3 OPERATIONS
# =============================================================================

def get_current_feed() -> Optional[str]:
    """Download current RSS feed from S3."""
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=S3_FEED_KEY)
        return response['Body'].read().decode('utf-8')
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"Error downloading feed: {e}")
        return None


def upload_feed(xml_content: str):
    """Upload RSS feed to S3."""
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=S3_FEED_KEY,
        Body=xml_content.encode('utf-8'),
        ContentType='application/rss+xml',
        CacheControl='max-age=300'  # 5 minute cache
    )
    print(f"Uploaded feed to s3://{S3_BUCKET}/{S3_FEED_KEY}")


# =============================================================================
# RSS GENERATION - Always rebuilds from MongoDB (source of truth)
# =============================================================================

def generate_rss_feed(db) -> tuple[str, int]:
    """
    Generate RSS XML from all episodes in PodcastEpisodes collection.

    This is the single source of RSS generation. It queries all episodes
    from MongoDB and builds the complete RSS XML. Both upsert and rebuild
    actions use this function after their respective CRUD operations.

    Returns:
        tuple: (rss_xml_string, episode_count)
    """
    # Query all episodes with valid audio, sorted newest first
    episodes = list(db['PodcastEpisodes'].find(
        {'audioUrl': {'$exists': True, '$nin': [None, '']}},
        sort=[('pubDate', pymongo.DESCENDING)]
    ))

    print(f"Generating RSS feed with {len(episodes)} episodes")

    # Build RSS XML
    last_build = format_rfc2822_date(datetime.now(timezone.utc))
    header = CHANNEL_HEADER.format(
        year=datetime.now().year,
        last_build_date=last_build
    )

    items = [build_item_xml(episode) for episode in episodes]
    full_xml = header + '\n'.join(items) + '\n' + CHANNEL_FOOTER

    return full_xml, len(episodes)


# =============================================================================
# UPSERT ACTION - CRUD on PodcastEpisodes, then regenerate RSS
# =============================================================================

def upsert_episode(db, episode_data: dict) -> dict:
    """
    Add or update a single episode in PodcastEpisodes collection, then regenerate RSS.

    Pattern:
    1. Validate and prepare episode data
    2. Upsert to PodcastEpisodes (CRUD)
    3. Regenerate entire RSS from MongoDB (source of truth)
    """
    message_id = episode_data.get('messageId')
    audio_url = episode_data.get('audioUrl')

    # Validation: Skip episodes without audio - they can't be played
    if not audio_url:
        print(f"Skipping episode {message_id} - no audio URL")
        return {'action': 'upsert', 'messageId': message_id, 'status': 'skipped', 'reason': 'no_audio'}

    # Generate description if we have transcript but no description
    if episode_data.get('transcript') and not episode_data.get('description'):
        episode_data['description'] = generate_podcast_description(
            transcript=episode_data['transcript'],
            title=episode_data.get('title', ''),
            speaker=episode_data.get('speaker', ''),
            passage_ref=episode_data.get('passageRef', '')
        )

    # Remove transcript before saving - it's huge and only needed for description generation
    episode_data.pop('transcript', None)

    # Ensure pubDate is a proper datetime (MongoDB sorts these correctly)
    if 'pubDate' in episode_data:
        episode_data['pubDate'] = ensure_datetime(episode_data['pubDate'])
        if episode_data['pubDate'] is None:
            print(f"Warning: Could not parse pubDate for episode {message_id}")

    # Set createdAt timestamp
    if 'createdAt' not in episode_data:
        episode_data['createdAt'] = datetime.now(timezone.utc)
    else:
        episode_data['createdAt'] = ensure_datetime(episode_data['createdAt']) or datetime.now(timezone.utc)

    # CRUD: Upsert to PodcastEpisodes collection
    db['PodcastEpisodes'].update_one(
        {'messageId': message_id},
        {'$set': episode_data},
        upsert=True
    )
    print(f"Upserted episode {message_id} to PodcastEpisodes")

    # Generate and upload RSS feed
    rss_xml, episode_count = generate_rss_feed(db)
    upload_feed(rss_xml)

    return {
        'action': 'upsert',
        'messageId': message_id,
        'status': 'success',
        'episodeCount': episode_count
    }


# =============================================================================
# REBUILD ACTION - Just regenerate RSS from MongoDB (no CRUD)
# =============================================================================

def rebuild_feed(db) -> dict:
    """
    Regenerate entire RSS feed from PodcastEpisodes collection.
    No CRUD operations - just reads from MongoDB and rebuilds RSS.
    """
    rss_xml, episode_count = generate_rss_feed(db)
    upload_feed(rss_xml)

    return {
        'action': 'rebuild',
        'episodeCount': episode_count,
        'status': 'success'
    }


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def lambda_handler(event, context):
    """
    Main Lambda handler.

    Event structure:
    {
        "action": "upsert" | "rebuild",
        "episode": { ... }  // Only for "upsert" action
    }

    For upsert, episode should contain:
    {
        "messageId": "abc123",
        "title": "Sermon Title",
        "seriesName": "The Big Relief",  // Series name for title construction
        "weekNum": 3,                     // Week number for title construction
        "podcastTitle": null,             // Optional: explicit override for podcast title
        "audioUrl": "https://...",
        "audioFileSize": 38248061,
        "audioDuration": 2389,
        "speaker": "Pastor Name",
        "pubDate": "2025-06-15T10:00:00Z",
        "artworkUrl": "https://...",
        "guid": "abc123",
        "transcript": "Full transcript...",  // Optional, for description generation
        "description": "Pre-generated description"  // Optional, if already have one
    }

    Title Construction Priority:
    1. If podcastTitle is set, use it directly
    2. Otherwise, build: "Series Name – Week # | Message Title"
    3. Falls back to just "Message Title" if no series info
    """
    action = event.get('action', 'upsert')

    # Get Lambda function name for Langfuse tagging
    lambda_name = os.environ.get('AWS_LAMBDA_FUNCTION_NAME', 'local')

    client = None
    try:
        # Import propagate_attributes for Langfuse tagging
        from langfuse import propagate_attributes

        client = get_mongodb_client()
        db = client[DB_NAME]

        # Wrap all operations with Lambda function name tag (upsert may call LLM)
        with propagate_attributes(tags=[lambda_name]):
            if action == 'rebuild':
                result = rebuild_feed(db)
            elif action == 'upsert':
                episode_data = event.get('episode', {})
                if not episode_data.get('messageId'):
                    return {
                        'statusCode': 400,
                        'body': 'Missing messageId in episode data'
                    }
                result = upsert_episode(db, episode_data)
            else:
                return {
                    'statusCode': 400,
                    'body': f'Unknown action: {action}'
                }

        return {
            'statusCode': 200,
            'body': result
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
        if client:
            client.close()


# =============================================================================
# LOCAL TESTING
# =============================================================================

if __name__ == '__main__':
    # For local testing
    from dotenv import load_dotenv
    load_dotenv()

    # Test rebuild
    test_event = {'action': 'rebuild'}
    result = lambda_handler(test_event, None)
    print(result)
