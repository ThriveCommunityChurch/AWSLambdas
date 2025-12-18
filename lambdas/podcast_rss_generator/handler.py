"""
Podcast RSS Generator Lambda

Handles two actions:
- upsert: Add/update a single episode and update RSS XML
- rebuild: Regenerate entire RSS feed from PodcastEpisodes collection

Environment Variables:
- MONGODB_SECRET_ARN: Secrets Manager ARN for MongoDB URI
- OPENAI_SECRET_ARN: Secrets Manager ARN for OpenAI API key
- S3_BUCKET: S3 bucket name (default: thrive-audio)
- S3_FEED_KEY: S3 key for RSS feed (default: feed/rss.xml)
"""

import boto3
import json
import pymongo
import os
import re
from datetime import datetime, timezone
from email.utils import formatdate
from typing import Optional
from html import escape

# Configuration
S3_BUCKET = os.environ.get('S3_BUCKET', 'thrive-audio')
S3_FEED_KEY = os.environ.get('S3_FEED_KEY', 'feed/rss.xml')
DB_NAME = 'SermonSeries'

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
    """Get OpenAI API key from Secrets Manager."""
    return get_secret(os.environ['OPENAI_SECRET_ARN'])


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
    <itunes:image href="https://thrive-fl.org/assets/podcast-artwork.jpg"/>
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


def format_duration(seconds: int) -> str:
    """Convert seconds to HH:MM:SS format for iTunes."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
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


def generate_podcast_description(transcript: str, title: str, speaker: str, passage_ref: str = "") -> str:
    """Generate a podcast-friendly description using GPT-4o mini."""
    if not transcript:
        # Fallback description if no transcript
        return f"Join {speaker} as they share a powerful message titled '{title}'."

    # Use first ~6000 chars of transcript (within token limits)
    transcript_excerpt = transcript[:6000]

    prompt = f"""You are writing a podcast episode description for a church sermon.
Create an engaging 2-3 paragraph description that:
- Summarizes the main themes and takeaways
- Is warm and inviting
- Encourages listeners to tune in
- Does NOT start with "In this episode" or similar clichés

Title: {title}
Speaker: {speaker}
Scripture Reference: {passage_ref or 'N/A'}

Transcript excerpt:
{transcript_excerpt}

Write the description in a warm, welcoming tone. Do not include the title or speaker name at the start."""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=get_openai_api_key())

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant writing podcast descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating description: {e}")
        return f"Join {speaker} as they share a powerful message titled '{title}'."



def build_item_xml(episode: dict) -> str:
    """Build an RSS <item> element from episode data."""
    title = escape_xml(episode.get('title', ''))
    description = escape_xml(episode.get('description', ''))
    audio_url = episode.get('audioUrl', '')
    audio_size = episode.get('audioFileSize', 0)
    duration_seconds = episode.get('audioDuration', 0)
    speaker = escape_xml(episode.get('speaker', ''))
    pub_date = episode.get('pubDate')
    artwork_url = episode.get('artworkUrl', '')
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
        <itunes:explicit>false</itunes:explicit>"""

    if artwork_url:
        item += f"""
        <itunes:image href="{artwork_url}"/>"""

    item += """
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
# UPSERT ACTION - Update single episode in existing feed
# =============================================================================

def upsert_episode(db, episode_data: dict) -> dict:
    """
    Add or update a single episode in the RSS feed.
    1. Generate description if transcript provided but no description
    2. Upsert to PodcastEpisodes collection
    3. Update RSS XML (find by GUID, replace or append)
    """
    message_id = episode_data.get('messageId')

    # Generate description if we have transcript but no description
    if episode_data.get('transcript') and not episode_data.get('description'):
        episode_data['description'] = generate_podcast_description(
            transcript=episode_data['transcript'],
            title=episode_data.get('title', ''),
            speaker=episode_data.get('speaker', ''),
            passage_ref=episode_data.get('passageRef', '')
        )

    # Add createdAt if new
    if 'createdAt' not in episode_data:
        episode_data['createdAt'] = datetime.now(timezone.utc)

    # Upsert to MongoDB
    db['PodcastEpisodes'].update_one(
        {'messageId': message_id},
        {'$set': episode_data},
        upsert=True
    )
    print(f"Upserted episode {message_id} to MongoDB")

    # Get current feed
    current_feed = get_current_feed()

    if not current_feed:
        # No existing feed, do a full rebuild
        return rebuild_feed(db)

    # Build new item XML
    new_item = build_item_xml(episode_data)
    guid = episode_data.get('guid', message_id)

    # Try to find and replace existing item by GUID
    guid_pattern = rf'<item>.*?<guid[^>]*>{re.escape(guid)}</guid>.*?</item>'

    if re.search(guid_pattern, current_feed, re.DOTALL):
        # Replace existing item
        updated_feed = re.sub(guid_pattern, new_item.strip(), current_feed, flags=re.DOTALL)
        print(f"Replaced existing item with GUID {guid}")
    else:
        # Append new item after opening channel items (before first </item> or </channel>)
        # Insert after lastBuildDate
        insert_pattern = r'(</lastBuildDate>)'
        updated_feed = re.sub(insert_pattern, r'\1\n' + new_item, current_feed, count=1)
        print(f"Appended new item with GUID {guid}")

    # Update lastBuildDate
    last_build = format_rfc2822_date(datetime.now(timezone.utc))
    updated_feed = re.sub(
        r'<lastBuildDate>.*?</lastBuildDate>',
        f'<lastBuildDate>{last_build}</lastBuildDate>',
        updated_feed
    )

    # Upload updated feed
    upload_feed(updated_feed)

    return {'action': 'upsert', 'messageId': message_id, 'status': 'success'}


# =============================================================================
# REBUILD ACTION - Regenerate entire feed from MongoDB
# =============================================================================

def rebuild_feed(db) -> dict:
    """
    Regenerate entire RSS feed from PodcastEpisodes collection.
    Used for initial migration or recovery.
    """
    # Query all episodes, sorted by date descending (newest first)
    episodes = list(db['PodcastEpisodes'].find().sort('pubDate', pymongo.DESCENDING))

    print(f"Rebuilding feed with {len(episodes)} episodes")

    # Build header
    last_build = format_rfc2822_date(datetime.now(timezone.utc))
    header = CHANNEL_HEADER.format(
        year=datetime.now().year,
        last_build_date=last_build
    )

    # Build all items
    items = []
    for episode in episodes:
        items.append(build_item_xml(episode))

    # Combine into full XML
    full_xml = header + '\n'.join(items) + '\n' + CHANNEL_FOOTER

    # Upload to S3
    upload_feed(full_xml)

    return {
        'action': 'rebuild',
        'episodeCount': len(episodes),
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
    """
    action = event.get('action', 'upsert')

    client = None
    try:
        client = get_mongodb_client()
        db = client[DB_NAME]

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
