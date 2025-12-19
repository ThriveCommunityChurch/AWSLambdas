"""
Sermon Processor Lambda

Receives transcript and messageId, then:
1. Generates sermon summary (GPT-4o mini)
2. Generates waveform data (librosa RMS) - if audio URL provided
3. Generates tags (GPT-4o mini)
4. Updates SermonMessages collection in MongoDB

Environment Variables:
- MONGODB_SECRET_ARN: Secrets Manager ARN for MongoDB URI
- OPENAI_SECRET_ARN: Secrets Manager ARN for OpenAI API key
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
COLLECTION_NAME = 'SermonMessages'

# AWS clients
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
    "Church", "EndTimes", "SinAndRepentance", "Sanctification", "Covenant",
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


def generate_sermon_summary(transcript: str, title: str, passage_ref: str = "") -> str:
    """
    Generate a single-paragraph, end-user-friendly summary of the sermon.
    Uses the sophisticated prompt from the LangGraph agent for high-quality output.
    """
    if not transcript:
        return ""

    # System prompt from LangGraph agent - focuses on what listeners will experience
    system_prompt = (
        "You are an expert at summarizing sermons for church audiences. "
        "Your task is to create a single-paragraph summary that captures what listeners will learn, "
        "experience, and take away from the sermon—not just a retelling of its content in a summarization. Focus on the "
        "insights, transformations, and practical wisdom available to those who engage with the message. "
        "\n\n"
        "Requirements:\n"
        "- Write for listeners of the sermon—as if speaking directly to them. Your tone should be conversational, invitational, and spiritually engaging\n"
        "- The summary must be a single paragraph (no line breaks within the summary)\n"
        "- The summary should be no more than 120 words maximum. Less is better if you can effectively communicate the core message\n"
        "- Use an opening question only if the sermon explicitly challenges behavior or worldview—otherwise, open with a statement that draws curiosity or empathy.\n"
        "- Frame the summary around what listeners will discover, learn, or be challenged by—not just what the sermon is about\n"
        "- Base your summary on what is explicitly or clearly implied in the transcription, without adding unstated ideas\n"
        "- DO NOT mention the church name, organization name, or pastor's name\n"
        "- DO NOT cite specific passage references directly (e.g., 'Matthew 2:1-12') unless the entire sermon is an exposition of a single passage\n"
        "- Vary your opening approach—avoid using the same opening verb or structure repeatedly. Use diverse, engaging starts that feel natural and specific to each sermon's unique message\n"
        "- Use creative phrases to avoid repetitiveness. Never use phrases like \"you'll discover\", \"You'll learn...\", \"In a world...\", \"is a journey through...\", \"By the end you'll..\", \"this message invites you...\", \"You'll rediscover...\", etc. Assume the listener is reading many of these at a time and we want to avoid repeating phrases. Be creative and think outside the box."
        "\n\n"
        "Guidelines:\n"
        "- Use language that helps listeners imagine themselves benefiting from the message using diverse, engaging and thought provoking language that feels natural and specific to each sermon's unique message\n"
        "- Focus on the transformation, insight, or practical wisdom the sermon offers\n"
        "- Capture the sermon's emotional tone and spiritual purpose\n"
        "- Be concise and punchy, avoiding dense or overly packed sentences\n"
        "- Keep sentences short and clear; if a sentence has multiple clauses, consider breaking the idea into simpler parts\n"
        "- Write to inform laypeople, not academics—content should be accessible but substantive\n"
        "- Write in an engaging tone that reflects the sermon's spirit and makes people want to listen\n"
        "- Balance accessibility with theological substance—explain concepts naturally without dumbing down\n"
        "- End with a closing sentence, or final insight or question that leaves the listener reflecting—avoid generic wrap-ups or repeating earlier phrases\n"
        "- Prioritize clarity, readability, and relatability for everyday people over religious jargon\n"
        "- Avoid simply retelling biblical stories or sermon content—focus on the takeaway and application\n"
        "- Reflect the emotional temperature of the sermon (e.g., convicting, comforting, celebratory, urgent) so each summary feels true to its heart\n"
        "- You may use light metaphor or imagery only when it directly reflects the sermon's expressed themes, not when inventing new symbolic language."
    )

    user_prompt = (
        f"Please summarize the following sermon transcription into a single paragraph "
        f"that captures its core message and purpose:\n\n{transcript}"
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=get_openai_api_key())

        print("Generating sermon summary with GPT-4o-mini...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=400,  # Increased for ~120 word paragraph
            temperature=0.3
        )

        summary_text = response.choices[0].message.content.strip()

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
    Generate semantic tags for the sermon using GPT-4o mini.
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
        '[\"Faith\", \"Prayer\", \"Suffering\", \"Hope\"]'
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
        from openai import OpenAI
        client = OpenAI(api_key=get_openai_api_key())

        print("Analyzing sermon content (summary + transcript) and applying tags with GPT-4o-mini...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100,
            temperature=0.2  # Lower temperature for more consistent tag selection
        )

        response_text = response.choices[0].message.content.strip()

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
    Download audio from S3 and generate waveform data using librosa.
    Returns list of normalized RMS values (0-1).
    """
    if not audio_url:
        return None

    try:
        import tempfile
        import librosa
        import numpy as np

        s3 = boto3.client('s3')

        # Parse S3 URL to get bucket and key
        # Format: https://thrive-audio.s3.amazonaws.com/2025/file.mp3
        # or s3://thrive-audio/2025/file.mp3
        if audio_url.startswith('s3://'):
            parts = audio_url[5:].split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
        elif 's3.amazonaws.com' in audio_url:
            # https://bucket.s3.amazonaws.com/key
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

        # Load audio and compute RMS
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)

        # Calculate frame length to get desired number of points
        hop_length = len(y) // num_points
        if hop_length < 1:
            hop_length = 1

        # Compute RMS energy
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # Normalize to 0-1 range
        if rms.max() > 0:
            rms = rms / rms.max()

        # Resample to exact number of points if needed
        if len(rms) != num_points:
            rms = np.interp(
                np.linspace(0, len(rms) - 1, num_points),
                np.arange(len(rms)),
                rms
            )

        # Clean up temp file
        os.unlink(tmp_path)

        return rms.tolist()

    except Exception as e:
        print(f"Error generating waveform: {e}")
        return None


def update_sermon_message(db, message_id: str, summary: str, tags: List[str],
                          waveform_data: Optional[List[float]] = None) -> bool:
    """Update SermonMessage document with generated content."""
    try:
        collection = db[COLLECTION_NAME]

        update_doc = {
            'Summary': summary,
            'Tags': tags,
            'LastUpdated': datetime.now(timezone.utc)
        }

        if waveform_data:
            update_doc['WaveformData'] = waveform_data

        result = collection.update_one(
            {'_id': ObjectId(message_id)},
            {'$set': update_doc}
        )

        if result.matched_count == 0:
            print(f"No document found with _id: {message_id}")
            return False

        print(f"Updated SermonMessage {message_id}: summary={len(summary)} chars, tags={tags}")
        return True

    except Exception as e:
        print(f"Error updating SermonMessage: {e}")
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
        "title": "Sermon Title",
        "passageRef": "John 3:16",
        "audioUrl": "https://thrive-audio.s3.amazonaws.com/2025/file.mp3",  // Optional
        "generateWaveform": true  // Optional, defaults to false
    }
    """
    message_id = event.get('messageId')
    transcript = event.get('transcript', '')
    title = event.get('title', '')
    passage_ref = event.get('passageRef', '')
    audio_url = event.get('audioUrl')
    generate_waveform = event.get('generateWaveform', False)

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
        summary = generate_sermon_summary(transcript, title, passage_ref)

        # Step 2: Generate tags using BOTH summary and transcript (hybrid approach)
        # This provides better context - summary for main themes, transcript for comprehensive coverage
        tags = generate_tags(summary, transcript, title)

        waveform_data = None
        if generate_waveform and audio_url:
            waveform_data = generate_waveform_from_s3(audio_url)

        # Update MongoDB
        client = get_mongodb_client()
        db = client[DB_NAME]

        success = update_sermon_message(db, message_id, summary, tags, waveform_data)

        if success:
            return {
                'statusCode': 200,
                'body': {
                    'messageId': message_id,
                    'summary': summary,
                    'tags': tags,
                    'hasWaveform': waveform_data is not None,
                    'status': 'success'
                }
            }
        else:
            return {
                'statusCode': 404,
                'body': f'Message not found: {message_id}'
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

