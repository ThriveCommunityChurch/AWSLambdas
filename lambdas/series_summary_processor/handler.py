"""
Series Summary Processor Lambda

Generates an AI summary for a completed sermon series.
Invoked when a series EndDate is set (series concluded).

1. Fetches series from SermonSeries collection (get Name)
2. Fetches all messages from Messages collection where SeriesId matches
3. Projects only: Title, Summary, Tags, PassageRef (no transcripts/notes)
4. Builds prompt with series name + message summaries
5. Calls Azure OpenAI (gpt-5-mini) to generate series summary
6. Updates SermonSeries.Summary field in MongoDB

Environment Variables:
- MONGODB_SECRET_ARN: Secrets Manager ARN for MongoDB URI
- MONGODB_SECRET_KEY: Key within secret for MongoDB connection string
- OPENAI_SECRET_ARN: Secrets Manager ARN for Azure OpenAI API key
- OPENAI_PROVIDER: 'azure' or 'openai' (default: azure)
"""

import boto3
import json
import pymongo
import os
from typing import Optional, Dict, Any, List
from bson import ObjectId

# Configuration
DB_NAME = 'SermonSeries'
SERIES_COLLECTION = 'Series'
MESSAGES_COLLECTION = 'Messages'

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


def get_series_by_id(db, series_id: str) -> Optional[Dict[str, Any]]:
    """Fetch series from MongoDB by ID."""
    try:
        series = db[SERIES_COLLECTION].find_one({'_id': ObjectId(series_id)})
        return series
    except Exception as e:
        print(f"Error fetching series {series_id}: {e}")
        return None


def get_messages_for_series(db, series_id: str) -> List[Dict[str, Any]]:
    """Fetch all messages for a series, projecting only needed fields."""
    try:
        messages = list(db[MESSAGES_COLLECTION].find(
            {'SeriesId': series_id},
            {
                '_id': 0,
                'Title': 1,
                'Summary': 1,
                'Tags': 1,
                'PassageRef': 1
            }
        ))
        return messages
    except Exception as e:
        print(f"Error fetching messages for series {series_id}: {e}")
        return []


def build_prompt(series_name: str, messages: List[Dict[str, Any]]) -> str:
    """Build the prompt for series summary generation."""
    # Format message summaries
    message_summaries = []
    for i, msg in enumerate(messages, 1):
        parts = [f"{i}. \"{msg.get('Title', 'Untitled')}\""]
        if msg.get('PassageRef'):
            parts.append(f"   Passage: {msg['PassageRef']}")
        if msg.get('Summary'):
            parts.append(f"   Summary: {msg['Summary']}")
        if msg.get('Tags'):
            # Tags may be enum values or strings
            tags = msg['Tags']
            if tags:
                parts.append(f"   Topics: {', '.join(str(t) for t in tags)}")
        message_summaries.append('\n'.join(parts))

    messages_text = '\n\n'.join(message_summaries)

    return f"""You are summarizing a sermon series called "{series_name}" for prospective listeners.

Message summaries from this series:
{messages_text}

Generate a summary that:
- Informs prospective listeners about the series
- Uses creative, inviting language that encourages listening
- Challenges them to improve themselves or their faith
- Keeps language relatable, not complex or academic
- Focuses on the series as a whole collective unit
- Emphasizes learning outcomes when completing all messages
- Speaks directly to the listener ("you")
- Uses varied phrases - no repetition

DO NOT:
- Mention dates, times, or speakers
- Use phrases like: "you'll discover", "[series] is a journey through...", "By the end you'll...", "this series invites you...", "You'll rediscover..."

Output: Single paragraph, under 80 words."""


def generate_series_summary(series_name: str, messages: List[Dict[str, Any]]) -> Optional[str]:
    """Generate series summary using OpenAI."""
    if not messages:
        print(f"No messages found for series '{series_name}', skipping summary generation")
        return None

    prompt = build_prompt(series_name, messages)
    print(f"Generating summary for series '{series_name}' with {len(messages)} messages")

    try:
        client = get_openai_client()
        model = get_chat_model_name()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a creative copywriter for a church, writing engaging summaries that invite people to explore sermon series."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=200,
            temperature=0.7
        )

        summary = response.choices[0].message.content.strip()
        print(f"Generated summary ({len(summary)} chars): {summary[:100]}...")
        return summary

    except Exception as e:
        print(f"Error generating summary: {e}")
        return None


def update_series_summary(db, series_id: str, summary: str) -> bool:
    """Update the Summary field on a series document."""
    try:
        result = db[SERIES_COLLECTION].update_one(
            {'_id': ObjectId(series_id)},
            {'$set': {'Summary': summary}}
        )
        if result.modified_count > 0:
            print(f"Updated series {series_id} with new summary")
            return True
        else:
            print(f"Series {series_id} not modified (may already have same summary)")
            return True
    except Exception as e:
        print(f"Error updating series {series_id}: {e}")
        return False


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def lambda_handler(event, context):
    """
    Main Lambda handler.

    Event structure:
    {
        "seriesId": "507f1f77bcf86cd799439011"
    }

    Returns:
    {
        "statusCode": 200,
        "body": {
            "seriesId": "...",
            "seriesName": "...",
            "summaryLength": 150,
            "status": "success"
        }
    }
    """
    series_id = event.get('seriesId')

    if not series_id:
        return {
            'statusCode': 400,
            'body': 'Missing seriesId'
        }

    print(f"Processing series summary for: {series_id}")

    client = None
    try:
        # Connect to MongoDB
        mongo_uri = get_mongodb_uri()
        client = pymongo.MongoClient(mongo_uri)
        db = client[DB_NAME]

        # Step 1: Get series
        series = get_series_by_id(db, series_id)
        if not series:
            return {
                'statusCode': 404,
                'body': f'Series not found: {series_id}'
            }

        series_name = series.get('Name', 'Untitled Series')
        print(f"Found series: {series_name}")

        # Step 2: Get messages for this series
        messages = get_messages_for_series(db, series_id)
        print(f"Found {len(messages)} messages for series")

        # Step 3: Generate summary
        summary = generate_series_summary(series_name, messages)
        if not summary:
            return {
                'statusCode': 500,
                'body': 'Failed to generate summary'
            }

        # Step 4: Update series with summary
        success = update_series_summary(db, series_id, summary)
        if not success:
            return {
                'statusCode': 500,
                'body': 'Failed to update series'
            }

        return {
            'statusCode': 200,
            'body': {
                'seriesId': series_id,
                'seriesName': series_name,
                'summaryLength': len(summary),
                'messageCount': len(messages),
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


# Local testing
if __name__ == '__main__':
    # Test with a sample series ID
    test_event = {
        'seriesId': '67b2c80d69ff6a2b9baae97a'  # Replace with actual series ID
    }
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))

