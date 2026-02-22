# Thrive Church AWS Lambdas

AWS Lambda functions for Thrive Church automation, including podcast RSS feed generation, sermon processing, and AI-powered content enrichment.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SERMON PROCESSING PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│   .NET API       │     │  transcription_      │     │  sermon_processor  │
│  (App Runner)    │────▶│  processor           │───▶│                    │
│                  │     │                      │     │  • Summary (GPT)   │
│ PodcastLambda    │     │  • Download audio    │     │  • Tags (GPT)      │
│ Service.cs       │     │  • Azure Speech API  │     │  • Waveform        │
│                  │     │  • Store transcript  │     │  • Update MongoDB  │
└──────────────────┘     └──────────┬───────────┘     │    Messages        │
                                    │                 └─────────┬──────────┘
                                    │                           │
                                    │                           │ (if series has EndDate)
                                    │                           ▼
                                    │                 ┌────────────────────┐
                                    │                 │  series_summary_   │
                                    │                 │  processor         │
                                    │                 │                    │
                                    │                 │  • Series Summary  │
                                    │                 │    (GPT)           │
                                    │                 │  • Update MongoDB  │
                                    │                 │    Series          │
                                    │                 └────────────────────┘
                                    │
                                    │                 ┌────────────────────┐
                                    └────────────────▶│  podcast_rss_      │
                                                      │  generator         │
                                                      │                    │
                                                      │  • Description(GPT)│
                                                      │  • Upsert MongoDB  │
                                                      │    PodcastEpisodes │
                                                      │  • Update RSS XML  │
                                                      │    to S3           │
                                                      └────────────────────┘
```

## What Gets Generated

| Content Type | Where | Description |
|--------------|-------|-------------|
| **Message Summary** | MongoDB `Messages.Summary` | TLDR-style, 130-180 words, uses we/us/our perspective, leads with the topic/lesson |
| **Topical Tags** | MongoDB `Messages.Tags` | 90+ tags across 17 categories (Theological, Spiritual Disciplines, Personal Growth, etc.) |
| **Waveform Data** | MongoDB `Messages.WaveformData` | 200-point waveform for audio player visualization |
| **Series Summary** | MongoDB `Series.Summary` | Present-tense timeless truths about the series themes (generated when series is complete) |
| **Podcast Description** | MongoDB `PodcastEpisodes.Description`, S3 RSS XML | Two paragraphs (130-180 words) for non-church audience, no speaker names |

### Prompt Engineering

All prompts have been refined through systematic evaluation using [promptfoo](https://promptfoo.dev/). Key prompt characteristics:

- **Message Summaries**: TLDR-style, straightforward educational tone, first person plural (we/us/our)
- **Series Summaries**: Present-tense timeless truths, neutral contemplative tone, varied openings
- **Podcast Descriptions**: Two paragraphs for spiritual seekers, names specific tensions, ends with curiosity (not questions)
- **Tags**: Comprehensive categorization using predefined tag taxonomy mapped to C# enums

### Series Summary Flow

When a message is processed by `sermon_processor`, it checks if the message's series has an `EndDate` (indicating the series is complete). If so, it triggers `series_summary_processor` which:

1. Verifies all messages with audio in the series have summaries (race condition protection)
2. Aggregates all message summaries
3. Generates a cohesive series-level summary using GPT
4. Saves the summary to the Series document in MongoDB

## Lambdas

| Lambda | Purpose |
|--------|---------|
| `transcription-processor` | Downloads audio from S3, transcribes with Azure Speech API, stores transcript in Azure Blob, invokes `sermon-processor` and `podcast-rss-generator` |
| `sermon-processor` | Generates message summary (TLDR-style), tags (90+), waveform. Triggers `series-summary-processor` for completed series |
| `podcast-rss-generator` | Generates podcast descriptions (two paragraphs, no speaker names), updates PodcastEpisodes and RSS XML |
| `series-summary-processor` | Generates series-level summaries as present-tense timeless truths |

## Project Structure

```
AWSLambdas/
├── README.md
├── template.yaml                 # SAM template
├── requirements-dev.txt          # Development dependencies
├── lambdas/
│   ├── podcast_rss_generator/
│   │   ├── handler.py
│   │   ├── prompts/              # Prompt files
│   │   └── requirements.txt
│   ├── sermon_processor/
│   │   ├── handler.py
│   │   ├── prompts/              # Prompt files
│   │   └── requirements.txt
│   ├── series_summary_processor/
│   │   ├── handler.py
│   │   ├── prompts/              # Prompt files
│   │   └── requirements.txt
│   ├── transcription_processor/
│   │   ├── handler.py
│   │   └── requirements.txt
│   └── shared/                   # Shared utilities
│       └── ...
├── promptfoo/                    # Prompt evaluation configs and tests
│   ├── config_*.yaml             # Evaluation configurations
│   ├── prompts/                  # Prompt templates for evaluation
│   ├── tests/                    # Dynamic test generators
│   └── golden_examples/          # Reference examples
├── scripts/                      # Backfill and utility scripts
│   └── ...
├── tests/
│   └── ...
└── docs/
    └── ...
```

## Local Development

### Prerequisites
- Python 3.11+
- AWS CLI configured
- MongoDB connection string
- Langfuse (for LLM visibility)

### Setup
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
```

## Environment Variables

Environment variables are configured in `template.yaml` and injected at runtime. Secrets are stored in AWS Secrets Manager.

### Transcription Processor
| Variable | Description |
|----------|-------------|
| `S3_BUCKET` | S3 bucket for audio files (e.g., `thrive-audio`) |
| `SERMON_LAMBDA_NAME` | Name of sermon processor Lambda to invoke |
| `PODCAST_LAMBDA_NAME` | Name of podcast RSS generator Lambda to invoke |
| `AZURE_STORAGE_ACCOUNT` | Azure Blob Storage account for transcripts |
| `AZURE_STORAGE_CONTAINER` | Azure container name (e.g., `transcripts`) |
| `AZURE_SPEECH_ENDPOINT` | Azure Speech-to-Text endpoint |

### Sermon Processor
| Variable | Description |
|----------|-------------|
| `S3_BUCKET` | S3 bucket for audio files |
| `SERIES_SUMMARY_LAMBDA_NAME` | Name of series summary Lambda to invoke |

### Series Summary Processor
| Variable | Description |
|----------|-------------|
| *(uses Secrets Manager)* | MongoDB URI and OpenAI API key from Secrets Manager |

### Podcast RSS Generator
| Variable | Description |
|----------|-------------|
| `RSS_BUCKET` | S3 bucket for RSS XML file |
| `RSS_KEY` | S3 key for RSS XML file |

## Deployment

Lambdas are deployed using [AWS SAM](https://aws.amazon.com/serverless/sam/) via GitHub Actions CI/CD.

```bash
# Validate template
sam validate

# Build all Lambdas
sam build

# Deploy (requires AWS credentials)
sam deploy --guided
```

See `template.yaml` for full configuration including IAM roles, log groups, and Lambda layers.

## Related Projects

- [ThriveChurchOfficialAPI](https://github.com/ThriveCommunityChurch/ThriveChurchOfficialAPI) - Main C# API
- [ThriveAPIMediaTool](https://github.com/ThriveCommunityChurch/ThriveAPIMediaTool) - Admin Tool UI
