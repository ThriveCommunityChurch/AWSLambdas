# Thrive Church AWS Lambdas

AWS Lambda functions for Thrive Church automation, including podcast RSS feed generation and sermon processing.

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
│ Service.cs       │     │  • Whisper API       │     │  • Waveform        │
│                  │     │  • Get transcript    │     │  • Update MongoDB  │
└──────────────────┘     └──────────┬───────────┘     │    SermonMessages  │
                                    │                 └────────────────────┘
                                    │
                                    │                 ┌────────────────────┐
                                    └────────────────▶│  podcast_rss_      |
                                                      │  generator         │
                                                      │                    │
                                                      │  • Description(GPT)│
                                                      │  • Upsert MongoDB  │
                                                      │    PodcastEpisodes │
                                                      │  • Update RSS XML  │
                                                      │    to S3           │
                                                      └────────────────────┘
```

## Lambdas

| Lambda | Purpose |
|--------|---------|
| `transcription-processor` | Downloads audio, transcribes with Whisper API, invokes `sermon-processor` and `podcast-rss-generator` |
| `sermon-processor` | Generates sermon summaries, tags, waveforms |
| `podcast-rss-generator` | Generates podcast RSS feed from PodcastEpisodes |

## Project Structure

```
AWSLambdas/
├── README.md
├── requirements-dev.txt          # Development dependencies
├── lambdas/
│   ├── podcast_rss_generator/
│   │   ├── handler.py
│   │   └── requirements.txt
│   ├── sermon_processor/
│   │   ├── handler.py
│   │   └── requirements.txt
│   └── transcription_processor/
│       ├── handler.py
│       └── requirements.txt
├── scripts/
│   └── backlog_import.py         # One-time backlog import script
└── infrastructure/
    └── template.yaml             # CloudFormation/SAM template
```

## Local Development

### Prerequisites
- Python 3.11+
- AWS CLI configured
- MongoDB connection string

### Setup
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
```

### Environment Variables
```bash
MONGODB_URI=mongodb+srv://...
OPENAI_API_KEY=sk-...
AWS_REGION=us-east-1
```

## Deployment

Each Lambda is deployed independently using [AWS SAM](https://aws.amazon.com/serverless/sam/). See `template.yaml` for details.

## Related Projects

- [ThriveChurchOfficialAPI](../ThriveChurchOfficialAPI) - Main C# API
- [Sermon_Summarization_Agent](../Sermon_Summarization_Agent) - Local batch processing (backlog only)