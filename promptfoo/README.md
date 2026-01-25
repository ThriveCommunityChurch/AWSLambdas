# Sermon Prompt Evaluation with Promptfoo

This directory contains promptfoo configurations for evaluating and improving the sermon notes and study guide generation prompts.

## Setup

1. **Install promptfoo** (already installed globally):
   ```bash
   npm install -g promptfoo
   ```

2. **Configure environment variables**:
   ```bash
   # Copy the example env file
   cp .env.example .env
   
   # Edit .env with your Azure OpenAI credentials
   ```

3. **Set the environment variable** before running:
   ```powershell
   # PowerShell
   $env:AZURE_OPENAI_API_KEY = "your-api-key-here"
   ```
   
   ```bash
   # Bash
   export AZURE_OPENAI_API_KEY="your-api-key-here"
   ```

## Running Evaluations

### Basic evaluation:
```bash
cd promptfoo
promptfoo eval
```

### View results in browser:
```bash
promptfoo view
```

### Output to file:
```bash
promptfoo eval -o output.html
```

## Configuration Files

- `promptfooconfig.yaml` - Main configuration with providers, prompts, and test cases
- `prompts/notes_prompt.txt` - Sermon notes generation prompt
- `prompts/study_guide_prompt.txt` - Study guide generation prompt
- `test_data/` - Sample sermon transcripts for testing

## Evaluation Criteria

The evaluations check for:

1. **Banned phrases** (should NOT appear):
   - "You are invited to..."
   - "This message invites you to..."
   - "You are called to..."
   - "You are encouraged to..."
   - "You are challenged to..."

2. **Quality criteria**:
   - Direct and instructional tone (not invitational)
   - Varied sentence structures (not always starting with "Stop")
   - Present tense, speaking to the reader
   - Specific to the sermon content, not generic

3. **Structure validation**:
   - Valid JSON output
   - Required fields present
   - Minimum items in arrays (keyPoints >= 2, etc.)

## Adding New Test Cases

Add new test cases to `promptfooconfig.yaml`:

```yaml
tests:
  - vars:
      title: 'Sermon Title'
      speaker: 'Speaker Name'
      date: '2024-01-01'
      transcript: file://test_data/your_transcript.txt
    assert:
      - type: javascript
        value: |
          // Custom assertion
          try {
            const json = JSON.parse(output);
            return json.keyPoints && json.keyPoints.length >= 2;
          } catch (e) {
            return false;
          }
```

## Comparing Prompt Variations

Create prompt variations by copying and modifying the prompt files, then add them to the `prompts` array in `promptfooconfig.yaml`:

```yaml
prompts:
  - file://prompts/notes_prompt.txt
  - file://prompts/notes_prompt_v2.txt
```

Promptfoo will run both prompts against all test cases, allowing you to compare results side by side.

