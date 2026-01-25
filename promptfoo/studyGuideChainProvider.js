const fs = require('fs');
const path = require('path');

// Load .env file manually
function loadEnvFile() {
  const envPath = path.join(__dirname, '.env');
  if (fs.existsSync(envPath)) {
    const envContent = fs.readFileSync(envPath, 'utf-8');
    for (const line of envContent.split('\n')) {
      const trimmedLine = line.trim();
      if (trimmedLine && !trimmedLine.startsWith('#')) {
        const [key, ...valueParts] = trimmedLine.split('=');
        const value = valueParts.join('=');
        if (key && value && !process.env[key]) {
          process.env[key] = value;
        }
      }
    }
  }
}
loadEnvFile();

// Load prompts from files
const devotionalPromptTemplate = fs.readFileSync(
  path.join(__dirname, 'prompts', 'devotional_prompt.txt'),
  'utf-8'
);
const studyGuidePromptTemplate = fs.readFileSync(
  path.join(__dirname, 'prompts', 'study_guide_prompt.txt'),
  'utf-8'
);

class StudyGuideChainProvider {
  constructor(options) {
    this.providerId = options?.id || 'study-guide-chain';
    this.config = options?.config || {};
  }

  id() {
    return this.providerId;
  }

  async callApi(prompt, context) {
    const vars = context?.vars || {};
    const apiKey = process.env.AZURE_API_KEY;
    const apiHost = this.config.apiHost || process.env.AZURE_API_HOST || 'domain.openai.azure.com';
    const apiVersion = this.config.apiVersion || process.env.AZURE_API_VERSION || '2024-10-21';

    // Replace template variables in devotional prompt
    let devotionalPrompt = devotionalPromptTemplate
      .replace('{{title}}', vars.title || '')
      .replace('{{speaker}}', vars.speaker || '')
      .replace('{{date}}', vars.date || '')
      .replace('{{transcript}}', vars.transcript || '');

    // STEP 1: Generate devotional with higher reasoning (with retry for short responses)
    let devotionalText = '';
    let devotionalTokenUsage = { total: 0, prompt: 0, completion: 0 };
    const maxRetries = 3;
    const minWordCount = 600;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      const devotionalResponse = await this.callAzureOpenAI(
        devotionalPrompt,
        apiHost,
        apiKey,
        apiVersion,
        {
          model: 'gpt-5-mini',
          max_tokens: 8000,
        }
      );

      if (devotionalResponse.error) {
        return { error: `Devotional generation failed: ${devotionalResponse.error}` };
      }

      devotionalText = devotionalResponse.output;
      devotionalTokenUsage = devotionalResponse.tokenUsage;

      // Check word count
      const wordCount = devotionalText.split(/\s+/).filter(w => w.length > 0).length;
      const finishReason = devotionalResponse.finishReason || 'unknown';

      console.log(`[Attempt ${attempt}] Devotional generated: ${wordCount} words, finish_reason: ${finishReason}`);

      if (wordCount >= minWordCount) {
        break; // Success, exit retry loop
      }

      if (attempt < maxRetries) {
        console.log(`[Attempt ${attempt}] Devotional too short (${wordCount} words), retrying...`);
        // Add explicit instruction to the prompt for retry
        devotionalPrompt = devotionalPromptTemplate
          .replace('{{title}}', vars.title || '')
          .replace('{{speaker}}', vars.speaker || '')
          .replace('{{date}}', vars.date || '')
          .replace('{{transcript}}', vars.transcript || '') +
          `\n\nIMPORTANT: Your previous response was too short (only ${wordCount} words). Please write a COMPLETE devotional with at least 700 words and 5-6 paragraphs. Do not stop mid-sentence.`;
      }
    }

    // STEP 2: Generate study guide with devotional as context (with retry for truncated JSON)
    let studyGuidePrompt = studyGuidePromptTemplate
      .replace('{{title}}', vars.title || '')
      .replace('{{speaker}}', vars.speaker || '')
      .replace('{{date}}', vars.date || '')
      .replace('{{transcript}}', vars.transcript || '');

    // Add the pre-generated devotional as context
    studyGuidePrompt += `\n\nPRE-GENERATED DEVOTIONAL (use this exactly as the "devotional" field in your JSON output):\n${devotionalText}`;

    let studyGuideOutput = '';
    let studyGuideTokenUsage = { total: 0, prompt: 0, completion: 0 };
    const maxStudyGuideRetries = 3;

    for (let attempt = 1; attempt <= maxStudyGuideRetries; attempt++) {
      const studyGuideResponse = await this.callAzureOpenAI(
        studyGuidePrompt,
        apiHost,
        apiKey,
        apiVersion,
        {
          model: 'gpt-5-mini',
          max_tokens: 16000,
          response_format: { type: 'json_object' },
        }
      );

      if (studyGuideResponse.error) {
        return { error: `Study guide generation failed: ${studyGuideResponse.error}` };
      }

      studyGuideOutput = studyGuideResponse.output;
      studyGuideTokenUsage = studyGuideResponse.tokenUsage;
      const finishReason = studyGuideResponse.finishReason || 'unknown';

      // Validate JSON
      let isValidJson = false;
      try {
        JSON.parse(studyGuideOutput);
        isValidJson = true;
      } catch (e) {
        isValidJson = false;
      }

      console.log(`[Study Guide Attempt ${attempt}] finish_reason: ${finishReason}, valid JSON: ${isValidJson}`);

      if (isValidJson) {
        break; // Success, exit retry loop
      }

      if (attempt < maxStudyGuideRetries) {
        console.log(`[Study Guide Attempt ${attempt}] Invalid JSON, retrying...`);
      }
    }

    // Combine token usage from both calls
    const totalTokenUsage = {
      total:
        (devotionalTokenUsage?.total || 0) +
        (studyGuideTokenUsage?.total || 0),
      prompt:
        (devotionalTokenUsage?.prompt || 0) +
        (studyGuideTokenUsage?.prompt || 0),
      completion:
        (devotionalTokenUsage?.completion || 0) +
        (studyGuideTokenUsage?.completion || 0),
    };

    return {
      output: studyGuideOutput,
      tokenUsage: totalTokenUsage,
      metadata: {
        devotionalTokens: devotionalTokenUsage,
        studyGuideTokens: studyGuideTokenUsage,
      },
    };
  }

  async callAzureOpenAI(prompt, apiHost, apiKey, apiVersion, config) {
    const url = `https://${apiHost}/openai/deployments/${config.model}/chat/completions?api-version=${apiVersion}`;

    const body = {
      messages: [{ role: 'user', content: prompt }],
      max_completion_tokens: config.max_tokens || 4000,
    };

    if (config.response_format) {
      body.response_format = config.response_format;
    }

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'api-key': apiKey,
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorText = await response.text();
        return { error: `API error ${response.status}: ${errorText}` };
      }

      const data = await response.json();
      const finishReason = data.choices[0].finish_reason;

      return {
        output: data.choices[0].message.content,
        finishReason: finishReason,
        tokenUsage: {
          total: data.usage?.total_tokens || 0,
          prompt: data.usage?.prompt_tokens || 0,
          completion: data.usage?.completion_tokens || 0,
        },
      };
    } catch (err) {
      return { error: err.message };
    }
  }
}

module.exports = StudyGuideChainProvider;

