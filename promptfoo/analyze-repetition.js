/**
 * Repetition Analysis Script for Promptfoo Results
 * 
 * Analyzes multiple sermon outputs to detect repetitive patterns:
 * - Similar summary openings
 * - Repeated phrases across outputs
 * - Structural patterns that indicate templated responses
 * 
 * Usage: node analyze-repetition.js [results.json]
 */

const fs = require('fs');
const path = require('path');

// Configuration
const CONFIG = {
  // Minimum similarity threshold for flagging (0-1)
  openingSimilarityThreshold: 0.5,
  // Number of words to compare in opening
  openingWordCount: 8,
  // Phrases that indicate templated/repetitive output
  repetitivePhrases: [
    'this sermon explores',
    'this message examines',
    'the passage reveals',
    'you are invited to',
    'you are invited into',
    'in this teaching',
    'the text shows',
    'we see that',
    'this study examines',
    'the scripture teaches',
    'right now you',
    'step into',
    'join us as',
    'come discover',
  ],
  // Minimum occurrences to flag a phrase as repetitive
  minOccurrences: 2,
};

function loadResults(filePath) {
  const data = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(data);
}

function extractSummaries(results) {
  const summaries = [];
  for (const result of results.results.results || []) {
    const output = result.response?.output;
    if (output && output.summary) {
      summaries.push({
        title: result.vars?.title || 'Unknown',
        summary: output.summary,
        keyPoints: output.keyPoints || [],
      });
    }
  }
  return summaries;
}

function getOpeningWords(text, count) {
  return text.trim().split(/\s+/).slice(0, count).join(' ').toLowerCase();
}

function calculateSimilarity(str1, str2) {
  const words1 = str1.toLowerCase().split(/\s+/);
  const words2 = str2.toLowerCase().split(/\s+/);
  const set1 = new Set(words1);
  const set2 = new Set(words2);
  const intersection = [...set1].filter(w => set2.has(w));
  const union = new Set([...words1, ...words2]);
  return intersection.length / union.size;
}

function analyzeOpenings(summaries) {
  const issues = [];
  const openings = summaries.map(s => ({
    title: s.title,
    opening: getOpeningWords(s.summary, CONFIG.openingWordCount),
  }));

  // Compare each pair
  for (let i = 0; i < openings.length; i++) {
    for (let j = i + 1; j < openings.length; j++) {
      const similarity = calculateSimilarity(openings[i].opening, openings[j].opening);
      if (similarity >= CONFIG.openingSimilarityThreshold) {
        issues.push({
          type: 'similar_opening',
          severity: similarity > 0.7 ? 'high' : 'medium',
          message: `Similar openings detected (${(similarity * 100).toFixed(0)}% similar)`,
          details: {
            sermon1: openings[i].title,
            opening1: openings[i].opening,
            sermon2: openings[j].title,
            opening2: openings[j].opening,
          },
        });
      }
    }
  }
  return issues;
}

function analyzeRepetitivePhrases(summaries) {
  const issues = [];
  const phraseOccurrences = {};

  for (const phrase of CONFIG.repetitivePhrases) {
    phraseOccurrences[phrase] = [];
  }

  for (const s of summaries) {
    const textLower = s.summary.toLowerCase();
    for (const phrase of CONFIG.repetitivePhrases) {
      if (textLower.includes(phrase)) {
        phraseOccurrences[phrase].push(s.title);
      }
    }
  }

  for (const [phrase, sermons] of Object.entries(phraseOccurrences)) {
    if (sermons.length >= CONFIG.minOccurrences) {
      issues.push({
        type: 'repetitive_phrase',
        severity: sermons.length >= 3 ? 'high' : 'medium',
        message: `Phrase "${phrase}" appears in ${sermons.length} outputs`,
        details: { phrase, sermons },
      });
    }
  }
  return issues;
}

function analyzeFirstWords(summaries) {
  const issues = [];
  const firstWords = {};
  // Common articles/determiners that are acceptable to repeat
  const ignoredFirstWords = ['a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'with'];

  for (const s of summaries) {
    const firstWord = s.summary.trim().split(/\s+/)[0].toLowerCase();
    if (ignoredFirstWords.includes(firstWord)) continue; // Skip common articles
    if (!firstWords[firstWord]) firstWords[firstWord] = [];
    firstWords[firstWord].push(s.title);
  }

  for (const [word, sermons] of Object.entries(firstWords)) {
    if (sermons.length >= CONFIG.minOccurrences) {
      issues.push({
        type: 'same_first_word',
        severity: sermons.length >= 3 ? 'high' : 'medium',
        message: `${sermons.length} summaries start with "${word}"`,
        details: { word, sermons },
      });
    }
  }
  return issues;
}

function printReport(summaries, issues) {
  console.log('\n' + '='.repeat(60));
  console.log('REPETITION ANALYSIS REPORT');
  console.log('='.repeat(60));
  console.log(`\nAnalyzed ${summaries.length} sermon outputs\n`);

  // Show all openings for reference
  console.log('SUMMARY OPENINGS:');
  console.log('-'.repeat(40));
  for (const s of summaries) {
    const opening = getOpeningWords(s.summary, CONFIG.openingWordCount);
    console.log(`  [${s.title}]: "${opening}..."`);
  }

  console.log('\n' + '='.repeat(60));
  
  if (issues.length === 0) {
    console.log('✅ NO REPETITION ISSUES DETECTED');
    console.log('='.repeat(60) + '\n');
    return true;
  }

  console.log(`⚠️  FOUND ${issues.length} POTENTIAL ISSUES:\n`);
  
  const highSeverity = issues.filter(i => i.severity === 'high');
  const mediumSeverity = issues.filter(i => i.severity === 'medium');

  if (highSeverity.length > 0) {
    console.log('HIGH SEVERITY:');
    for (const issue of highSeverity) {
      console.log(`  ❌ ${issue.message}`);
      if (issue.details.sermon1) {
        console.log(`     - "${issue.details.sermon1}": ${issue.details.opening1}`);
        console.log(`     - "${issue.details.sermon2}": ${issue.details.opening2}`);
      }
      if (issue.details.sermons) {
        console.log(`     Sermons: ${issue.details.sermons.join(', ')}`);
      }
    }
  }

  if (mediumSeverity.length > 0) {
    console.log('\nMEDIUM SEVERITY:');
    for (const issue of mediumSeverity) {
      console.log(`  ⚠️  ${issue.message}`);
      if (issue.details.sermons) {
        console.log(`     Sermons: ${issue.details.sermons.join(', ')}`);
      }
    }
  }

  console.log('\n' + '='.repeat(60) + '\n');
  return highSeverity.length === 0;
}

// Main execution
const resultsFile = process.argv[2] || 'results.json';
const resultsPath = path.resolve(__dirname, resultsFile);

if (!fs.existsSync(resultsPath)) {
  console.error(`Error: Results file not found: ${resultsPath}`);
  console.error('Run: npx promptfoo eval -c config_notes.yaml --output results.json');
  process.exit(1);
}

const results = loadResults(resultsPath);
const summaries = extractSummaries(results);

if (summaries.length < 2) {
  console.error('Error: Need at least 2 outputs to analyze repetition');
  process.exit(1);
}

const issues = [
  ...analyzeOpenings(summaries),
  ...analyzeRepetitivePhrases(summaries),
  ...analyzeFirstWords(summaries),
];

const passed = printReport(summaries, issues);
process.exit(passed ? 0 : 1);

