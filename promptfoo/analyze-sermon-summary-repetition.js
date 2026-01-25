/**
 * Cross-Example Repetition Analysis for Sermon Summaries
 * 
 * Detects patterns that repeat across multiple sermon summaries:
 * - Same first word/phrase ("Your...", "You...")
 * - Same sentence structure ("Your _ isn't _")
 * - Similar openings across examples
 * 
 * Usage: node analyze-sermon-summary-repetition.js [results.json]
 * 
 * Exit codes:
 *   0 = No high-severity issues (pass)
 *   1 = High-severity repetition detected (fail)
 */

const fs = require('fs');
const path = require('path');

const CONFIG = {
  // Max allowed summaries starting with same word (excluding articles)
  maxSameFirstWord: 2,
  // Max allowed summaries with same 3-word opening pattern
  maxSameThreeWordPattern: 1,
  // Words to ignore when checking first word
  ignoredFirstWords: ['a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'with'],
  // Second-person words that should NOT start summaries (we want third-person)
  bannedFirstWords: ['you', 'your'],
  // Structural patterns to detect (regex)
  structuralPatterns: [
    { name: "Your _ isn't _", regex: /^your \w+ isn't/i },
    { name: "You don't need to _", regex: /^you don't need to/i },
    { name: "You face a choice", regex: /^you face a choice/i },
    { name: "You are called", regex: /^you are called/i },
    { name: "Starts with You/Your", regex: /^(you|your)\b/i },
    { name: "Scripture teaches", regex: /^scripture teaches/i },
    { name: "The Bible says", regex: /^the bible says/i },
  ],
};

function loadResults(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function extractSummaries(results) {
  const summaries = [];
  for (const result of results.results?.results || []) {
    const output = result.response?.output;
    if (output?.summary) {
      summaries.push({
        title: result.vars?.title || 'Unknown',
        summary: output.summary,
      });
    }
  }
  return summaries;
}

function analyzeFirstWords(summaries) {
  const issues = [];
  const firstWordCounts = {};

  for (const s of summaries) {
    const firstWord = s.summary.trim().split(/\s+/)[0].toLowerCase().replace(/[^a-z]/g, '');
    if (CONFIG.ignoredFirstWords.includes(firstWord)) continue;
    if (!firstWordCounts[firstWord]) firstWordCounts[firstWord] = [];
    firstWordCounts[firstWord].push(s.title);
  }

  for (const [word, titles] of Object.entries(firstWordCounts)) {
    if (titles.length > CONFIG.maxSameFirstWord) {
      issues.push({
        type: 'same_first_word',
        severity: titles.length >= 4 ? 'high' : 'medium',
        message: `${titles.length}/${summaries.length} summaries start with "${word}"`,
        titles,
      });
    }
  }
  return issues;
}

function analyzeThreeWordOpenings(summaries) {
  const issues = [];
  const patternCounts = {};

  for (const s of summaries) {
    const words = s.summary.trim().split(/\s+/).slice(0, 3).join(' ').toLowerCase();
    if (!patternCounts[words]) patternCounts[words] = [];
    patternCounts[words].push(s.title);
  }

  for (const [pattern, titles] of Object.entries(patternCounts)) {
    if (titles.length > CONFIG.maxSameThreeWordPattern) {
      issues.push({
        type: 'same_three_word_opening',
        severity: 'high',
        message: `${titles.length} summaries start with "${pattern}"`,
        titles,
      });
    }
  }
  return issues;
}

function analyzeStructuralPatterns(summaries) {
  const issues = [];

  for (const pattern of CONFIG.structuralPatterns) {
    const matches = summaries.filter(s => pattern.regex.test(s.summary));
    if (matches.length > 1) {
      issues.push({
        type: 'structural_pattern',
        severity: matches.length >= 3 ? 'high' : 'medium',
        message: `Pattern "${pattern.name}" found in ${matches.length} summaries`,
        titles: matches.map(m => m.title),
      });
    }
  }
  return issues;
}

function printReport(summaries, issues) {
  console.log('\n' + '='.repeat(70));
  console.log('SERMON SUMMARY CROSS-EXAMPLE REPETITION ANALYSIS');
  console.log('='.repeat(70));
  console.log(`\nAnalyzed ${summaries.length} sermon summaries\n`);

  console.log('SUMMARY OPENINGS:');
  console.log('-'.repeat(50));
  for (const s of summaries) {
    const opening = s.summary.split(/\s+/).slice(0, 8).join(' ');
    console.log(`  [${s.title}]: "${opening}..."`);
  }

  console.log('\n' + '='.repeat(70));

  if (issues.length === 0) {
    console.log('✅ NO CROSS-EXAMPLE REPETITION ISSUES DETECTED');
    console.log('='.repeat(70) + '\n');
    return true;
  }

  const high = issues.filter(i => i.severity === 'high');
  const medium = issues.filter(i => i.severity === 'medium');

  console.log(`⚠️  FOUND ${issues.length} ISSUES (${high.length} high, ${medium.length} medium)\n`);

  if (high.length > 0) {
    console.log('HIGH SEVERITY (must fix):');
    for (const issue of high) {
      console.log(`  ❌ ${issue.message}`);
      console.log(`     Sermons: ${issue.titles.join(', ')}`);
    }
  }

  if (medium.length > 0) {
    console.log('\nMEDIUM SEVERITY (should improve):');
    for (const issue of medium) {
      console.log(`  ⚠️  ${issue.message}`);
      console.log(`     Sermons: ${issue.titles.join(', ')}`);
    }
  }

  console.log('\n' + '='.repeat(70) + '\n');
  return high.length === 0;
}

// Main
const resultsFile = process.argv[2] || 'results_sermon_summary_repetition.json';
const resultsPath = path.resolve(__dirname, resultsFile);

if (!fs.existsSync(resultsPath)) {
  console.error(`Error: ${resultsPath} not found`);
  console.error('Run: npx promptfoo eval -c config_sermon_summary_repetition.yaml --output results_sermon_summary_repetition.json --no-cache');
  process.exit(1);
}

const summaries = extractSummaries(loadResults(resultsPath));
if (summaries.length < 3) {
  console.error('Error: Need at least 3 outputs for cross-example analysis');
  process.exit(1);
}

const issues = [
  ...analyzeFirstWords(summaries),
  ...analyzeThreeWordOpenings(summaries),
  ...analyzeStructuralPatterns(summaries),
];

const passed = printReport(summaries, issues);
process.exit(passed ? 0 : 1);

