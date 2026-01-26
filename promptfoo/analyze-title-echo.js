/**
 * Title Echo Analysis for Sermon Summaries
 * 
 * Detects patterns where summaries parrot or echo the sermon title:
 * - Title words appearing in first sentence
 * - "[Title]: ..." colon format
 * - Reflective phrases: "is presented as", "frames", "centers on"
 * - Passive meta-commentary patterns
 * 
 * Usage: node analyze-title-echo.js
 * 
 * Reads from scripts/.transcript_cache/*.json
 */

const fs = require('fs');
const path = require('path');

const CONFIG = {
  cacheDir: path.resolve(__dirname, '../scripts/.transcript_cache'),
  // Phrases that indicate reflective/meta commentary (bad)
  reflectivePhrases: [
    'is presented as',
    'is identified as',
    'is confessed as',
    'is framed as',
    'centers on',
    'frames',
    'core insight',
    'core teaching',
    'main point',
    'sermon shows',
    'sermon explores',
    'sermon teaches',
    'passage shows',
  ],
  // Words to ignore when checking title echo
  stopWords: ['a', 'an', 'the', 'of', 'for', 'to', 'and', 'in', 'is', 'it', 'on', 'at', 'by', 'or'],
};

function loadCachedTranscripts() {
  const files = fs.readdirSync(CONFIG.cacheDir).filter(f => f.endsWith('.json'));
  const transcripts = [];
  
  for (const file of files) {
    try {
      const content = JSON.parse(fs.readFileSync(path.join(CONFIG.cacheDir, file), 'utf8'));
      if (content.title && content.notes?.summary) {
        transcripts.push({
          id: file.replace('.json', ''),
          title: content.title,
          summary: content.notes.summary,
        });
      }
    } catch (e) {
      // Skip invalid files
    }
  }
  return transcripts;
}

function getTitleWords(title) {
  return title.toLowerCase()
    .replace(/[^a-z\s]/g, '')
    .split(/\s+/)
    .filter(w => w.length > 2 && !CONFIG.stopWords.includes(w));
}

function getFirstSentence(summary) {
  const firstPeriod = summary.indexOf('.');
  return firstPeriod > 0 ? summary.substring(0, firstPeriod) : summary.substring(0, 100);
}

function analyzeTitleEcho(item) {
  const issues = [];
  const firstSentence = getFirstSentence(item.summary).toLowerCase();
  const titleWords = getTitleWords(item.title);
  
  // Check for title words in first sentence
  const matchedWords = titleWords.filter(w => firstSentence.includes(w));
  const echoRatio = matchedWords.length / titleWords.length;
  
  if (echoRatio >= 0.5 && titleWords.length >= 2) {
    issues.push({
      type: 'title_echo',
      severity: echoRatio >= 0.75 ? 'high' : 'medium',
      message: `Title echo: ${Math.round(echoRatio * 100)}% of title words in first sentence`,
      matched: matchedWords,
    });
  }
  
  // Check for "[Title]: ..." pattern
  if (item.summary.match(new RegExp(`^${item.title}\\s*:`, 'i'))) {
    issues.push({
      type: 'colon_title',
      severity: 'high',
      message: 'Summary starts with "Title: ..." format',
    });
  }
  
  // Check for reflective phrases
  for (const phrase of CONFIG.reflectivePhrases) {
    if (firstSentence.includes(phrase.toLowerCase())) {
      issues.push({
        type: 'reflective_phrase',
        severity: 'high',
        message: `Reflective phrase: "${phrase}"`,
      });
    }
  }
  
  return issues;
}

function printReport(transcripts) {
  console.log('\n' + '='.repeat(70));
  console.log('SERMON SUMMARY TITLE ECHO ANALYSIS');
  console.log('='.repeat(70));
  console.log(`\nAnalyzed ${transcripts.length} sermon summaries\n`);
  
  const allIssues = [];
  
  for (const item of transcripts) {
    const issues = analyzeTitleEcho(item);
    if (issues.length > 0) {
      allIssues.push({ ...item, issues });
    }
  }
  
  const highSeverity = allIssues.filter(i => i.issues.some(x => x.severity === 'high'));
  const mediumSeverity = allIssues.filter(i => !i.issues.some(x => x.severity === 'high'));
  
  console.log(`FOUND ${allIssues.length} summaries with issues (${highSeverity.length} high, ${mediumSeverity.length} medium)\n`);
  
  if (highSeverity.length > 0) {
    console.log('HIGH SEVERITY (must fix):');
    console.log('-'.repeat(50));
    for (const item of highSeverity.slice(0, 15)) {
      console.log(`\nTITLE: ${item.title}`);
      console.log(`FIRST: ${getFirstSentence(item.summary).substring(0, 100)}...`);
      for (const issue of item.issues) {
        console.log(`  ❌ ${issue.message}`);
      }
    }
  }
  
  // Summary statistics
  console.log('\n' + '='.repeat(70));
  console.log('PATTERN BREAKDOWN:');
  const patternCounts = {};
  for (const item of allIssues) {
    for (const issue of item.issues) {
      patternCounts[issue.type] = (patternCounts[issue.type] || 0) + 1;
    }
  }
  for (const [pattern, count] of Object.entries(patternCounts)) {
    console.log(`  ${pattern}: ${count}`);
  }
  console.log('='.repeat(70) + '\n');
}

// Main
const transcripts = loadCachedTranscripts();
if (transcripts.length < 5) {
  console.error('Error: Need at least 5 cached transcripts for analysis');
  process.exit(1);
}

printReport(transcripts);

