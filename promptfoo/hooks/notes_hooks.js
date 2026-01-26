/**
 * Extension hooks for notes evaluation.
 * Detects repetitive patterns in summary field and semicolon patterns.
 * 
 * Usage: Add to config YAML:
 *   extensions:
 *     - file://hooks/notes_hooks.js:extensionHook
 */

const CONFIG = {
  // Max percentage of summaries that can start with same word
  maxSameFirstWordPercent: 25,
  // Max allowed summaries with same 3-word opening
  maxSameThreeWordOpening: 2,
  // Words to ignore when checking first word (articles, prepositions)
  ignoredFirstWords: ['a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'with'],
};

function extractSummaries(results) {
  const summaries = [];

  for (const result of results || []) {
    try {
      const output = typeof result.response?.output === 'string'
        ? JSON.parse(result.response.output)
        : result.response?.output;

      const title = result.vars?.title || 'Unknown';

      if (output?.summary) {
        summaries.push({ title, text: output.summary });
      }
    } catch (e) {
      console.log(`  [Warning] Skipping result with invalid JSON: ${result.vars?.title || 'Unknown'}`);
    }
  }
  return summaries;
}

function analyzeFirstWords(summaries) {
  const issues = [];
  const firstWordCounts = {};

  for (const s of summaries) {
    const firstWord = s.text.trim().split(/\s+/)[0].toLowerCase().replace(/[^a-z]/g, '');
    if (CONFIG.ignoredFirstWords.includes(firstWord)) continue;
    if (!firstWordCounts[firstWord]) firstWordCounts[firstWord] = [];
    firstWordCounts[firstWord].push(s.title);
  }

  const threshold = Math.ceil(summaries.length * CONFIG.maxSameFirstWordPercent / 100);
  
  for (const [word, titles] of Object.entries(firstWordCounts)) {
    const percent = Math.round((titles.length / summaries.length) * 100);
    if (titles.length > threshold) {
      issues.push({
        type: 'same_first_word',
        severity: percent >= 50 ? 'high' : 'medium',
        word,
        count: titles.length,
        percent,
        titles,
      });
    }
  }
  return { issues, firstWordCounts };
}

function analyzeTitleEcho(summaries) {
  const issues = [];
  
  for (const s of summaries) {
    const titleWords = s.title.toLowerCase().split(/\s+/).filter(w => w.length > 3);
    const summaryFirstWord = s.text.trim().split(/\s+/)[0].toLowerCase().replace(/[^a-z]/g, '');
    
    if (titleWords.includes(summaryFirstWord)) {
      issues.push({
        type: 'title_echo',
        title: s.title,
        firstWord: summaryFirstWord,
        summaryPreview: s.text.substring(0, 80) + '...',
      });
    }
  }
  return issues;
}

function analyzeSemicolonPatterns(summaries) {
  const issues = [];
  // Pattern: "Word; explanation" or "Point: explanation" within sentences
  const semicolonRegex = /[A-Z][a-z]+\s*[;:]\s*[a-z]/g;
  
  for (const s of summaries) {
    const matches = s.text.match(semicolonRegex);
    if (matches && matches.length >= 2) {
      issues.push({
        type: 'semicolon_pattern',
        title: s.title,
        matches: matches.slice(0, 3),
        summaryPreview: s.text.substring(0, 100) + '...',
      });
    }
  }
  return issues;
}

function printReport(summaries, firstWordAnalysis, titleEchoIssues, semicolonIssues) {
  console.log('\n' + '='.repeat(70));
  console.log('NOTES CROSS-SAMPLE REPETITION ANALYSIS');
  console.log('='.repeat(70));
  console.log(`Analyzed ${summaries.length} sermon notes\n`);

  // Show first word distribution
  console.log('SUMMARY FIRST WORD DISTRIBUTION:');
  const sorted = Object.entries(firstWordAnalysis.firstWordCounts)
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 10);
  for (const [word, titles] of sorted) {
    const pct = Math.round((titles.length / summaries.length) * 100);
    const bar = '█'.repeat(Math.ceil(pct / 5));
    const status = pct > CONFIG.maxSameFirstWordPercent ? ' ⚠️' : '';
    console.log(`  ${word.padEnd(15)} ${String(titles.length).padStart(2)}/${summaries.length} (${String(pct).padStart(2)}%) ${bar}${status}`);
  }

  // Title echo section
  console.log('\n' + '-'.repeat(70));
  if (titleEchoIssues.length > 0) {
    console.log(`\n⚠️  TITLE ECHO DETECTED (${titleEchoIssues.length} summaries start with title word):\n`);
    for (const issue of titleEchoIssues) {
      console.log(`  ❌ "${issue.title}" → starts with "${issue.firstWord}"`);
      console.log(`     Preview: ${issue.summaryPreview}\n`);
    }
  } else {
    console.log('\n✅ NO TITLE ECHO - Summaries do not start with title words');
  }

  // Semicolon pattern section  
  console.log('\n' + '-'.repeat(70));
  if (semicolonIssues.length > 0) {
    console.log(`\n⚠️  SEMICOLON PATTERNS DETECTED (${semicolonIssues.length} summaries):\n`);
    for (const issue of semicolonIssues) {
      console.log(`  ❌ "${issue.title}"`);
      console.log(`     Patterns: ${issue.matches.join(', ')}`);
    }
  } else {
    console.log('\n✅ NO SEMICOLON PATTERNS - Summaries use complete sentences');
  }

  // Overall result
  const allIssues = [...firstWordAnalysis.issues, ...titleEchoIssues, ...semicolonIssues];
  console.log('\n' + '-'.repeat(70));
  if (allIssues.length === 0) {
    console.log('✅ ALL CHECKS PASSED - No repetition issues detected!');
  } else {
    console.log(`⚠️  FOUND ${allIssues.length} TOTAL ISSUES`);
  }
  console.log('='.repeat(70) + '\n');
  
  return allIssues.length === 0;
}

async function extensionHook(hookName, context) {
  if (hookName === 'afterAll') {
    const summaries = extractSummaries(context.results);
    
    if (summaries.length < 5) {
      console.log('\n[Notes Repetition Analysis] Skipped - need at least 5 summaries');
      return;
    }

    const firstWordAnalysis = analyzeFirstWords(summaries);
    const titleEchoIssues = analyzeTitleEcho(summaries);
    const semicolonIssues = analyzeSemicolonPatterns(summaries);
    printReport(summaries, firstWordAnalysis, titleEchoIssues, semicolonIssues);
  }
}

module.exports = { extensionHook };

