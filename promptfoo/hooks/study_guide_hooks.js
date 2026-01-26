/**
 * Extension hooks for study guide evaluation.
 * Detects repetitive patterns in summary and devotional fields across all outputs.
 * 
 * Usage: Add to config YAML:
 *   extensions:
 *     - file://hooks/study_guide_hooks.js:extensionHook
 */

const CONFIG = {
  // Max percentage of outputs that can start with same word
  maxSameFirstWordPercent: 25,
  // Max allowed outputs with same 3-word opening
  maxSameThreeWordOpening: 2,
  // Words to ignore when checking first word (articles, prepositions)
  ignoredFirstWords: ['a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'with'],
};

function extractFields(results) {
  const summaries = [];
  const devotionals = [];

  for (const result of results || []) {
    try {
      const output = typeof result.response?.output === 'string'
        ? JSON.parse(result.response.output)
        : result.response?.output;

      const title = result.vars?.title || 'Unknown';

      if (output?.summary) {
        summaries.push({ title, text: output.summary });
      }
      if (output?.devotional) {
        devotionals.push({ title, text: output.devotional });
      }
    } catch (e) {
      // Skip results with invalid JSON
      console.log(`  [Warning] Skipping result with invalid JSON: ${result.vars?.title || 'Unknown'}`);
    }
  }
  return { summaries, devotionals };
}

function analyzeFirstWords(items, fieldName) {
  const issues = [];
  const firstWordCounts = {};

  for (const item of items) {
    const firstWord = item.text.trim().split(/\s+/)[0].toLowerCase().replace(/[^a-z]/g, '');
    if (CONFIG.ignoredFirstWords.includes(firstWord)) continue;
    if (!firstWordCounts[firstWord]) firstWordCounts[firstWord] = [];
    firstWordCounts[firstWord].push(item.title);
  }

  const threshold = Math.ceil(items.length * CONFIG.maxSameFirstWordPercent / 100);
  
  for (const [word, titles] of Object.entries(firstWordCounts)) {
    const percent = Math.round((titles.length / items.length) * 100);
    if (titles.length > threshold) {
      issues.push({
        type: 'same_first_word',
        field: fieldName,
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

function analyzeThreeWordOpenings(items, fieldName) {
  const issues = [];
  const patternCounts = {};

  for (const item of items) {
    const words = item.text.trim().split(/\s+/).slice(0, 3).join(' ').toLowerCase();
    if (!patternCounts[words]) patternCounts[words] = [];
    patternCounts[words].push(item.title);
  }

  for (const [pattern, titles] of Object.entries(patternCounts)) {
    if (titles.length > CONFIG.maxSameThreeWordOpening) {
      issues.push({
        type: 'same_three_word_opening',
        field: fieldName,
        severity: 'high',
        pattern,
        count: titles.length,
        titles,
      });
    }
  }
  return issues;
}

function analyzeTitleEcho(items, fieldName) {
  const issues = [];

  for (const item of items) {
    // Get words from title that are meaningful (longer than 3 chars)
    const titleWords = item.title.toLowerCase().split(/\s+/).filter(w => w.length > 3);
    const textFirstWord = item.text.trim().split(/\s+/)[0].toLowerCase().replace(/[^a-z]/g, '');

    if (titleWords.includes(textFirstWord)) {
      issues.push({
        type: 'title_echo',
        field: fieldName,
        title: item.title,
        firstWord: textFirstWord,
        textPreview: item.text.substring(0, 80) + '...',
      });
    }
  }
  return issues;
}

/**
 * Analyzes whether the summary and devotional start with the same word
 * This is the key issue: they shouldn't echo each other
 */
function analyzeSummaryDevotionalEcho(results) {
  const issues = [];

  for (const result of results || []) {
    try {
      const output = typeof result.response?.output === 'string'
        ? JSON.parse(result.response.output)
        : result.response?.output;

      const title = result.vars?.title || 'Unknown';

      if (output?.summary && output?.devotional) {
        const summaryFirstWord = output.summary.trim().split(/\s+/)[0].toLowerCase().replace(/[^a-z]/g, '');
        const devFirstWord = output.devotional.trim().split(/\s+/)[0].toLowerCase().replace(/[^a-z]/g, '');

        // Check if both start with the same word (ignore short common words)
        if (summaryFirstWord === devFirstWord && summaryFirstWord.length > 2) {
          issues.push({
            type: 'summary_devotional_echo',
            title: title,
            firstWord: summaryFirstWord,
            summaryPreview: output.summary.substring(0, 60) + '...',
            devotionalPreview: output.devotional.substring(0, 60) + '...',
          });
        }
      }
    } catch (e) {
      // Skip results with invalid JSON
    }
  }
  return issues;
}

function printFieldReport(items, fieldName, firstWordAnalysis, threeWordIssues) {
  const allIssues = [...firstWordAnalysis.issues, ...threeWordIssues];
  
  console.log(`\n${fieldName.toUpperCase()} FIRST WORD DISTRIBUTION:`);
  const sorted = Object.entries(firstWordAnalysis.firstWordCounts)
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 8);
  for (const [word, titles] of sorted) {
    const pct = Math.round((titles.length / items.length) * 100);
    const bar = '█'.repeat(Math.ceil(pct / 5));
    const status = pct > CONFIG.maxSameFirstWordPercent ? ' ⚠️' : '';
    console.log(`  ${word.padEnd(15)} ${String(titles.length).padStart(2)}/${items.length} (${String(pct).padStart(2)}%) ${bar}${status}`);
  }

  return allIssues;
}

async function extensionHook(hookName, context) {
  if (hookName === 'afterAll') {
    const { summaries, devotionals } = extractFields(context.results);

    if (summaries.length < 5) {
      console.log('\n[Study Guide Repetition Analysis] Skipped - need at least 5 outputs');
      return;
    }

    console.log('\n' + '='.repeat(70));
    console.log('STUDY GUIDE CROSS-SAMPLE REPETITION ANALYSIS');
    console.log('='.repeat(70));
    console.log(`Analyzed ${summaries.length} study guides\n`);

    // Analyze summaries
    const summaryFirstWord = analyzeFirstWords(summaries, 'summary');
    const summaryThreeWord = analyzeThreeWordOpenings(summaries, 'summary');
    const summaryIssues = printFieldReport(summaries, 'summary', summaryFirstWord, summaryThreeWord);

    // Analyze devotionals
    const devFirstWord = analyzeFirstWords(devotionals, 'devotional');
    const devThreeWord = analyzeThreeWordOpenings(devotionals, 'devotional');
    const devIssues = printFieldReport(devotionals, 'devotional', devFirstWord, devThreeWord);

    // Analyze summary-devotional echo (MOST IMPORTANT CHECK)
    const summaryDevEcho = analyzeSummaryDevotionalEcho(context.results);

    console.log('\n' + '-'.repeat(70));
    if (summaryDevEcho.length > 0) {
      const echoRate = Math.round((summaryDevEcho.length / summaries.length) * 100);
      console.log(`\n⚠️  SUMMARY-DEVOTIONAL ECHO (${summaryDevEcho.length}/${summaries.length} = ${echoRate}% start with same word):\n`);
      for (const issue of summaryDevEcho) {
        console.log(`  ❌ "${issue.title}" → both start with "${issue.firstWord}"`);
        console.log(`     Summary:    ${issue.summaryPreview}`);
        console.log(`     Devotional: ${issue.devotionalPreview}\n`);
      }
    } else {
      console.log('\n✅ NO SUMMARY-DEVOTIONAL ECHO - Summary and devotional have distinct openings');
    }

    // Title echo (secondary check)
    const summaryTitleEcho = analyzeTitleEcho(summaries, 'summary');
    const devTitleEcho = analyzeTitleEcho(devotionals, 'devotional');
    const allTitleEcho = [...summaryTitleEcho, ...devTitleEcho];

    console.log('\n' + '-'.repeat(70));
    if (allTitleEcho.length > 0) {
      console.log(`\n⚠️  TITLE ECHO DETECTED (${allTitleEcho.length} fields start with title word):\n`);
      for (const issue of allTitleEcho) {
        console.log(`  ❌ [${issue.field}] "${issue.title}" → starts with "${issue.firstWord}"`);
        console.log(`     Preview: ${issue.textPreview}\n`);
      }
    } else {
      console.log('\n✅ NO TITLE ECHO - Fields do not start with title words');
    }

    const allIssues = [...summaryIssues, ...devIssues];
    console.log('\n' + '-'.repeat(70));

    const totalIssues = allIssues.length + allTitleEcho.length + summaryDevEcho.length;
    if (totalIssues === 0) {
      console.log('✅ ALL CHECKS PASSED - Opening word variety is good!');
    } else {
      const high = allIssues.filter(i => i.severity === 'high');
      console.log(`⚠️  FOUND ${totalIssues} TOTAL ISSUES:`);
      console.log(`   - ${summaryDevEcho.length} summary-devotional echo (MOST IMPORTANT)`);
      console.log(`   - ${allTitleEcho.length} title echo`);
      console.log(`   - ${high.length} high severity repetition\n`);
      for (const issue of allIssues) {
        const icon = issue.severity === 'high' ? '❌' : '⚠️';
        console.log(`${icon} [${issue.field}] "${issue.word || issue.pattern}" - ${issue.count} occurrences (${issue.percent || ''}%)`);
      }
    }
    console.log('='.repeat(70) + '\n');
  }
}

module.exports = { extensionHook };

