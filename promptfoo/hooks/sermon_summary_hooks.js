/**
 * Extension hooks for sermon summary evaluation.
 * Detects repetitive patterns across all summaries in afterAll.
 * 
 * Usage: Add to config YAML:
 *   extensions:
 *     - file://hooks/sermon_summary_hooks.js:extensionHook
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
    const output = typeof result.response?.output === 'string' 
      ? JSON.parse(result.response.output) 
      : result.response?.output;
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

function analyzeThreeWordOpenings(summaries) {
  const issues = [];
  const patternCounts = {};

  for (const s of summaries) {
    const words = s.summary.trim().split(/\s+/).slice(0, 3).join(' ').toLowerCase();
    if (!patternCounts[words]) patternCounts[words] = [];
    patternCounts[words].push(s.title);
  }

  for (const [pattern, titles] of Object.entries(patternCounts)) {
    if (titles.length > CONFIG.maxSameThreeWordOpening) {
      issues.push({
        type: 'same_three_word_opening',
        severity: 'high',
        pattern,
        count: titles.length,
        titles,
      });
    }
  }
  return issues;
}

function printReport(summaries, firstWordAnalysis, threeWordIssues) {
  const allIssues = [...firstWordAnalysis.issues, ...threeWordIssues];
  
  console.log('\n' + '='.repeat(70));
  console.log('CROSS-SAMPLE REPETITION ANALYSIS');
  console.log('='.repeat(70));
  console.log(`Analyzed ${summaries.length} sermon summaries\n`);

  // Show first word distribution
  console.log('FIRST WORD DISTRIBUTION:');
  const sorted = Object.entries(firstWordAnalysis.firstWordCounts)
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 10);
  for (const [word, titles] of sorted) {
    const pct = Math.round((titles.length / summaries.length) * 100);
    const bar = '█'.repeat(Math.ceil(pct / 5));
    const status = pct > CONFIG.maxSameFirstWordPercent ? ' ⚠️' : '';
    console.log(`  ${word.padEnd(15)} ${String(titles.length).padStart(2)}/${summaries.length} (${String(pct).padStart(2)}%) ${bar}${status}`);
  }

  console.log('\n' + '-'.repeat(70));

  if (allIssues.length === 0) {
    console.log('✅ NO REPETITION ISSUES - Opening word variety is good!');
    console.log('='.repeat(70) + '\n');
    return true;
  }

  const high = allIssues.filter(i => i.severity === 'high');
  const medium = allIssues.filter(i => i.severity === 'medium');

  console.log(`⚠️  FOUND ${allIssues.length} REPETITION ISSUES (${high.length} high, ${medium.length} medium)\n`);

  for (const issue of allIssues) {
    const icon = issue.severity === 'high' ? '❌' : '⚠️';
    if (issue.type === 'same_first_word') {
      console.log(`${icon} "${issue.word}" starts ${issue.count}/${summaries.length} summaries (${issue.percent}%)`);
    } else {
      console.log(`${icon} "${issue.pattern}" appears in ${issue.count} summaries`);
    }
    console.log(`   Sermons: ${issue.titles.slice(0, 5).join(', ')}${issue.titles.length > 5 ? '...' : ''}`);
  }

  console.log('\n' + '='.repeat(70) + '\n');
  return high.length === 0;
}

async function extensionHook(hookName, context) {
  if (hookName === 'afterAll') {
    const summaries = extractSummaries(context.results);
    
    if (summaries.length < 5) {
      console.log('\n[Repetition Analysis] Skipped - need at least 5 summaries');
      return;
    }

    const firstWordAnalysis = analyzeFirstWords(summaries);
    const threeWordIssues = analyzeThreeWordOpenings(summaries);
    const passed = printReport(summaries, firstWordAnalysis, threeWordIssues);

    if (!passed && process.env.CI === 'true') {
      process.exitCode = 1;
    }
  }
}

module.exports = { extensionHook };

