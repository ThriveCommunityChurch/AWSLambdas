/**
 * Compare Production vs Eval Results
 * 
 * Extracts summary openings from production cache and compares with eval results
 * to identify improvements in repetitive patterns.
 */

const fs = require('fs');
const path = require('path');

const CACHE_DIR = path.join(__dirname, '..', 'scripts', '.transcript_cache');
const EVAL_RESULTS = path.join(__dirname, 'results_repetition.json');

// Phrases we're trying to eliminate
const PROBLEMATIC_PHRASES = [
  'you are invited to',
  'you are invited into',
  'right now, you',
  'step into',
  'join us as',
  'this sermon explores',
  'this message examines',
  'the passage reveals',
];

function getOpeningWords(text, count = 8) {
  if (!text) return '';
  return text.trim().split(/\s+/).slice(0, count).join(' ').toLowerCase();
}

function analyzeProduction() {
  const files = fs.readdirSync(CACHE_DIR).filter(f => f.endsWith('.json'));
  const summaries = [];
  const phraseCount = {};
  
  PROBLEMATIC_PHRASES.forEach(p => phraseCount[p] = 0);
  
  for (const file of files) {
    try {
      const data = JSON.parse(fs.readFileSync(path.join(CACHE_DIR, file), 'utf8'));
      // Check both 'notes' (production) and 'sermonNotes' (older format)
      const notes = data.notes || data.sermonNotes;
      if (notes && notes.summary) {
        const summary = notes.summary;
        summaries.push({
          title: data.title,
          opening: getOpeningWords(summary),
          summary: summary.substring(0, 200),
        });
        
        // Count problematic phrases
        const lowerSummary = summary.toLowerCase();
        PROBLEMATIC_PHRASES.forEach(phrase => {
          if (lowerSummary.includes(phrase)) {
            phraseCount[phrase]++;
          }
        });
      }
    } catch (e) {
      // Skip invalid files
    }
  }
  
  return { summaries, phraseCount, total: summaries.length };
}

function analyzeEval() {
  if (!fs.existsSync(EVAL_RESULTS)) {
    console.error('Eval results not found:', EVAL_RESULTS);
    return null;
  }
  
  const data = JSON.parse(fs.readFileSync(EVAL_RESULTS, 'utf8'));
  const summaries = [];
  const phraseCount = {};
  
  PROBLEMATIC_PHRASES.forEach(p => phraseCount[p] = 0);
  
  for (const result of data.results?.results || []) {
    const output = result.response?.output;
    if (output && output.summary) {
      const summary = output.summary;
      const title = result.vars?.title || 'Unknown';
      summaries.push({
        title,
        opening: getOpeningWords(summary),
        summary: summary.substring(0, 200),
      });
      
      const lowerSummary = summary.toLowerCase();
      PROBLEMATIC_PHRASES.forEach(phrase => {
        if (lowerSummary.includes(phrase)) {
          phraseCount[phrase]++;
        }
      });
    }
  }
  
  return { summaries, phraseCount, total: summaries.length };
}

// Main
const production = analyzeProduction();
const evalResults = analyzeEval();

console.log('\n' + '='.repeat(70));
console.log('PRODUCTION vs EVAL COMPARISON');
console.log('='.repeat(70));

console.log(`\nProduction cache: ${production.total} sermons with summaries`);
console.log(`Eval results: ${evalResults?.total || 0} sermons\n`);

console.log('PROBLEMATIC PHRASE OCCURRENCES:');
console.log('-'.repeat(50));
console.log('Phrase'.padEnd(30) + 'Production'.padEnd(12) + 'Eval');
console.log('-'.repeat(50));

PROBLEMATIC_PHRASES.forEach(phrase => {
  const prodCount = production.phraseCount[phrase];
  const evalCount = evalResults?.phraseCount[phrase] || 0;
  const status = evalCount < prodCount ? '✅' : (evalCount === 0 ? '✅' : '⚠️');
  console.log(`${phrase.padEnd(30)}${String(prodCount).padEnd(12)}${evalCount} ${status}`);
});

console.log('\n' + '='.repeat(70));
console.log('SAMPLE PRODUCTION OPENINGS (first 15):');
console.log('-'.repeat(70));
production.summaries.slice(0, 15).forEach(s => {
  console.log(`[${s.title}]: "${s.opening}..."`);
});

console.log('\n' + '='.repeat(70));
console.log('EVAL OPENINGS:');
console.log('-'.repeat(70));
evalResults?.summaries.forEach(s => {
  console.log(`[${s.title}]: "${s.opening}..."`);
});

// Analyze opening word patterns in production
console.log('\n' + '='.repeat(70));
console.log('PRODUCTION OPENING WORD ANALYSIS:');
console.log('-'.repeat(70));
const firstWords = {};
production.summaries.forEach(s => {
  const first = s.opening.split(' ')[0];
  firstWords[first] = (firstWords[first] || 0) + 1;
});
const sortedFirstWords = Object.entries(firstWords)
  .sort((a, b) => b[1] - a[1])
  .slice(0, 10);
console.log('Most common first words:');
sortedFirstWords.forEach(([word, count]) => {
  const pct = ((count / production.total) * 100).toFixed(1);
  console.log(`  "${word}": ${count} times (${pct}%)`);
});

// Check for "you" starting pattern
const youCount = production.summaries.filter(s => s.opening.startsWith('you ')).length;
const youPct = ((youCount / production.total) * 100).toFixed(1);
console.log(`\n⚠️  ${youCount} summaries (${youPct}%) start with "you" - indicates templated pattern`);

console.log('\n' + '='.repeat(70));
console.log('CONCLUSION:');
console.log('-'.repeat(70));
const totalProdIssues = Object.values(production.phraseCount).reduce((a, b) => a + b, 0);
const totalEvalIssues = Object.values(evalResults?.phraseCount || {}).reduce((a, b) => a + b, 0);
console.log(`Production: ${totalProdIssues} total problematic phrase occurrences across ${production.total} sermons`);
console.log(`Eval: ${totalEvalIssues} total problematic phrase occurrences across ${evalResults?.total || 0} sermons`);
if (totalEvalIssues < totalProdIssues) {
  console.log('\n✅ IMPROVEMENT: New prompts significantly reduce repetitive patterns!');
} else {
  console.log('\n⚠️  No significant improvement detected');
}
console.log('='.repeat(70));

