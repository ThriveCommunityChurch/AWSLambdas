/**
 * Analyze devotionals from production cache
 */

const fs = require('fs');
const path = require('path');

const CACHE_DIR = path.join(__dirname, '..', 'scripts', '.transcript_cache');

const files = fs.readdirSync(CACHE_DIR).filter(f => f.endsWith('.json'));
const devotionals = [];

for (const file of files) {
  try {
    const data = JSON.parse(fs.readFileSync(path.join(CACHE_DIR, file), 'utf8'));
    const guide = data.studyGuide;
    if (guide && guide.devotional) {
      const dev = guide.devotional;
      const paragraphs = dev.split(/\n\n/).filter(p => p.trim());
      const wordCount = dev.split(/\s+/).length;
      const firstSentence = dev.split(/[.!?]/)[0].trim();
      devotionals.push({
        title: data.title,
        paragraphs: paragraphs.length,
        wordCount,
        firstSentence: firstSentence.substring(0, 120),
        startsWithYou: firstSentence.toLowerCase().startsWith('you'),
        hasReadSlowly: dev.toLowerCase().includes('read') && dev.toLowerCase().includes('slowly'),
        hasOpenYourBible: dev.toLowerCase().includes('open your bible'),
        hasInvitedTo: dev.toLowerCase().includes('invited to'),
        hasThisMessage: dev.toLowerCase().includes('this message'),
        hasThisSermon: dev.toLowerCase().includes('this sermon'),
      });
    }
  } catch (e) {}
}

console.log('DEVOTIONAL ANALYSIS');
console.log('='.repeat(70));
console.log('Total devotionals found:', devotionals.length);
console.log('');

// First sentence patterns
console.log('FIRST SENTENCE PATTERNS:');
console.log('-'.repeat(70));
const firstWordCounts = {};
devotionals.forEach(d => {
  const firstWord = d.firstSentence.split(' ')[0].toLowerCase();
  firstWordCounts[firstWord] = (firstWordCounts[firstWord] || 0) + 1;
});
const sortedFirstWords = Object.entries(firstWordCounts).sort((a, b) => b[1] - a[1]).slice(0, 10);
sortedFirstWords.forEach(([word, count]) => {
  const pct = ((count / devotionals.length) * 100).toFixed(1);
  console.log(`  "${word}": ${count} times (${pct}%)`);
});

console.log('');
console.log('PROBLEMATIC PATTERNS:');
console.log('-'.repeat(70));
console.log(`  Starts with "you": ${devotionals.filter(d => d.startsWithYou).length} (${((devotionals.filter(d => d.startsWithYou).length / devotionals.length) * 100).toFixed(1)}%)`);
console.log(`  Has "read slowly": ${devotionals.filter(d => d.hasReadSlowly).length} (${((devotionals.filter(d => d.hasReadSlowly).length / devotionals.length) * 100).toFixed(1)}%)`);
console.log(`  Has "open your bible": ${devotionals.filter(d => d.hasOpenYourBible).length}`);
console.log(`  Has "invited to": ${devotionals.filter(d => d.hasInvitedTo).length}`);
console.log(`  Has "this message": ${devotionals.filter(d => d.hasThisMessage).length}`);
console.log(`  Has "this sermon": ${devotionals.filter(d => d.hasThisSermon).length}`);

console.log('');
console.log('LENGTH STATISTICS:');
console.log('-'.repeat(70));
const avgParagraphs = (devotionals.reduce((a, d) => a + d.paragraphs, 0) / devotionals.length).toFixed(1);
const avgWordCount = Math.round(devotionals.reduce((a, d) => a + d.wordCount, 0) / devotionals.length);
const minWords = Math.min(...devotionals.map(d => d.wordCount));
const maxWords = Math.max(...devotionals.map(d => d.wordCount));
console.log(`  Avg paragraphs: ${avgParagraphs}`);
console.log(`  Avg word count: ${avgWordCount}`);
console.log(`  Min word count: ${minWords}`);
console.log(`  Max word count: ${maxWords}`);

console.log('');
console.log('SAMPLE FIRST SENTENCES (first 15):');
console.log('-'.repeat(70));
devotionals.slice(0, 15).forEach(d => {
  console.log(`[${d.title}]:`);
  console.log(`  "${d.firstSentence}..."`);
  console.log(`  (${d.paragraphs} paragraphs, ${d.wordCount} words)`);
  console.log('');
});

