/**
 * Dynamic test generator for sermon summary evaluation.
 * Fetches 10 random sermons from MongoDB and compares with cached summaries.
 * 
 * Usage: Set SERMON_EVAL_LIMIT env var to change count (default: 10)
 */
const { MongoClient } = require('mongodb');
const fs = require('fs');
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '../../scripts/.env') });

const MONGODB_URI = process.env.MONGODB_URI;
const LIMIT = parseInt(process.env.SERMON_EVAL_LIMIT || '10', 10);
const CACHE_DIR = path.join(__dirname, '../../scripts/.transcript_cache');

function loadCachedTranscript(messageId) {
  const cachePath = path.join(CACHE_DIR, `${messageId}.json`);
  if (fs.existsSync(cachePath)) {
    try {
      return JSON.parse(fs.readFileSync(cachePath, 'utf8'));
    } catch (e) {
      return null;
    }
  }
  return null;
}

async function fetchRandomSermons() {
  if (!MONGODB_URI) {
    throw new Error('MONGODB_URI not set in scripts/.env');
  }

  const client = new MongoClient(MONGODB_URI);
  
  try {
    await client.connect();
    const db = client.db('SermonSeries');
    
    // Fetch random messages that have summaries
    const messages = await db.collection('Messages')
      .aggregate([
        { $match: { Summary: { $exists: true, $ne: null, $ne: '' } } },
        { $sample: { size: LIMIT * 2 } } // Get extra in case some don't have cached transcripts
      ])
      .toArray();

    const results = [];
    
    for (const msg of messages) {
      if (results.length >= LIMIT) break;
      
      const messageId = msg._id.toString();
      const cached = loadCachedTranscript(messageId);
      
      if (!cached || !cached.transcript) continue;
      
      const date = new Date(msg.Date).toLocaleDateString('en-US', { 
        year: 'numeric', month: 'long', day: 'numeric' 
      });

      results.push({
        messageId,
        title: msg.Title,
        speaker: msg.Speaker,
        date,
        transcript: cached.transcript,
        existingSummary: msg.Summary,
        cachedSummary: cached.notes?.summary || null
      });
    }

    return results;
  } finally {
    await client.close();
  }
}

module.exports = async function() {
  const sermons = await fetchRandomSermons();
  
  console.log(`\n[sermon_summary_dynamic] Loaded ${sermons.length} sermons from MongoDB:`);
  sermons.forEach(s => console.log(`  - ${s.title} (${s.speaker})`));
  console.log('\n--- EXISTING SUMMARIES (for comparison) ---');
  sermons.forEach(s => {
    console.log(`\n[${s.title}]`);
    console.log(`  PROD: ${(s.existingSummary || '').substring(0, 100)}...`);
  });
  console.log('\n');

  return sermons.map(s => ({
    description: `Sermon Summary - ${s.title} (${s.speaker}, ID: ${s.messageId})`,
    vars: {
      title: s.title,
      speaker: s.speaker,
      date: s.date,
      transcript: s.transcript
    },
    metadata: {
      existingSummary: s.existingSummary,
      cachedSummary: s.cachedSummary
    },
    assert: [
      // Word count check (max 120 words)
      {
        type: 'javascript',
        value: `
          const json = typeof output === 'string' ? JSON.parse(output) : output;
          const summary = json.summary || '';
          const wordCount = summary.split(/\\s+/).filter(w => w.length > 0).length;
          if (wordCount > 130) {
            return { pass: false, reason: 'Word count ' + wordCount + ' exceeds max 120' };
          }
          return true;
        `
      },
      // No colon/label pattern - avoid "Topic: rest of sentence" or "Point; subpoint" outline style
      {
        type: 'javascript',
        value: `
          const json = typeof output === 'string' ? JSON.parse(output) : output;
          const summary = (json.summary || '').trim();
          // Check for colon pattern in first ~50 chars (e.g., "Love puffs up: blah" or "Knowledge: blah")
          const openingChunk = summary.substring(0, 50);
          // Pattern: short phrase (1-5 words) followed by colon, then more text (outline style)
          // Semicolons are fine - they connect complete sentences properly
          const colonPattern = /^[A-Z][^.!?]{0,40}:\\s+[a-z]/;
          if (colonPattern.test(summary)) {
            return false;
          }
          return true;
        `
      },
      // No reflective phrases
      {
        type: 'javascript',
        value: `
          const json = typeof output === 'string' ? JSON.parse(output) : output;
          const summary = (json.summary || '').toLowerCase();
          const reflective = ['is presented as', 'is identified as', 'is confessed as', 'is framed as', 'the core insight', 'the core teaching'];
          const found = reflective.filter(p => summary.includes(p));
          if (found.length > 0) {
            return { pass: false, reason: 'Reflective phrase: ' + found.join(', ') };
          }
          return true;
        `
      },
      // Don't start with 'We'
      {
        type: 'javascript',
        value: `
          const json = typeof output === 'string' ? JSON.parse(output) : output;
          const summary = (json.summary || '').trim();
          // Don't start with "We" - lead with the insight/truth, not the community
          return !summary.toLowerCase().startsWith('we ');
        `
      }
    ]
  }));
};

