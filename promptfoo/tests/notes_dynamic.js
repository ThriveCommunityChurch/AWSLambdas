/**
 * Dynamic test generator for sermon notes evaluation.
 * Fetches random sermons from MongoDB with cached transcripts.
 * 
 * Usage: Set NOTES_EVAL_LIMIT env var to change count (default: 10)
 */
const { MongoClient } = require('mongodb');
const fs = require('fs');
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '../../scripts/.env') });

const MONGODB_URI = process.env.MONGODB_URI;
const LIMIT = parseInt(process.env.NOTES_EVAL_LIMIT || '10', 10);
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
    
    // Fetch random messages that have summaries (indicating they've been processed)
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
        transcript: cached.transcript
      });
    }

    return results;
  } finally {
    await client.close();
  }
}

module.exports = async function() {
  const sermons = await fetchRandomSermons();
  
  console.log(`\n[notes_dynamic] Loaded ${sermons.length} sermons from MongoDB:`);
  sermons.forEach(s => console.log(`  - ${s.title} (${s.speaker})`));
  console.log('\n');

  return sermons.map(s => ({
    description: `Notes - ${s.title} (${s.speaker}, ID: ${s.messageId})`,
    vars: {
      title: s.title,
      speaker: s.speaker,
      date: s.date,
      transcript: s.transcript
    },
    assert: [
      // Summary word count (2-3 paragraphs, ~250 words)
      {
        type: 'javascript',
        value: `
          const json = typeof output === 'string' ? JSON.parse(output) : output;
          const summary = json.summary || '';
          const wordCount = summary.split(/\\s+/).filter(w => w.length > 0).length;
          if (wordCount < 150 || wordCount > 350) {
            return { pass: false, reason: 'Summary word count ' + wordCount + ' (expected 150-350)' };
          }
          return true;
        `
      },
      // Required arrays have minimum items
      {
        type: 'javascript',
        value: `
          const json = typeof output === 'string' ? JSON.parse(output) : output;
          if (!json.keyPoints || json.keyPoints.length < 2) return { pass: false, reason: 'keyPoints needs at least 2 items' };
          if (!json.applicationPoints || json.applicationPoints.length < 1) return { pass: false, reason: 'applicationPoints needs at least 1 item' };
          return true;
        `
      }
    ]
  }));
};

