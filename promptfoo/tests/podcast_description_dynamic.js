/**
 * Dynamic test generator for podcast description evaluation.
 * Fetches PodcastEpisodes from MongoDB and loads transcripts from local cache.
 */
const { MongoClient } = require('mongodb');
const path = require('path');
const fs = require('fs');
require('dotenv').config({ path: path.join(__dirname, '../../scripts/.env') });

const MONGODB_URI = process.env.MONGODB_URI;
const LIMIT = parseInt(process.env.PODCAST_EVAL_LIMIT || '5', 10);
const CACHE_DIR = path.join(__dirname, '../../scripts/.transcript_cache');

/**
 * Get transcript from local cache by messageId.
 */
function getTranscriptFromCache(messageId) {
  const blobName = `${messageId}.json`;
  const cachePath = path.join(CACHE_DIR, blobName);
  
  if (!fs.existsSync(cachePath)) {
    return null;
  }
  
  try {
    const content = fs.readFileSync(cachePath, 'utf8');
    const doc = JSON.parse(content);
    return doc.transcript || null;
  } catch (err) {
    console.error(`Error reading cache for ${messageId}: ${err.message}`);
    return null;
  }
}

async function fetchPodcastEpisodes() {
  if (!MONGODB_URI) {
    throw new Error('MONGODB_URI not set in scripts/.env');
  }

  const client = new MongoClient(MONGODB_URI);
  
  try {
    await client.connect();
    const db = client.db('SermonSeries');
    
    // Fetch random podcast episodes using $sample
    const episodes = await db.collection('PodcastEpisodes')
      .aggregate([
        { $sample: { size: LIMIT * 2 } } // Fetch extra in case some don't have transcripts
      ])
      .toArray();

    const results = [];
    
    for (const ep of episodes) {
      if (results.length >= LIMIT) break;
      
      const messageId = ep.messageId;
      if (!messageId) continue;
      
      // Get transcript from cache
      const transcript = getTranscriptFromCache(messageId);
      if (!transcript || transcript.length < 500) continue;
      
      results.push({
        messageId,
        title: ep.title || 'Unknown',
        speaker: ep.speaker || 'Unknown',
        podcastTitle: ep.podcastTitle || ep.title || 'Unknown',
        oldDescription: ep.description || '',
        transcript
      });
    }

    return results;
  } finally {
    await client.close();
  }
}

module.exports = async function() {
  const episodes = await fetchPodcastEpisodes();
  
  console.log(`[podcast_description_dynamic] Loaded ${episodes.length} episodes from MongoDB:`);
  episodes.forEach(ep => console.log(`  - ${ep.title} (${ep.speaker})`));

  return episodes.map(ep => ({
    description: `Podcast Description - ${ep.title} by ${ep.speaker} (ID: ${ep.messageId})`,
    vars: {
      title: ep.title,
      speaker: ep.speaker,
      transcript: ep.transcript
    },
    assert: [
      // Word count: 80-250 words (two paragraphs, 130-180 target)
      {
        type: 'javascript',
        value: `
          const desc = output.trim();
          const wordCount = desc.split(/\\s+/).filter(w => w.length > 0).length;
          if (wordCount < 80 || wordCount > 250) {
            return { pass: false, reason: 'Word count ' + wordCount + ' (expected 80-250)' };
          }
          return true;
        `
      },
      // Should have two paragraphs (blank line separator)
      {
        type: 'javascript',
        value: `
          const desc = output.trim();
          const paragraphs = desc.split(/\\n\\n+/).filter(p => p.trim().length > 0);
          if (paragraphs.length !== 2) {
            return { pass: false, reason: 'Expected 2 paragraphs, got ' + paragraphs.length };
          }
          return true;
        `
      },
      // No speaker name
      {
        type: 'javascript',
        value: `
          const desc = output.toLowerCase();
          const speaker = '${ep.speaker}'.toLowerCase();
          if (desc.includes(speaker)) {
            return { pass: false, reason: 'Contains speaker name: ${ep.speaker}' };
          }
          // Check last name only
          const nameParts = '${ep.speaker}'.split(' ');
          if (nameParts.length > 1) {
            const lastName = nameParts[nameParts.length - 1].toLowerCase();
            if (lastName.length > 2 && desc.includes(lastName)) {
              return { pass: false, reason: 'Contains speaker last name: ' + nameParts[nameParts.length - 1] };
            }
          }
          return true;
        `
      },
      // Banned opening patterns
      {
        type: 'javascript',
        value: `
          const desc = output.toLowerCase().trim();
          const bannedStarts = [
            'what do you do when', 'have you ever', 'imagine', 'in a world',
            'life often feels', 'that awkward moment', 'you notice', "you're juggling"
          ];
          const found = bannedStarts.find(b => desc.startsWith(b));
          if (found) return { pass: false, reason: 'Starts with banned phrase: ' + found };
          return true;
        `
      },
      // No question as first sentence
      {
        type: 'javascript',
        value: `
          const desc = output.trim();
          const firstSentence = desc.split(/[.!?]/)[0];
          if (firstSentence && firstSentence.trim().endsWith('?')) {
            return { pass: false, reason: 'First sentence is a question' };
          }
          // Also check if starts with question word pattern
          const questionStarts = ['who ', 'what ', 'when ', 'where ', 'why ', 'how ', 'is ', 'are ', 'do ', 'does ', 'can ', 'could ', 'would ', 'should '];
          const lower = desc.toLowerCase();
          const startsWithQuestion = questionStarts.some(q => lower.startsWith(q));
          if (startsWithQuestion && desc.includes('?')) {
            return { pass: false, reason: 'Starts with a question' };
          }
          return true;
        `
      },
      // Forbidden phrases
      {
        type: 'javascript',
        value: `
          const desc = output.toLowerCase();
          const forbidden = [
            "in this episode", "join us", "this week", "this message explores",
            "this sermon invites", "you'll discover", "you'll learn", "you'll find",
            "you'll be challenged", "transformative", "transformation", "profound",
            "profoundly", "how will you respond", "what will you do", "are you ready",
            "how might your life change"
          ];
          const found = forbidden.filter(p => desc.includes(p));
          if (found.length > 0) {
            return { pass: false, reason: 'Contains forbidden phrase: ' + found.join(', ') };
          }
          return true;
        `
      },
      // No rhetorical closing question
      {
        type: 'javascript',
        value: `
          const desc = output.trim();
          const lastSentence = desc.split(/[.!]/).pop().trim();
          if (lastSentence.endsWith('?')) {
            return { pass: false, reason: 'Ends with a rhetorical question' };
          }
          return true;
        `
      }
    ]
  }));
};

