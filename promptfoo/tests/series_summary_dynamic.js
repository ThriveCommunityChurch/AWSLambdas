/**
 * Dynamic test generator for series summary evaluation.
 * Fetches real series data from MongoDB at runtime.
 */
const { MongoClient, ObjectId } = require('mongodb');
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '../../scripts/.env') });

const MONGODB_URI = process.env.MONGODB_URI;
const LIMIT = parseInt(process.env.SERIES_EVAL_LIMIT || '3', 10);

async function fetchSeriesWithMessages() {
  if (!MONGODB_URI) {
    throw new Error('MONGODB_URI not set in scripts/.env');
  }

  const client = new MongoClient(MONGODB_URI);
  
  try {
    await client.connect();
    const db = client.db('SermonSeries');
    
    // Fetch random completed series (those with EndDate set) using $sample
    const series = await db.collection('Series')
      .aggregate([
        { $match: { EndDate: { $ne: null } } },
        { $sample: { size: LIMIT } }
      ])
      .toArray();

    const results = [];
    
    for (const s of series) {
      // Fetch messages for this series
      const messages = await db.collection('Messages')
        .find({ SeriesId: s._id })
        .sort({ Date: 1 })
        .project({ Title: 1, Date: 1, PassageRef: 1, Summary: 1 })
        .toArray();

      if (messages.length === 0) continue;

      // Format messages for the prompt
      const messageSummaries = messages.map((m, idx) => {
        const date = new Date(m.Date).toLocaleDateString('en-US', { 
          year: 'numeric', month: 'long', day: 'numeric' 
        });
        return `${idx + 1}. "${m.Title}"
   Date: ${date}
   Passage: ${m.PassageRef || 'N/A'}
   Summary: ${m.Summary || 'No summary available'}`;
      }).join('\n\n');

      results.push({
        seriesName: s.Name,
        seriesId: s._id.toString(),
        messageCount: messages.length,
        messageSummaries
      });
    }

    return results;
  } finally {
    await client.close();
  }
}

module.exports = async function() {
  const seriesData = await fetchSeriesWithMessages();
  
  console.log(`[series_summary_dynamic] Loaded ${seriesData.length} series from MongoDB:`);
  seriesData.forEach(s => console.log(`  - ${s.seriesName} (${s.messageCount} messages)`));

  return seriesData.map(s => ({
    description: `Series Summary - ${s.seriesName} (${s.messageCount} messages, ID: ${s.seriesId})`,
    vars: {
      seriesName: s.seriesName,
      messageSummaries: s.messageSummaries
    },
    assert: [
      {
        type: 'javascript',
        value: `
          const json = typeof output === 'string' ? JSON.parse(output) : output;
          const summary = json.summary || '';
          const wordCount = summary.split(/\\s+/).filter(w => w.length > 0).length;
          if (wordCount < 35 || wordCount > 80) {
            return { pass: false, reason: 'Word count ' + wordCount + ' (expected 35-80)' };
          }
          return true;
        `
      },
      {
        type: 'javascript',
        value: `
          const json = typeof output === 'string' ? JSON.parse(output) : output;
          const summary = json.summary || '';
          return !summary.includes('\\n');
        `
      },
      {
        type: 'javascript',
        value: `
          const json = typeof output === 'string' ? JSON.parse(output) : output;
          const summary = (json.summary || '').toLowerCase().trim();
          const bannedStarts = ['we ', 'in this series', 'this series', 'across', 'through '];
          const found = bannedStarts.find(b => summary.startsWith(b));
          if (found) return { pass: false, reason: 'Starts with banned phrase: ' + found };
          return true;
        `
      },
      {
        type: 'javascript',
        value: `
          const json = typeof output === 'string' ? JSON.parse(output) : output;
          const summary = (json.summary || '').toLowerCase();
          // Use word boundaries for single words to avoid false positives (e.g., "reframes" matching "frames")
          const forbiddenPhrases = ["you'll discover", "journey through", "by the end", "invites you to", "you'll learn", "you will", "your life"];
          const forbiddenWords = ["emerges", "emerging", "thread", "weaves", "woven", "surfaces", "frames"];
          const foundPhrases = forbiddenPhrases.filter(p => summary.includes(p));
          const foundWords = forbiddenWords.filter(w => new RegExp('\\\\b' + w + '\\\\b').test(summary));
          const found = [...foundPhrases, ...foundWords];
          if (found.length > 0) return { pass: false, reason: 'Forbidden words/phrases: ' + found.join(', ') };
          return true;
        `
      },
      {
        type: 'javascript',
        value: `
          // Check for abstract colon-list structures (poetic/vague items after colon)
          const json = typeof output === 'string' ? JSON.parse(output) : output;
          const summary = json.summary || '';
          // Ban: "topic: abstract, abstract, abstract" patterns
          // Allow: "traced the claims: Father, Jesus, resurrection" (concrete items)
          const abstractPatterns = [
            /reframed as[^:]*:/i,
            /thread[^:]*:/i,
            /emerges[^:]*:/i,
            /surfaces[^:]*:/i
          ];
          for (const pattern of abstractPatterns) {
            if (pattern.test(summary)) {
              return { pass: false, reason: 'Contains abstract colon-list structure' };
            }
          }
          return true;
        `
      },
      {
        type: 'javascript',
        value: `
          const json = typeof output === 'string' ? JSON.parse(output) : output;
          const summary = (json.summary || '').toLowerCase();
          const weCount = (summary.match(/\\bwe\\b/g) || []).length;
          const usCount = (summary.match(/\\bus\\b/g) || []).length;
          const ourCount = (summary.match(/\\bour\\b/g) || []).length;
          const total = weCount + usCount + ourCount;
          if (total > 3) return { pass: false, reason: 'Too many we/us/our: ' + total + ' (max 3)' };
          return true;
        `
      }
    ]
  }));
};

