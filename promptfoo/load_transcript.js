const fs = require('fs');
const path = require('path');

module.exports = function(varName, prompt, otherVars, provider) {
  // Get the transcript file path from otherVars
  const transcriptFile = otherVars.transcript_file;
  if (!transcriptFile) {
    return { error: 'No transcript_file variable provided' };
  }
  
  // Resolve the path relative to this file's directory
  const filePath = path.join(__dirname, transcriptFile);
  
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    return { output: content };
  } catch (err) {
    return { error: `Failed to read file ${filePath}: ${err.message}` };
  }
};

