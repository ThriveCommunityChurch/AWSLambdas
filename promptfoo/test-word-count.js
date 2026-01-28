// Test what word count logic looks like when eval'd from a string (like promptfoo does)

const summary = "Courage looks less like swagger and more like an 80-year-old kneeling in prayer. The Daniel story flips our expectations: faithful living does not guarantee fair treatment, and moral courage is rarer than physical bravado.";

// Test 1: Direct regex
console.log("=== Test 1: Direct regex ===");
console.log("Word count:", summary.split(/\s+/).filter(w => w.length > 0).length);

// Test 2: Simulate what promptfoo does - eval a string with \\s+ in it
console.log("\n=== Test 2: Eval with \\\\s+ ===");
const codeDoubleBackslash = `
  const summary = "${summary}";
  const wordCount = summary.split(/\\s+/).filter(w => w.length > 0).length;
  wordCount;
`;
console.log("Code string:", codeDoubleBackslash);
console.log("Result:", eval(codeDoubleBackslash));

// Test 3: What if we use a literal space pattern instead
console.log("\n=== Test 3: Split on space ===");
console.log("Word count:", summary.split(' ').filter(w => w.length > 0).length);

// Test 4: Use [ ]+ pattern (space in character class)
console.log("\n=== Test 4: Split on [ ]+ ===");
console.log("Word count:", summary.split(/[ ]+/).filter(w => w.length > 0).length);

