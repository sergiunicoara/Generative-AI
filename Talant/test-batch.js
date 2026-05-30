'use strict';
const Anthropic = require('@anthropic-ai/sdk');
const client = new Anthropic();

const batch = [
  {globalIdx: 0, question: 'Cine a zis: „Doamne, Tu să-mi speli mie picioarele?"', correctAnswer: 'Simon Petru'},
  {globalIdx: 6, question: 'Cum se numește localitatea cu multe ape, în care boteza Ioan?', correctAnswer: 'Enon'},
  {globalIdx: 9, question: 'Isus le-a zis: „Umpleți _ acestea cu apă."', correctAnswer: 'Vasele'},
];

const prompt = `You are building a Bible quiz about the Gospel of John (Ioan), written in Romanian.

For each question below, I need exactly 3 WRONG answer choices (distractors).
The distractors MUST be:
  - The same type/category as the correct answer
    * person name correct  -> give 3 other person names from the Gospel of John
    * place name correct   -> give 3 other place names from the Gospel of John
    * short phrase/word    -> give 3 similarly short phrases in the same grammatical form
    * Bible verse ref      -> give 3 other verse references from Ioan (e.g. "Ioan 3:5")
  - Written in Romanian
  - Plausible-sounding (a student might be fooled)
  - Different from the correct answer and from each other

Return ONLY a valid JSON array -- no markdown, no explanation.
Each element: {"idx": <number>, "distractors": ["wrong1","wrong2","wrong3"]}

Questions:
${batch.map(b => JSON.stringify({idx:b.globalIdx, question:b.question, correct:b.correctAnswer})).join('\n')}`;

(async () => {
  const msg = await client.messages.create({
    model: 'claude-haiku-4-5',
    max_tokens: 512,
    messages: [{role:'user', content: prompt}],
  });
  console.log(msg.content[0].text);
})().catch(console.error);
