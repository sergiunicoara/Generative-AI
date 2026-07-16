const { VERSES_1SAMUEL_MASTER } = require("./js/verses-1samuel-master.js");
console.log("Total verses in master:", VERSES_1SAMUEL_MASTER.length);
let errors = [];
for (const v of VERSES_1SAMUEL_MASTER) {
  const blankCount = (v.text.match(/\{0\}/g) || []).length;
  if (blankCount !== 1) errors.push(v.ref + ": expected 1 {0}, found " + blankCount);
  const opens = (v.text.match(/\{/g) || []).length;
  const closes = (v.text.match(/\}/g) || []).length;
  if (opens !== 1 || closes !== 1) errors.push(v.ref + ": brace imbalance, open=" + opens + " close=" + closes);
  if (v.blanks.length !== 1) errors.push(v.ref + ": expected 1 blank obj");
  const b = v.blanks[0];
  if (new Set(b.options).size !== 4) errors.push(v.ref + ": options not 4 unique");
  if (!b.options.includes(b.answer)) errors.push(v.ref + ": answer not in options");
  const textWithoutBlank = v.text.replace("{0}", "");
  const escaped = b.answer.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const letter = "A-Za-zĂÂÎȘȚăâîșț";
  const re = new RegExp("(?<![" + letter + "])" + escaped + "(?![" + letter + "])", "i");
  if (re.test(textWithoutBlank)) errors.push(v.ref + ': answer "' + b.answer + '" repeats literally in text!');
}
console.log("Errors:", errors.length ? errors : "none");
