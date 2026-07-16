// Generează js/verses-1samuel.js și js/verses-2samuel.js din fișierele master
// (niciodată distribuite ca atare). Publică TOT conținutul disponibil — nu mai
// există fereastră săptămânală (aplicația are acum acord legal pentru textul
// integral). Rulează manual după orice modificare a fișierelor master:
//   node scripts/build-window.js

const fs = require("fs");
const path = require("path");

function loadMaster(fileName, exportName) {
  const filePath = path.join(__dirname, "..", "js", fileName);
  if (!fs.existsSync(filePath)) return [];
  delete require.cache[require.resolve(filePath)];
  return require(filePath)[exportName] || [];
}

function serializeVerse(v) {
  const blanksStr = v.blanks
    .map(
      (b) =>
        `{ answer: ${JSON.stringify(b.answer)}, options: ${JSON.stringify(b.options)} }`
    )
    .join(", ");
  return `  {\n    ref: ${JSON.stringify(v.ref)},\n    text: ${JSON.stringify(
    v.text
  )},\n    blanks: [${blanksStr}],\n  },`;
}

function writeFile(bookLabel, varName, outFileName, verses) {
  const header = `// GENERAT AUTOMAT din fișierul master de scripts/build-window.js — nu edita direct.\n// Conținut complet (${bookLabel}).\n\n`;
  const body = `const ${varName} = [\n${verses.map(serializeVerse).join("\n")}\n];\n`;
  fs.writeFileSync(path.join(__dirname, "..", "js", outFileName), header + body, "utf8");
  return verses.length;
}

function main() {
  const samuel1 = loadMaster("verses-1samuel-master.js", "VERSES_1SAMUEL_MASTER");
  const samuel2 = loadMaster("verses-2samuel-master.js", "VERSES_2SAMUEL_MASTER");

  const n1 = writeFile("1 Samuel", "VERSES_1SAMUEL", "verses-1samuel.js", samuel1);
  const n2 = writeFile("2 Samuel", "VERSES_2SAMUEL", "verses-2samuel.js", samuel2);

  console.log(`Publicat integral: 1 Samuel=${n1} versete, 2 Samuel=${n2} versete`);
}

if (require.main === module) main();

module.exports = { loadMaster };
