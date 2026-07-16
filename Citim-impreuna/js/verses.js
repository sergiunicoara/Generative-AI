// Combină fereastra săptămânii curente (generată de scripts/build-window.js
// din fișierele master, niciodată distribuite) în lista folosită de joc.
const VERSES = [...VERSES_1SAMUEL, ...(typeof VERSES_2SAMUEL !== "undefined" ? VERSES_2SAMUEL : [])];
