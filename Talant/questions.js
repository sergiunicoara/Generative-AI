// ════════════════════════════════════════════════════════════
// QUESTIONS — Cartea lui Ioan
// Generated from fata1..10 / verso1..10.
// Rules:
//   Q3 → only the a/b/c options shown on the card (multi-select allowed)
//   Q6 → always Adevărat / Fals only (2 choices)
//   Q1,Q2,Q4,Q5 → distractors sampled from same-word-count correct answers
// ════════════════════════════════════════════════════════════

const QUESTIONS = [
  {
    "card": "Cartea lui Ioan · Card 1.01",
    "question": "Cine a zis: „Doamne, Tu să-mi speli mie picioarele?”",
    "answers": [
      "Ioan",
      "Simon Petru",
      "Iacov",
      "Andrei"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Simon Petru (Ioan 13:6)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.01",
    "question": "Cine și cui a zis: „Ce fac Eu, tu nu pricepi acum, dar vei pricepe după aceea\"?",
    "answers": [
      "Domnul Isus, lui Simon Petru",
      "Domnul Isus, lui Ioan Botezătorul",
      "Domnul Isus, lui Iuda Iscariotul",
      "Domnul Isus, lui Natanael"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Domnul Isus, lui Simon Petru."
  },
  {
    "card": "Cartea lui Ioan · Card 1.01",
    "question": "Cum Îl numeau ucenicii, pe Domnul Isus:",
    "answers": [
      "Domnul",
      "Împăratul lui Israel",
      "Învătătorul"
    ],
    "correct": [
      0,
      2
    ],
    "feedback": "Răspunsul corect este a, c (Ioan 13:13)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.01",
    "question": "Dacă Eu, Domnul și Învățătorul vostru, v-am spălat picioarele, și voi sunteți datori să vă spălați picioarele __.",
    "answers": [
      "Familiei voastre",
      "Fraților voștri",
      "Unii altora",
      "Copiilor voștri"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Unii altora (Ioan 13:114)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.01",
    "question": "„Le-a orbit ochii și le-a împietrit inima, ca să nu vadă și, să nu înțeleagă, să nu se întoarcă la Dumnezeu și să-i vindec\"- a profețit?",
    "answers": [
      "Ieremia",
      "Isaia",
      "Ezechiel",
      "Maleahi"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Isaia (Ioan 12:40)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.01",
    "question": "Aveți credință în Dumnezeu și aveți credință în Mine.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 14:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.02",
    "question": "Cum se numește localitatea cu multe ape, în care boteza Ioan?",
    "answers": [
      "Betania",
      "Capernaum",
      "Sicar",
      "Enon"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Enon (Ioan 3:23)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.02",
    "question": "Cine l-a zis lui Isus: ”știm că ești un Învățător venit de ta Dumnezeu, căci nimeni nu poate face semnele pe care le faci Tu\"?",
    "answers": [
      "Iacov din Zebedei",
      "Matei vameșul",
      "Nicodim",
      "Filimon"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Nicodim (Ioan 3:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.02",
    "question": "„Omul nu poate primi decât ce-i este dat din cer\", a zis:",
    "answers": [
      "Ioan",
      "Isus",
      "Petru"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 3:27)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.02",
    "question": "Isus le-a zis: „Umpleți _ acestea cu apă.”",
    "answers": [
      "Vasele",
      "Coșurile",
      "Căzile",
      "Borcanele"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Vasele (Ioan 2:7)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.02",
    "question": "Cui a zis Isus: „Ridicați acestea de aici și nu faceți din casa Tatălui Meu o casă de negustorie.”",
    "answers": [
      "Schimbătorilor de bani",
      "Cumpărătorilor de pești",
      "Făuritorilor de unelte",
      "Vânzătorilor de porumbei"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Vânzătorilor de porumbei (Ioan 2:16)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.02",
    "question": "Fiul slujbașului împărătesc vindecat de Isus era bolnav de lepră.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă."
  },
  {
    "card": "Cartea lui Ioan · Card 1.03",
    "question": "Cum mai este numită marea Galileii?",
    "answers": [
      "Marea Caspică",
      "Marea Tiberiadei",
      "Marea Moartă",
      "Marea Egee"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Marea Tiberiadei (Ioan 6:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.03",
    "question": "Cum se numea, în evreiește scăldătoarea de la Poarta Oilor, din Ierusalim?",
    "answers": [
      "Betesda",
      "Siloam",
      "Gihon",
      "Cedron"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Betesda (Ioan 5:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.03",
    "question": "Au zis lui Isus: „Pleacă de aici și du-Te în Iudeea, ca să vadă și ucenicii Tăi lucrările pe care le faci\":",
    "answers": [
      "fratii lui Isus",
      "fariseii",
      "niște preoți"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a."
  },
  {
    "card": "Cartea lui Ioan · Card 1.03",
    "question": "Pe voi lumea nu vă poate urî; pe Mine Mă urăște, pentru că mărturisesc despre ea că lucrările ei sunt _.",
    "answers": [
      "Bune",
      "Vrednice",
      "Rele",
      "Sfinte"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Rele."
  },
  {
    "card": "Cartea lui Ioan · Card 1.03",
    "question": "Scriptura zice că Hristosul are să vină din sămânța lui... și din satul?",
    "answers": [
      "Salomon, Nazaretul",
      "David, Betleem",
      "Iesse, Ierusalimul",
      "Iuda, Galadul"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: David, Betleem (Ioan 7:42)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.03",
    "question": "Mama lui Isus nu a fost la nuna din Cana Galileei",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă."
  },
  {
    "card": "Cartea lui Ioan · Card 1.04",
    "question": "Despre cine spunea Domnul Isus că, el de ta început a fost ucigaș?",
    "answers": [
      "Iuda Iscariotul",
      "Pilat",
      "Caiafa",
      "Diavol"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Diavol (Ioan 8:44)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.04",
    "question": "Pe ale cui cuvinte le ascultă cel ce este din Dumnezeu?",
    "answers": [
      "Satana",
      "Lumea",
      "Dumnezeu",
      "Părinții lui"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Dumnezeu (Ioan 8:47)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.04",
    "question": "Spuneau că Isus, nu vine de la Dumnezeu fiindcă nu ține Sabatul:",
    "answers": [
      "unii din farisei",
      "ucenicii Lui",
      "frații Lui"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 9:16)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.04",
    "question": "Cine vorbește de la sine caută __ dar cine caută slava Celui ce L-a trimis, Acela este adevărat și în El nu este strâmbătate.",
    "answers": [
      "Slava lui însuși",
      "Glorie trecătoare",
      "Onoarea oamenilor",
      "Mărturie mincinoasă"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Slava lui însuși (Ioan 7:16)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.04",
    "question": "Cine a zis: „noi am crezut și am ajuns la cunoștința că Tu ești Hristosul, Sfântul lui Dumnezeu\"?",
    "answers": [
      "Ioan Botezătorul",
      "Iacov, fratele lui Iacov",
      "Matei",
      "Simon Petru"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Simon Petru (Ioan 6:68)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.04",
    "question": "Cel ce nu crede în Domnul Isus va muri în păcatul lui.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 6:24)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.05",
    "question": "Cine este numit de Isus, Mângâietorul pe care-L va trimite Tatăl?",
    "answers": [
      "Cuvântul",
      "Duhul Sfânt",
      "Îngerul Domnului",
      "Fiul Omului"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Duhul Sfânt (Ioan 14:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.05",
    "question": "Ce personaj este modelul de iubire pe care trebuie să-t avem noi?",
    "answers": [
      "Domnul Isus",
      "Moise",
      "Avraam",
      "Apostolul Ioan"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 13:34)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.05",
    "question": "Nimeni nu vine la Tatăl decât prin:",
    "answers": [
      "rugăciune",
      "Domnul Isus",
      "botez"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este b (Ioan 14:6)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.05",
    "question": "În casa Tatălui Meu sunt multe _. Dacă n-ar fi așa, v-aș fi spus.",
    "answers": [
      "Judecăți",
      "Minuni",
      "Locașuri",
      "Taine"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Locașuri (Ioan 114:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.05",
    "question": "Cum putem dovedi că Îl iubim pe Domnul Isus, potrivit așteptărilor Lui?",
    "answers": [
      "Mărturisind despre El",
      "Păzind poruncile Lui",
      "Dând milostenie",
      "Postind și rugând"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Păzind poruncile Lui."
  },
  {
    "card": "Cartea lui Ioan · Card 1.05",
    "question": "Duhul Sfânt trimis de Tatăl, vă va învăța toate lucrurile.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 14:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.06",
    "question": "Unde se afla Isus, când Maria l-a uns picioarele cu mir?",
    "answers": [
      "La Ierusalim",
      "La Cana din Galileea",
      "La Iacob",
      "În Betania"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Law În Betania (Ioan 12:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.06",
    "question": "Cine și cui a zis: „Las-o în pace, căci ea l-a păstrat pentru ziua îngropării Mele\"?",
    "answers": [
      "Isus lui Petru",
      "Maria lui Iuda Iscarioteanu",
      "Isus lui Iuda Iscarioteanu",
      "Isus lui Natanael"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Isus lui Iuda Iscarioteanu (Ioan 12:14)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.06",
    "question": "După minunea învierii tui Lazăr, iudeii martori:",
    "answers": [
      "toți au crezut în Isus",
      "mulți au crezut în Isus",
      "unii au mers să spună fariseilor"
    ],
    "correct": [
      1,
      2
    ],
    "feedback": "Răspunsul corect este b, c (Ioan 11:45)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.06",
    "question": "Maria a luat un litru cu mir de nard curat, de _, a uns picioarele lui Isus și l-a șters picioarele cu părul ei.",
    "answers": [
      "Mare preț",
      "Bună calitate",
      "Miros puternic",
      "Gust fin"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Mare preț (Ioan 12:3)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.06",
    "question": "Cine a zis despre Sine însuși: ”am venit ca să fiu o lumină În lume\"?",
    "answers": [
      "Tatăl cerului",
      "Fiul omului",
      "Împăratul Iudeilor",
      "Domnul Isus"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 12:46)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.06",
    "question": "S-a dezbrăcat de hainele Lui, a luat un ștergar și S-a încins cu el.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 13:4)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.07",
    "question": "Peste puțină vreme, nu Mă veți mai vedea. Unde a zis Isus că merge?",
    "answers": [
      "La cer",
      "La Tatăl",
      "La Betania",
      "La Ghersetani"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: La Tatăl (Ioan 16:16)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.07",
    "question": "Dacă știți aceste lucruri și le faceți, va fi... de voi?",
    "answers": [
      "Ferice",
      "Binecuvântat",
      "Mântuitor",
      "Vrednic"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Ferice (Ioan 13:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.07",
    "question": "Isus S-a rugat pentru:",
    "answers": [
      "ucenicii Săi",
      "pentru toată lumea",
      "pentru cei ce vor crede în El"
    ],
    "correct": [
      0,
      2
    ],
    "feedback": "Răspunsul corect este a, c (Ioan 17:)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.07",
    "question": "Când va veni Mângâietorul, _, are să vă călăuzească în tot adevărul.",
    "answers": [
      "Duhul Sfânt",
      "Mângâietorul",
      "Duhul adevărului",
      "Duhul de adevăr"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Duhul adevărului (Ioan 16:)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.07",
    "question": "Cum l-a numit Isus, în rugăciunea Sa, pe Iuda care a pierit ca să se împlinească Scriptura?",
    "answers": [
      "Păcătosul lumii",
      "Fiul pierzării",
      "Vrăjitorul",
      "Ispititorul"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Fiul pierzării (Ioan 17:12)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.07",
    "question": "Duhul Sfânt vă va învăța toate lucrurile și vă va aduce aminte de tot ce a spus Isus.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 14:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.08",
    "question": "Cine a zis despre Isus: „Dacă Omul acesta n-ar veni de la Dumnezeu, n-ar putea face nimic\"?",
    "answers": [
      "Samarineanul",
      "Lazărul din Betania",
      "Cleopa",
      "Orbul din naștere vindecat de"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Orbul din naștere vindecat de."
  },
  {
    "card": "Cartea lui Ioan · Card 1.08",
    "question": "Cum recunosc oile pe păstorul după care merg?",
    "answers": [
      "Dupa vocea lui",
      "Dupa mireasma sa",
      "Dupa pașii lui",
      "Dupa înfățișarea lui"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Dupa vocea lui (Ioan 10:3-4)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.08",
    "question": "Întrebat de farisei, ce crede el despre Isus, orbul din nastere vindecat a zis:",
    "answers": [
      "nu stiu",
      "nu-L cunosc",
      "este un proroc"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este 3 c (Ioan 9:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.08",
    "question": "Mai am și alte oi, care nu sunt din __ și pe acelea trebuie să le aduc. Ele vor asculta de glasul Me.u, și va fi o turmă și un Păstor.",
    "answers": [
      "Staulul acesta",
      "Templul acesta",
      "Grădinii acesteia",
      "Poarta aceasta"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Staulul acesta (Ioan 10:16)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.08",
    "question": "Cine și cui a zis: „Doar n-ați vrea să vă faceți și voi ucenicii",
    "answers": [
      "Orbul, iudeilor",
      "Nevăzătorul, scriitorilor",
      "Bolnavul, săducheilor",
      "Orbul, fariseilor"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Orbul, fariseilor."
  },
  {
    "card": "Cartea lui Ioan · Card 1.08",
    "question": "Marta a zis lui Isus: știu că orice vei cere de la Dumnezeu, Îți va da”.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 11:22)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.09",
    "question": "Cui i-a zis Isus: ”împărăția Mea nu este din lumea aceasta\"?",
    "answers": [
      "Caiafa",
      "Pilat",
      "Iuda",
      "Petru"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Pilat (Ioan 16:35)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.09",
    "question": "Ce le-a zis Isus, ucenicilor după ce a suflat peste ei?",
    "answers": [
      "Luati Duh Sfântl",
      "Plecați și predicați",
      "Rămâneți în Ierusalim",
      "Mergeți în toate națiile"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Luati Duh Sfântl."
  },
  {
    "card": "Cartea lui Ioan · Card 1.09",
    "question": "Isus a zis: „Dacă ar fi Împărăția Mea din lumea aceasta, s-ar fi luptat ca să nu fiu dat în mâinile iudeilor”",
    "answers": [
      "ucenicii Mei",
      "Tată' Meu",
      "slujitorii Mei"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este 20:22 c (Ioan 18:36)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.09",
    "question": "\"Voi sunteți prietenii Mei, dacă _ ce vă poruncesc Eu.\"",
    "answers": [
      "Ascultați",
      "Păziți",
      "Faceți",
      "Iubiți"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Faceți (Ioan 15:14)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.09",
    "question": "Cine va dovedi lumea vinovată în ce privește păcatul, neprihănirea și judecata?",
    "answers": [
      "Tatăl",
      "Mângăietorul I Duhul Sfânt",
      "Fiul omului",
      "Îngerul Domnului"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Mângăietorul I Duhul Sfânt (Ioan 16:7)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.09",
    "question": "Păzirea poruncilor este o condiție de a rămâne în draaostea lui Isus.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 15:10)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.10",
    "question": "Cine a zis despre Dumnezeu, Tatăl: \"Eu Î' cunosc bine\"? I",
    "answers": [
      "Apele",
      "Marta",
      "Maica lui Isus",
      "Domnul Isus"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 8:55)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.10",
    "question": "În ce situație în toc să răspundă unei întrebări, Isus a început să scrie cu degetul pe pământ?",
    "answers": [
      "Când a fost întrebat despre răstignire",
      "Când a fost întrebat despre viață veșnică",
      "Când a fost întrebat în. legătură cu femeia prinsă în",
      "Când a fost întrebat despre învierea morților"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Când a fost întrebat în. legătură cu femeia prinsă în."
  },
  {
    "card": "Cartea lui Ioan · Card 1.10",
    "question": "Cui a zis Isus: 'Voi judecați după înfățișare; Eu nu judec",
    "answers": [
      "fariseilor",
      "ucenicilor",
      "preoților pe nimeni\"."
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este păcat (Ioan 6:14)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.10",
    "question": "atunci veți cunoaște că nu fac „Când veți înălța pe __ nimic de la Mine.",
    "answers": [
      "Fiul omului",
      "Tatăl Ceresc",
      "Părintele vostru",
      "Stăpânul casei"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Fiul omului (Ioan 8:28)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.10",
    "question": "Cine a întrebat: „Legea noastră osândește ea pe un om înainte ca să-l asculte și să știe ce face\"?",
    "answers": [
      "Zaharia",
      "Ioan Botezătorul",
      "Arimaticianul Iosif",
      "Nicodim"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Nicodim (Ioan 7:50)."
  },
  {
    "card": "Cartea lui Ioan · Card 1.10",
    "question": "„M-am dus, m-am spălat și mi-am căpătat vederea” a zis Bartimeu.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 9:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.01",
    "question": "Despre cine a zis Isus: \"Vă va învăța toate lucrurile\"? I",
    "answers": [
      "Tatăl Ceresc",
      "Duhul Sfânt",
      "Mesajul Evangheliei",
      "Cuvântul lui Dumnezeu"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Duhul Sfânt (Ioan 14:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.01",
    "question": "Ce pildă practică a dat Isus ucenicilor și apoi le-a zis: și voi să faceți cum am făcut Eu?",
    "answers": [
      "Spălarea picioarelor",
      "Întinziunea mânilor",
      "Ungerea capului cu ulei",
      "Vorbirea în limbi"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Spălarea picioarelor."
  },
  {
    "card": "Cartea lui Ioan · Card 2.01",
    "question": "Cine primește pe acela pe care-l trimit Eu pe __ primește și cine Mă primește pe Mine primește pe Cel ce __ pe Mine.",
    "answers": [
      "Mine, m-a trimis",
      "Tatăl, m-a creat",
      "Lumea, m-a urât"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Mine, m-a trimis (Ioan 13:20)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.01",
    "question": "Să vă iubiți unii pe alții; cum _ așa să vă iubiți și voi unii pe alții.",
    "answers": [
      "Isus vi l-a poruncit",
      "Tatăl meu v-a poruncit",
      "V-am iubit Eu",
      "El v-a învățat"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: V-am iubit Eu (Ioan 13:34)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.01",
    "question": "Ce replică i-a dat Isus lui Petru care a zis: „Niciodată nu-mi vei spăla picioarele\"?",
    "answers": [
      "„Tu nu înțelegi acum ce fac, dar vei înțelege mai târziu.'",
      "„Dacă nu te spăl Eu, nu vei avea parte deloc cu Mine.\"",
      "„Cel ce mă primește pe Mine, mă primește pe Cel ce m-a trimis.'",
      "„Dacă nu ți-ai schimbat gândurile, nu poți intra în Împărăția Cerului.'"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: „Dacă nu te spăl Eu, nu vei avea parte deloc cu Mine.\"."
  },
  {
    "card": "Cartea lui Ioan · Card 2.01",
    "question": "Dacă știți aceste lucruri, ferice de voi dacă le faceți.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 13:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.02",
    "question": "Cine a mărturisit despre Domnul Isus, la Sihar și mulți au crezut în El?",
    "answers": [
      "Maria din Magdala",
      "Marta din Betania",
      "Femeia gbegriff legii",
      "Femeia samariteancă"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Femeia samariteancă."
  },
  {
    "card": "Cartea lui Ioan · Card 2.02",
    "question": "În ce localitate a vindecat Isus, pe fiul unui slujbaș împărătesc?",
    "answers": [
      "Betania",
      "Ierihon",
      "Capernaum",
      "Sihar"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Capernaum (Ioan 14:146)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.02",
    "question": "„Dacă nu vedeți semne și minuni, cu niciun chip nu credețil” - a zis:",
    "answers": [
      "Ioan Botezătorul",
      "Isus",
      "Petru"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este b (Ioan 14:48)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.02",
    "question": "Cine seceră primește o plată și strânge rod pentru __, pentru ca și cel ce seamănă și cel ce seceră să se bucure în același timp.",
    "answers": [
      "Viața veșnică",
      "Harul Dumnezeului",
      "Împărăția lui Dumnezeu",
      "Puterea Duhului"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Viața veșnică (Ioan 4:36)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.02",
    "question": "Ca ce S-a referit Isus când a zis: „Eu am de mâncat o mâncare pe care voi n-o cunoașteți\"?",
    "answers": [
      "Sfinții care se-ntorc din morți",
      "Învăința învățătorilor legii",
      "Porunca primului cuvânt",
      "Voia Tatălui"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Voia Tatălui (Ioan 4:32)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.02",
    "question": "Ioan boteza în Nain, acolo erau multe ape.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 3:23)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.03",
    "question": "Ce a spus Domnul Isus că se întâmplă cu mlădița care n-aduce rod?",
    "answers": [
      "Este aruncată în foc",
      "Este tăiată",
      "Este lăsată să se usuce",
      "Este transformată în rod"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Este tăiată (Ioan 15:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.03",
    "question": "Cine a zis: ”stăpânitorul lumii acesteia, n-are nimic în Mine\"?",
    "answers": [
      "Domnul Isus",
      "Petru cel Apostol",
      "Diavolul și oștenii lui",
      "Fiul celui rău"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 14:30)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.03",
    "question": "V-am numit..., pentru că v-am făcut cunoscut tot ce am auzit de la Tatăl Meu.",
    "answers": [
      "prieteni",
      "robi",
      "slujitori"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 15:15)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.03",
    "question": "Când va veni Mângâietorul, adică Duhul adevărului, El va _ despre Mine.",
    "answers": [
      "Va judeca",
      "Va pedepsi",
      "Mărturisi",
      "Va condamna"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Mărturisi (Ioan 15:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.03",
    "question": "Cine a zis: ”În lume veți avea necazuri, dar îndrăzniți, Eu am biruit lumea\"?",
    "answers": [
      "Petru",
      "Domnul Isus",
      "Pavel",
      "Ioan Botezătorul"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 16:33)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.03",
    "question": "Robul este mai mare decât stăpânul său.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 15:20)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.04",
    "question": "La ce eveniment a plâns Domnul Isus?",
    "answers": [
      "La Golgota în momentul crucificării",
      "La Cana când s-a învechit vinul",
      "La Ierusalim când a văzut templul",
      "La Mormântul lui Lazăr din Betania"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: La Mormântul lui Lazăr din Betania (Ioan 11:34)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.04",
    "question": "Cine a zis: 'feste în folosul vostru să moară un singur om pentru norod și să nu piară tot neamul\"?",
    "answers": [
      "Anania",
      "Sfetnicul iudeilor",
      "Caiafa",
      "Pilat"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Caiafa (Ioan 11:)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.04",
    "question": "Când a uns picioarele lui Isus, Maria a folosit... de mir:",
    "answers": [
      "un litru",
      "o sticlută",
      "un vas"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 12:3)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.04",
    "question": "„Nu te teme, fiica Sionului; iată că Împăratul tău vine călare pe __ _",
    "answers": [
      "Mânzul unei măgărițe",
      "Unui cal alb",
      "Unui măgar purtător de poveri",
      "Unui cămilă"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Mânzul unei măgărițe."
  },
  {
    "card": "Cartea lui Ioan · Card 2.04",
    "question": "Cui i-a zis Isus: „Nu ți-am spus că, dacă vei crede, vei vedea slava lui Dumnezeu\"?",
    "answers": [
      "Mariei din Magdala",
      "Femeii samaritence",
      "Mariei, mama lui Isus",
      "Martei"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Martei (Ioan 11:3)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.04",
    "question": "Lângă crucea lui Isus stătea mama și mătusa Lui.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 19:25)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.05",
    "question": "Cine a trimis lui Isus vestea: „Doamne, iată că acela pe care-l iubești este bolna'./'?",
    "answers": [
      "Elisabeta și Elisei, verişoarele lui Isus",
      "Maria și Marta, surorile lui Lazăr",
      "Maria din Magdala și Ioana",
      "Susana și alta Maria"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Maria și Marta, surorile lui Lazăr (Ioan 11:3)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.05",
    "question": "Cine și cui a zis: „Haidem să mergem și noi să murim cu EV?",
    "answers": [
      "Toma, celorlalți ucenici",
      "Petru, celorlalți ucenici",
      "Filipi, celorlalți apostoli",
      "Bartolomeu, celorlalți credincioși"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Toma, celorlalți ucenici (Ioan 11:16)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.05",
    "question": "Betania era:",
    "answers": [
      "cetate",
      "sat",
      "ținut/ regiune Preoții cei mai de seamă și _ porunciseră că, dacă va ști"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 11:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.05",
    "question": "cineva unde este Isus, să le dea de știre ca să-L prindă - la praznicul Paștelor.",
    "answers": [
      "Irozii",
      "Arcieraii",
      "Fariseii",
      "Precizii și Saducheii"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Fariseii (Ioan 11:55)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.05",
    "question": "Cât timp, a mai rămas Isus în locul în care se afla, înainte de a pleca spre Betania unde locuia Lazăr care era grav bolnav?",
    "answers": [
      "Trei zile",
      "Două zile",
      "Patru zile",
      "O zi"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Două zile (Ioan 11:6)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.05",
    "question": "Maria Magdalena a privit de departe răstignirea lui Isus.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 19:25)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.06",
    "question": "Isus a zis: ”Eu sunt adevărata Viță și Tatăl Meu este. \"?",
    "answers": [
      "Porumbarul",
      "Oasele",
      "Sămânța",
      "Vierul"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Vierul (Ioan 15:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.06",
    "question": "Cine când a fost întrebat de Isus, dacă crede în Fiul lui Dumnezeu, a zis: „Cred, Doamne” și I s-a închinat?",
    "answers": [
      "Samarinenii",
      "Nicedem",
      "Orbul din naștere",
      "Femeia samarineană"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Orbul din naștere (Ioan 9:)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.06",
    "question": ", Ferice de cei ce n-au văzut, și au crezut” - a zis:",
    "answers": [
      "Toma",
      "Petru",
      "Isus"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c (Ioan 20:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.06",
    "question": "Isus a suflat peste ucenici și le-a zis: „Luați __ I",
    "answers": [
      "Duh Sfânt",
      "Putere",
      "Cuvântul",
      "Credința"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Duh Sfânt (Ioan 20:22)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.06",
    "question": "Cine și cui a zis: ce le veți ierta păcatele vor fi iertate și celor ce le veți ține vor fi ținute\"?",
    "answers": [
      "Petru, enoriașilor",
      "Ioan, mulțimii",
      "Matei, oamenilor",
      "Isus, ucenicilor"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Isus, ucenicilor."
  },
  {
    "card": "Cartea lui Ioan · Card 2.06",
    "question": "Domnul Isus S-a rugat pentru ucenicii Săi, ca ei să fie una.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 17:11)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.07",
    "question": "Care om a fost ținta planurilor ucigașe ale preoților căci din pricina lui mulți iudei plecau de la ei și credeau în Isus?",
    "answers": [
      "Bartimeu din Iehona",
      "Lazăr din Betania",
      "Fiul cuvanului regal",
      "Omul de la Betesda"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Lazăr din Betania."
  },
  {
    "card": "Cartea lui Ioan · Card 2.07",
    "question": "Cine și cui a zis: „Domnule, am vrea să vedem pe Isus\"?",
    "answers": [
      "Niște greci, lui Filip",
      "Nicodemus, lui Isus",
      "Iuda Iscariotul, apostolilor",
      "Femeia la fântână, unei vecine"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Niște greci, lui Filip."
  },
  {
    "card": "Cartea lui Ioan · Card 2.07",
    "question": "A făcut această declarație lui Isus; \"Eu îmi voi da viata pentru T inel”",
    "answers": [
      "Ioan",
      "Petru",
      "Varianta c"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este 12:20-21 b (Ioan 13:37)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.07",
    "question": "Când le-a zis Isus: „Eu sunt\", ei s-au dat înapoi și au",
    "answers": [
      "Cântat cântări de laudă",
      "Plâns cu glas tare",
      "Căzut la pământ",
      "Deschis ochii lor"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Căzut la pământ (Ioan 18:6)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.07",
    "question": "Toți cei ce fuseseră împreună cu Isus, când chemase pe Lazăr din Mormânt și-l înviase din morți, _ despre El.",
    "answers": [
      "Murmurau",
      "Mărturiseau",
      "Se-ndoiau",
      "Scoteau"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Mărturiseau (Ioan 12:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.07",
    "question": "Isus i-a zis lui Ioan: nu va cânta cocoșul, până te vei lepăda de Mine.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 13:37)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.08",
    "question": "Care este localitatea de baștină a lui Filip, Andrei și Petru?",
    "answers": [
      "Capernaum",
      "Nazaret",
      "Ieriho",
      "Betsaida"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Betsaida (Ioan 1:44)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.08",
    "question": "Despre cine s-a zis: „lată cu adevărat un israelit în care nu este vicleșug\"?",
    "answers": [
      "Matei Levitul",
      "Iuda fratele lui Iacov",
      "Natanael",
      "Toma"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Natanael (Ioan 1:)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.08",
    "question": "În Cana din Galileea, Domnul Isus a fost cu ucenicii Lui la:",
    "answers": [
      "sinagogă",
      "o nuntă",
      "Petru acasă"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 2:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.08",
    "question": "Și acolo erau de piatră, puse după obiceiul de curățare al iudeilor.",
    "answers": [
      "Șase vase",
      "Opt vase",
      "Zece vase",
      "Patru vase"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Șase vase (Ioan 2:6)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.08",
    "question": "Cine i-a spus fratelui său: ”Am găsit pe Mesia și l-a adus la Isus\"?",
    "answers": [
      "Filip",
      "Petru",
      "Ioan",
      "Andrei"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Andrei (Ioan 1:140)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.08",
    "question": "După ce s-a întors de la nuntă, Isus a mers la Capernaum, cu mama, frații și ucenicii Lui.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 2:12)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.09",
    "question": "Cetatea în care se afla Isus, când s-a așezat lângă o fântână fiind ostenit de călătorie, se numea?",
    "answers": [
      "Betania",
      "Sihar",
      "Cana",
      "Ierihon"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Sihar (Ioan 4:5)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.09",
    "question": "Cui i s-a adresat Isus spunând: ”Dă-Mi să beau\"?",
    "answers": [
      "Unei femei samaritence",
      "Unui fariseului",
      "Unui slujitor",
      "Unui soldat roman"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Unei femei samaritence."
  },
  {
    "card": "Cartea lui Ioan · Card 2.09",
    "question": "Femeia samariteancă și-a lăsat..., s-a dus în cetate și a zis",
    "answers": [
      "ulciorul",
      "găleata",
      "vasul oamenilor."
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este b (Ioan 14:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.09",
    "question": "Scăldătoarea, Betesda, de la Poarta Oilor, din Ierusalim avea _ pridvoare.",
    "answers": [
      "Trei",
      "Șase",
      "Cinci",
      "Nouă"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Cinci (Ioan 5:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.09",
    "question": "În cât timp a fost zidit Templul de ta Ierusalim?",
    "answers": [
      "50 de ani",
      "46 de ani",
      "40 de ani",
      "35 de ani"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: 46 de ani (Ioan 2:20)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.09",
    "question": "Cana este o localitate situată în regiunea Iudeea.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 2:11)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.10",
    "question": "În cine trebuie să credem pentru a avea viață veșnică?",
    "answers": [
      "Moise, Profetul",
      "Isus, Fiul lui Dumnezeu",
      "Ioan Botezătorul",
      "Avraam, părintele credinței"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Isus, Fiul lui Dumnezeu (Ioan 3:36)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.10",
    "question": "La cine trebuie să mergi, ca să nu flămânzești și să nu însetezi niciodată?",
    "answers": [
      "La Spiritul Sfânt",
      "La Duhul adevărului",
      "La Domnul Isus",
      "La Cuvântul etern"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: La Domnul Isus (Ioan 6:)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.10",
    "question": "Nimeni nu poate veni la Mine dacă..., care M-a trimis;",
    "answers": [
      "nu-I atrage Tatăl",
      "nu se roagă Celui",
      "Varianta c"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 6:)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.10",
    "question": "Dar cine lucrează după adevăr vine la lumină, pentru ca să i se arate faptele, fiindcă sunt făcute în _",
    "answers": [
      "Dumnezeu",
      "Adevăr",
      "Lumină",
      "Iubire"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Dumnezeu (Ioan 3:21)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.10",
    "question": "Voi vă închinați la ce nu cunoașteți, noi ne închinăm [a ce cunoaștem, căci Mântuirea vine de la?",
    "answers": [
      "Samariteni",
      "Romani",
      "Egipt",
      "Iudei"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Iudei (Ioan 4:22)."
  },
  {
    "card": "Cartea lui Ioan · Card 2.10",
    "question": "Mânia lui Dumnezeu rămâne peste cel ce nu crede în Fiul.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 3:36)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.01",
    "question": "Cine este Ușa oilor, potrivit spuselor domnului Isus?",
    "answers": [
      "El este Păstorul",
      "El este Usa",
      "El este Lumina lumii",
      "El este Vița adevărată"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: El este Usa (Ioan 10:7)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.01",
    "question": "Cine a zis: Eu Îmi cunosc oile Mele, și ele Mă cunosc pe",
    "answers": [
      "Domnul Isus",
      "Ioan Botezătorul",
      "Petru",
      "Filip"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 10:114)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.01",
    "question": "După vindecarea orbului din naștere, iudeii hotărâseră că, dacă va mărturisi cineva că Isus este Hristosul, să fie:",
    "answers": [
      "dat afară din sinagogă",
      "ucis",
      "arestat"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 9:22)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.01",
    "question": "Cine nu intră pe ușă în _, ci sare pe altă parte, este un hoț și un tâlhar.",
    "answers": [
      "Templul",
      "Piața din Ierusalim",
      "Staulul oilor",
      "Grădina"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Staulul oilor (Ioan 10:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.01",
    "question": "Ce a răspuns Isus ta întrebarea: „cine a păcătuit: omul acesta sau părinții lui, de s-a născut orb?”",
    "answers": [
      "Părinții lui au păcătuit",
      "Nici el, nici părinții lui, s-a născut așa, ca să se arate în el lucrările tui",
      "El a păcătuit înainte să se nască",
      "Trebuie să fie izgonit din sinagogă"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Nici el, nici părinții lui, s-a născut așa, ca să se arate în el lucrările tui."
  },
  {
    "card": "Cartea lui Ioan · Card 3.01",
    "question": "Grăuntele de grâu dacă nu moare, rămâne singur.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 9:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.02",
    "question": "Cine și cui a zis: „Du-te de te spală în scăldătoarea Sitoamului”?",
    "answers": [
      "Petru, unui bolnav",
      "Filip, unei femei samaritence",
      "Andrei, unui cuvios",
      "Domnul Isus, omului născut orb"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Domnul Isus, omului născut orb."
  },
  {
    "card": "Cartea lui Ioan · Card 3.02",
    "question": "Cine erau cei ce spuneau despre ei: 'tnoi suntem ucenicii lui Moise\"?",
    "answers": [
      "Saducheii",
      "Martorii lui Iehova",
      "Fariseii",
      "Ucznicii lui Ioan"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Fariseii."
  },
  {
    "card": "Cartea lui Ioan · Card 3.02",
    "question": "Când au văzut minunea făcută de Isus, au zis: „Cu adevărat, Acesta este Prorocul ce' așteptat în lume\":",
    "answers": [
      "oamenii",
      "ucenicii",
      "fariseii"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 6:14)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.02",
    "question": "Oile Mele ascultă glasul _ Eu le cunosc și ele vin după Mine.",
    "answers": [
      "Meu",
      "Tău",
      "Lui Dumnezeu",
      "Păstorului"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Meu (Ioan 10:27)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.02",
    "question": "Isus te-a zis oamenilor: ”Mă căutați nu pentru că ați văzut semne, ci pentru că...'l?",
    "answers": [
      "Ați văzut zece semne",
      "Ați crezut în Mine din inimă",
      "Ați fost cu Mine la Betania",
      "Ați mâncat din pâinile acelea"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Ați mâncat din pâinile acelea."
  },
  {
    "card": "Cartea lui Ioan · Card 3.02",
    "question": "Păstorul plătit fuge, când vine lupul, lui nu-i pasă de oi.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 6:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.03",
    "question": "nu sunt vrednic să-l• dezleg cureaua Cine la zis: încălțămintei Lui\"?",
    "answers": [
      "Petru",
      "Ioan",
      "Matei",
      "Natan"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Ioan (Ioan 1:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.03",
    "question": "Cine L-a întrebat pe Isus: ”învățătorule unde locuiești\"?",
    "answers": [
      "Doi ucenici de-ai tui Ioan",
      "Nicodim și Iosif din Arimatea",
      "Marta și Maria",
      "O femeie samariteancă și servitorul ei"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Doi ucenici de-ai tui Ioan."
  },
  {
    "card": "Cartea lui Ioan · Card 3.03",
    "question": "Când a fost interogat de către trimișii fariseilor, despre cine este el, Ioan se afla în:",
    "answers": [
      "Betabara",
      "Ierusalim",
      "Nazaret"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este Botezătorul (Ioan 1:37)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.03",
    "question": "CNatanael i-a zis: „Poate ieși ceva bun din Nazaret?” „Vino și vezil” i-a răspuns _.",
    "answers": [
      "Andrei",
      "Ioan",
      "Filip",
      "Iacov"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Filip (Ioan 1:46)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.03",
    "question": "Care ucenic al Domnului Isus, din cei 12, a fost mai întâi ucenicul lui Ioan Botezătorul?",
    "answers": [
      "Petru",
      "Andrei",
      "Ioan",
      "Filip"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Andrei (Ioan 1:37)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.03",
    "question": "Chifa În tâlmăcire/ traducere înseamnă Petru.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 1:42)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.04",
    "question": "Cine a zis: ”Eu am biruit lumea\"?",
    "answers": [
      "Satana",
      "Lumea",
      "Părintele",
      "Domnul Isus"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 16:33)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.04",
    "question": "Isus L-a rugat pe Dumnezeu Tatăl: Nu Te rog să-i iei din lume, ci..?",
    "answers": [
      "Să-i sfinți în adevăr",
      "Să-i umple cu putere",
      "Să-i păzești de cel rău",
      "Să-i pregătească locul"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Să-i păzești de cel rău."
  },
  {
    "card": "Cartea lui Ioan · Card 3.04",
    "question": "Isus le-a spus ucenicilor despre prigonirile prin care vor",
    "answers": [
      "ele să nu fie pentru ei un prilej de cădere trece ca:",
      "să nu se sperie",
      "să-și aducă aminte că El le-a spus"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este Ioan 17: 15 a, c (Ioan 16:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.04",
    "question": "'Eu Te-am proslăvit pe pământ, am sfârșit _ pe care Mi-ai dat-o s-o fac” - a spus Isus în rugăciunea Sal",
    "answers": [
      "Lucrarea",
      "Porunca",
      "Fapta",
      "Voia"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Lucrarea (Ioan 17:4)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.04",
    "question": "Potrivit rugăciunii lui Isus - Ce este viața veșnică?",
    "answers": [
      "Să păzească poruncile lui Dumnezeu și ale lui Isus",
      "Să trăiască în chiesa și în dreapta",
      "Să iubească pe aproapele lor ca pe ei înșiși",
      "Să-l cunoști pe singurul Dumnezeu adevărat și pe Isus Hristos pe care L-a"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Să-l cunoști pe singurul Dumnezeu adevărat și pe Isus Hristos pe care L-a."
  },
  {
    "card": "Cartea lui Ioan · Card 3.04",
    "question": "Le-am dat Cuvântul Tău, și lumea i-a urât, pentru că ei nu sunt din lume.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 17:3)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.05",
    "question": "Cum se numea Marea unde Isus S-a arătat ucenicilor Săi, după învierea din morți?",
    "answers": [
      "Galileei",
      "Tiberiadei",
      "Morții",
      "Genezaretului"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Tiberiadei (Ioan 21:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.05",
    "question": "Cum se numește orașul de baștină, al lui Natanael?",
    "answers": [
      "Cana Galileea",
      "Betania",
      "Ierihon",
      "Naim"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Cana Galileea (Ioan 21:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.05",
    "question": "Cine și cui a zis că a văzut pe Domnu'?",
    "answers": [
      "Maria - ucenicilor",
      "Nicodim lui Pilat",
      "Ucenicii lui Toma"
    ],
    "correct": [
      0,
      2
    ],
    "feedback": "Răspunsul corect este a, c (Ioan 20:18)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.05",
    "question": "Când a auzit Simon Petru că este Domnul, și-a pus haina pe e' și s-a încins, căci era _, și s-a aruncat în mare.",
    "answers": [
      "Gol",
      "Murdar",
      "Dezbrăcat",
      "Obosit"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Dezbrăcat (Ioan 21:7)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.05",
    "question": "Unde s-a arătat Isus a treia oară ucenicilor, după învierea Sa?",
    "answers": [
      "La Betania",
      "La Marea Tiberiadei",
      "La Sfântul Mormânt",
      "La Muntele Măslinilor"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: La Marea Tiberiadei."
  },
  {
    "card": "Cartea lui Ioan · Card 3.05",
    "question": "Isus l-a întrebat pe Toma, de trei ori: \"Mă iubești?'",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 21:15)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.06",
    "question": "Pe cine a întrebat Domnul Isus: „Vrei să te faci sănătos\"?",
    "answers": [
      "Pe femeia samarineancă de la fântâna lui Iacov",
      "Pe orbul din naștere",
      "Pe omul înduplecatul pe pat",
      "Pe omul bolnav de 38 de ani"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Pe omul bolnav de 38 de ani."
  },
  {
    "card": "Cartea lui Ioan · Card 3.06",
    "question": "Cine a fost întrebat de niște preoți și leviți: \"Ești Ilie?”",
    "answers": [
      "Petru",
      "Ioan Botezătorul",
      "Filip",
      "Natanael"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Ioan Botezătorul (Ioan 1:21)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.06",
    "question": "„Scoală-te, ridică-ți patul și umblă\", a zis Domnul Isus:",
    "answers": [
      "omului bolnav de 36 de ani",
      "soacrei lui Petru",
      "unui orb"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 5:6)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.06",
    "question": "Cercetați _, pentru că socotiți că în ele aveți viata veșnică, dar tocmai ele mărturisesc despre Mine.",
    "answers": [
      "Scripturile",
      "Legea",
      "Prorocii",
      "Poruncile"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Scripturile (Ioan 5:3)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.06",
    "question": "Cine și cui a zis: „n-am pe nimeni să mă bage în scăldătoare când se tulbură apa și, până să mă duc eu, se coboară altul înaintea mea.”",
    "answers": [
      "Omul orb din naștere",
      "Femeia samarineancă",
      "Omul care se târa pe picioare",
      "Omul bolnav de 38 de ani"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Omul bolnav de 38 de ani."
  },
  {
    "card": "Cartea lui Ioan · Card 3.06",
    "question": "Eu nu umblu după slava cară vine de la oameni, a zis Petru.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 5:7)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.07",
    "question": "Cum se numea tatăl lui Simon Petru?",
    "answers": [
      "Zebedei",
      "Iona",
      "Andrei",
      "Alfeu"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Iona (Ioan 1:142)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.07",
    "question": "Cum se numea fratele lui Andrei, ucenicul lui Ioan?",
    "answers": [
      "Simon Petru",
      "Ioan",
      "Iacov",
      "Filip"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Simon Petru (Ioan 1:40)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.07",
    "question": "Lucruri mai mari decât acestea vei vedea, i-a- zis Isus tui:",
    "answers": [
      "Petru",
      "Varianta b",
      "Natanael"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c (Ioan 1:)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.07",
    "question": "Isus i-a zis 'ui Natanael: „Te-am văzut mai înainte -ca să te cheme Filip, când erai __.”",
    "answers": [
      "La fântâna lui Iacov",
      "La Betesda",
      "Sub smochin",
      "Sub frunzele viei"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Sub smochin (Ioan 1:148)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.07",
    "question": "Cine și cui a zis: 'tam găsit pe Acela despre care a scris Moise, în Lege, și Prorocii: pe Isus din Nazaret\"?",
    "answers": [
      "Andrei lui Filip",
      "Filip lui Natanael",
      "Iacov lui Ioan",
      "Natanael lui Simon Petru"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Filip lui Natanael."
  },
  {
    "card": "Cartea lui Ioan · Card 3.07",
    "question": "Andrei s-a dus să vadă unde locuia Isus și în acea zi a rămas la El.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 1:3)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.08",
    "question": "De unde era Iosif, omul care s-a ocupat de îngroparea lui",
    "answers": [
      "Din Betania",
      "Din Nazaret",
      "Din Capernaum",
      "Ica Arimateea"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Ica Arimateea (Ioan 19:38)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.08",
    "question": "Despre cine scrie că: „NiciunuI din oasele Lui nu va fi sfărâmat\"?",
    "answers": [
      "Despre Ioan Botezătorul",
      "Despre Moise",
      "Domnul Isus",
      "Despre oasele Iacov"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 19:36)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.08",
    "question": "Isus a zis: „S-a isprăvitl”, când:",
    "answers": [
      "l-au străpuns coasta",
      "a luat oțetul",
      "I-au zdrobit fluierele picioarelor"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este Â b (Ioan 19:30)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.08",
    "question": "Lui Isus, unul din ostași l-a străpuns coasta cu o suliță; și îndată a ieșit din ea _ și _.",
    "answers": [
      "Sânge, apă",
      "Apă, vânt",
      "Sânge, foc",
      "Apă, lapte"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Sânge, apă (Ioan 19:34)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.08",
    "question": "Cine și cui a zis: „Domnule, dacă L-ai luat, spune-mi unde L-ai pus și mă voi duce să-L iau\"?",
    "answers": [
      "Maria, Mariei Magdalenei",
      "Maria, lui Isus (crezând că e grădinarul)",
      "Marta, lui Isus",
      "Maria, la poalele crucii"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Maria, lui Isus (crezând că e grădinarul) (Ioan 20:15)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.08",
    "question": "În dimineața învierii, Isus s-a întâlnit cu Maria, iar seara cu ucenicii.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 20:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.09",
    "question": "IAtie În ce loc din Ierusalim era situată scăldătoarea Betesda?",
    "answers": [
      "Lângă poarta Damisc",
      "Lângă poarta Oilor - loal",
      "Lângă poarta Estului - loc",
      "Lângă poarta Inului - dom"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Lângă poarta Oilor - loal."
  },
  {
    "card": "Cartea lui Ioan · Card 3.09",
    "question": "Ce reacție au avut ucenicii când L-au văzut pe Isus umblând pe mare?",
    "answers": [
      "S-au înfricoșat",
      "Au crezut în El",
      "Au fost bucuroși",
      "S-au minunat"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: S-au înfricoșat (Ioan 6:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.09",
    "question": "Despre cine s-a spus: „Cum are Omul acesta învățătură, căci n-a învățat niciodată?”",
    "answers": [
      "Petru",
      "Ioan Botezătorul",
      "Isus"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c (Ioan 7:15)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.09",
    "question": "„Cine dintre voi este fără păcat să arunce cel dintâi cu în ea.”",
    "answers": [
      "Cuțitul",
      "Bâta",
      "Piatra",
      "Sabia"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Piatra (Ioan 8:7)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.09",
    "question": "În ce localitate au mers noroadele, cu corăbiile ca să-L caute pe ISUS?",
    "answers": [
      "Betania",
      "Capernaum",
      "Cana din Galileea",
      "Samaria"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Capernaum (Ioan 6:)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.09",
    "question": "Nici frații lui Isus, nu credeau în El.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 7:)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.10",
    "question": "Isus i-a zis: „Eu sunt Învierea și...?",
    "answers": [
      "Drumul",
      "Adevărul",
      "Lumina",
      "Viața"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Viața (Ioan 11:25)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.10",
    "question": "Ce trebuie să se întâmple cu bobul de grâu pentru a aduce roadă?",
    "answers": [
      "Să germineze",
      "Să se-nclină",
      "Să moară",
      "Să cească"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Să moară (Ioan 12:24)."
  },
  {
    "card": "Cartea lui Ioan · Card 3.10",
    "question": "„Da, Doamne cred că Tu ești Hristosul, Fiul lui Dumnezeu, care trebuia să vină în lume\", a zis:",
    "answers": [
      "Maria",
      "femeia din Samaria",
      "Marta"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c."
  },
  {
    "card": "Cartea lui Ioan · Card 3.10",
    "question": "Isus a zis: „Eu am venit în lumea aceasta pentru judecată: ca cei ce nu văd să _ și cei ce văd să __",
    "answers": [
      "Vadă, ajungă orbi",
      "Plângă, se-nvinuiască",
      "Creadă, se-ntoarcă",
      "Deslușească, se-ntrebe"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Vadă, ajungă orbi."
  },
  {
    "card": "Cartea lui Ioan · Card 3.10",
    "question": "Cine a zis: Dumnezeu n-ascultă pe păcătoși, ci, dacă este cineva temător de Dumnezeu și face voia Lui, pe acela îl ascultă?",
    "answers": [
      "Petru, apostolul lui Isus",
      "Nicodim, fariseul",
      "Ioan, discipolul iubit",
      "Orbul din nastere vindecat de Isus"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Orbul din nastere vindecat de Isus."
  },
  {
    "card": "Cartea lui Ioan · Card 3.10",
    "question": "Isus a zis: Cine crede în Mine, chiar dacă ar fi murit, va trăi.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 11:25)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.01",
    "question": "I'ô Cine L-a întrebat pe Isus: „Ești Tu Împăratul iudeilor\"?",
    "answers": [
      "Caifa",
      "Pilat",
      "Irodul",
      "Iustin"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Pilat (Ioan 18:33)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.01",
    "question": "Cine și cui a zis: ”Nouă nu ne este Îngăduit de Lege să omorâm pe nimeni\"?",
    "answers": [
      "Iudeii, lui Pilat",
      "Pilat, iudeilor",
      "Caifa, mulțimii",
      "Solații, lui Isus"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Iudeii, lui Pilat (Ioan 18:31)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.01",
    "question": "Când Pilat a zis: ”Atunci un Împărat tot estil”, cum a reactionat Isus?",
    "answers": [
      "nu a răspuns",
      "a zis: ”Da”",
      "a zis: \"Eu sunt Împărat”"
    ],
    "correct": [
      1,
      2
    ],
    "feedback": "Răspunsul corect este b, c (Ioan 18:37)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.01",
    "question": "Osatașii, L-au îmbrăcat pe Isus cu o haină de _",
    "answers": [
      "Albă",
      "Roșie",
      "Purpură",
      "Neagră"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Purpură (Ioan 19:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.01",
    "question": "Ce vină, demnă de moarte potrivit Legii tor, au adus iudeii împotriva lui Isus - motiv pentru care lui Pilat i-a fost și mai mare frică?",
    "answers": [
      "Că a stricat templul",
      "Că S-a făcut pe Sine Fiul lui",
      "Că a vrăjit poporul",
      "Că a refuzat să se închine împaratului"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Că S-a făcut pe Sine Fiul lui."
  },
  {
    "card": "Cartea lui Ioan · Card 4.01",
    "question": "Preoții cei mai de seamă și aprozii, nu au cerut răstignirea",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 19:7)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.02",
    "question": "Ce a răspuns Isus când I S-a zis: udă-ne totdeauna această pâine\"?",
    "answers": [
      "Eu sunt Ușa oilor",
      "Eu sunt Vița adevărată",
      "Eu sunt Lumina lumii",
      "Eu sunt Pâinea vieții"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Eu sunt Pâinea vieții."
  },
  {
    "card": "Cartea lui Ioan · Card 4.02",
    "question": "Isus a zis: ”pe cel ce vine la Mine, nu-t voi izgoni...\"?",
    "answers": [
      "Înuntru",
      "Sus",
      "Afară",
      "Departe"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Afară (Ioan 6:37)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.02",
    "question": "Dacă ati crede pe..., M-ați crede și pe Mine, pentru că el a scris despre Mine.",
    "answers": [
      "Moise",
      "Ioan Botezătorul",
      "Ilie"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 5:46)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.02",
    "question": "Nimeni nu s-a suit în cer, afară de Cel ce S-a coborât din cer, adică __, care este în cer.",
    "answers": [
      "Fiul omului",
      "Cuvântul",
      "Logosul",
      "Mântuitorul"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Fiul omului (Ioan 3:13)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.02",
    "question": "Cei ce au făcut binele vor învia pentru... iar cei ce au făcut răul?",
    "answers": [
      "Înviere, pedeapsă",
      "Slava, condamnare",
      "Ție, pierzanie",
      "Viată, judecată"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Viată, judecată (Ioan 5:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.02",
    "question": "Oricine face răut urăște lumina și nu vine la lumină, ca să nu i se vădească faptele.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 3:20)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.03",
    "question": "Cum se numea omul căruia Petru i-a tăiat urechea?",
    "answers": [
      "Iacov",
      "Malhu",
      "Simeon",
      "Iuda"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Malhu (Ioan 18:10)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.03",
    "question": "Cine a zis: „Dacă am vorbit rău, arată ce am spus rău, dar, dacă am vorbit bine, de ce mă bați?”",
    "answers": [
      "Domnul Isus",
      "Petru",
      "Iacov",
      "Ioan"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 18:23)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.03",
    "question": "A zis lui Petru: „Nu cumva și tu ești unul din ucenicii omului acestuia?”",
    "answers": [
      "portărița",
      "o femeie oarecare",
      "slujnica"
    ],
    "correct": [
      0,
      2
    ],
    "feedback": "Răspunsul corect este a, c (Ioan 18:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.03",
    "question": "„Și-au împărțit _ Mele între ei, și pentru _ Mea au tras la sorti!'",
    "answers": [
      "Mantia, hainele",
      "Veșmintele, mantia",
      "Hainele, cămașa",
      "Tunica, veșmintele"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Hainele, cămașa (Ioan 19:214)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.03",
    "question": "Cine și cum L-a trimis pe Domnul Isus, la marele preot, Caiafa?",
    "answers": [
      "Pilat, liber",
      "Marele preot Ana, legat",
      "Gardianul templului, cu lanțuri",
      "Un soldat roman, în arest"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Marele preot Ana, legat."
  },
  {
    "card": "Cartea lui Ioan · Card 4.03",
    "question": "Caiafa, L-a întrebat pe Isus, ”Ce este adevărul\"?",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 18:38)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.04",
    "question": "Cine a zis: \"Nu sunt eu Hristosut, ci sunt trimis înaintea",
    "answers": [
      "Andrei",
      "Phillip",
      "Natanael",
      "Ioan Botezătorul"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Ioan Botezătorul (Ioan 3:28)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.04",
    "question": "Cine are mireasă este...?",
    "answers": [
      "Prieten al mirelui",
      "Oaspete",
      "Mire",
      "Păzitor"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Mire (Ioan 3:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.04",
    "question": "Când Isus ostenit de călătorie, s-a așezat la fântâna din Sihar, ucenicii Lui se duseseră în cetate să:",
    "answers": [
      "cumpere",
      "caute o gazdă mâncare",
      "evanghelizeze cetatea"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a."
  },
  {
    "card": "Cartea lui Ioan · Card 4.04",
    "question": "că a dat pe Fiindcă atât de mult a iubit Dumnezeu _, singurul Lui Fiu, pentru ca oricine crede în El să nu piară, ci să aibă viața veșnică.",
    "answers": [
      "Lumea",
      "Poporul ales",
      "Împărăția cerurilor",
      "Oamenii păcătoși"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Lumea (Ioan 3:16)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.04",
    "question": "Pe ce subiect, ucenicii lui Ioan au avut o neînțelegere cu un iudeu?",
    "answers": [
      "Sabatul",
      "Slujbele din templu",
      "Postul",
      "Curătare"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Curătare (Ioan 3:25)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.04",
    "question": "Tatăl iubește pe Fiul și a dat toate lucrurile în mâna Lui.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 3:35)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.05",
    "question": "Cine a zis despre Isus: \"Ce pâră aduceți împotriva Omului acestuia?”",
    "answers": [
      "Marele preot Ana",
      "Pilat",
      "Caiafa",
      "Unul din fariseeni"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Pilat (Ioan 18:29)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.05",
    "question": "Pe cine au cerut iudei, lui Pilat, să le elibereze, în locul lui Isus?",
    "answers": [
      "Baraba",
      "Dimas",
      "Gestas",
      "Simeon"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Baraba (Ioan 18:3)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.05",
    "question": "Cine a zis despre Isus: ”Eu nu găsesc nicio vină în El\"?",
    "answers": [
      "Caiafa",
      "Pilat",
      "un soldat roman"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este b (Ioan 16:38)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.05",
    "question": "Ei n-au intrat în odaia de judecată, ca să nu se _ și să poată mânca pastile.",
    "answers": [
      "Desfiin",
      "Pângărească",
      "Spurce",
      "Înjosească"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Spurce (Ioan 16:28)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.05",
    "question": "Cine și cui a zis: ”Eu pentru asta M-am născut și am venit în lume ca să mărturisesc despre adevăr\"?",
    "answers": [
      "Domnul Isus, lui Irod",
      "Domnul Isus, lui Pilat",
      "Domnul Isus, lui Caiafa",
      "Domnul Isus, ucenicilor"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Domnul Isus, lui Pilat (Ioan 18:)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.05",
    "question": "Baraba era un învățător al legii.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă."
  },
  {
    "card": "Cartea lui Ioan · Card 4.06",
    "question": "Cine a zis: ”Nu sunt eu Hristosul, ci sunt trimis înaintea",
    "answers": [
      "Andrei",
      "Filipi",
      "Natanael",
      "Ioan Botezătorul"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Ioan Botezătorul (Ioan 3:28)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.06",
    "question": "Cine are mireasă este...7",
    "answers": [
      "Oaspete",
      "Prieten",
      "Mire",
      "Martor"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Mire (Ioan 3:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.06",
    "question": "Când Isus ostenit de călătorie, s-a așezat la fântâna din Sihar, ucenicii Lui se duseseră în cetate să:",
    "answers": [
      "cumpere",
      "caute o gazdă mâncare",
      "evanghelizeze cetatea"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 4:8)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.06",
    "question": "că a dat pe Fiindcă atât de mult a iubit Dumnezeu _, singurul Lui Fiu, pentru ca oricine crede în Et să nu piară, ci să aibă viata veșnică.",
    "answers": [
      "Lumea",
      "Oamenii",
      "Păcatul",
      "Întunericul"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Lumea (Ioan 3:16)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.06",
    "question": "Pe ce subiect, ucenicii lui Ioan au avut o neînțelegere cu un iudeu?",
    "answers": [
      "Sabatul",
      "Legea",
      "Templul",
      "Curătare"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Curătare (Ioan 3:25)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.06",
    "question": "Tatăl iubește pe Fiul și a dat toate lucrurile în mâna Lui.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 3:35)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.07",
    "question": "Cu ce l-a lovit Petru, pe robul marelui preot?",
    "answers": [
      "Sulița",
      "Sabia",
      "Bâta",
      "Cuțitul"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Sabia (Ioan 18:10)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.07",
    "question": "Isus se aduna deseori, cu ucenicii Săi, în grădina care se afla dincolo de pârâul?",
    "answers": [
      "Chedron",
      "Iordanul",
      "Iacobul",
      "Marea Moartă"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Chedron (Ioan 1)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.07",
    "question": "Isus a zis: „Proslăvește pe Fiul Tău, ca și Fiul Tău să Te proslăvească pe Tine”",
    "answers": [
      "când au venit niște greci să-L vadă",
      "în rugăciunea Sa",
      "Varianta c"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este b (Ioan 17:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.07",
    "question": "le-am dat slava pe care Mi-ai dat-o Tu, pentru ca ei să fie _, cum și Noi suntem _",
    "answers": [
      "Sfinte",
      "Vrednice",
      "Una",
      "Desăvârșite"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Una (Ioan 17:22)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.07",
    "question": "În ce împrejurare a spus Isus: ”toți să fie una, cum Tu, Tată, ești în Mine și Eu, în Tine, ca și ei să fie una în Noi\"?",
    "answers": [
      "Când a fost pe cruce",
      "Când S-a rugat",
      "Când a înviat",
      "Când a fost în Getsemani"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Când S-a rugat (Ioan 17:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.07",
    "question": "O portăriță l-a întrebat pe Petru dacă și el este ucenic al lui Isus.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 18:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.08",
    "question": "Cui a spus Maria, că motivul pentru care plângea era: „au luat pe Domnul meu și nu știu unde L-au pus\"?",
    "answers": [
      "Petrului",
      "Mariei Magdalenei",
      "Lui Isus",
      "Celor doi îngeri întâlniți la"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Celor doi îngeri întâlniți la."
  },
  {
    "card": "Cartea lui Ioan · Card 4.08",
    "question": "Cum i-a salutat Isus, pe ucenici când a venit la ei după",
    "answers": [
      "Pacea fie cu voi",
      "Veniți și mâncați",
      "Mormânt",
      "Nu vă temeți"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Mormânt (Ioan 20:12)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.08",
    "question": "Simon Petru, a intrat în Mormânt și a văzut:",
    "answers": [
      "cuiele",
      "fâșiile de pânză",
      "ștergarul"
    ],
    "correct": [
      1,
      2
    ],
    "feedback": "Răspunsul corect este b, c (Ioan 20:6)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.08",
    "question": "După înviere, când s-a arătat ucenicilor, Isus le-a arătat",
    "answers": [
      "Mâinile, coasta",
      "Picioarele, fruntea",
      "Mâinile, genunchii",
      "Capul, umerii"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Mâinile, coasta (Ioan 20:20)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.08",
    "question": "Cui i-a zis Isus: „Adu-ți degetul încoace și uită-te la mâinile Mele și adu-ți mâna și pune-o în coasta Mea\"?",
    "answers": [
      "Filip",
      "Petru",
      "Ioan",
      "Toma"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Toma (Ioan 20:27)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.08",
    "question": "După învierea Sa, Isus S-a mai arătat ucenicilor Săi, la Marea Rosie,",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 21:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.09",
    "question": "De când este lumea, nu s-a auzit să fi deschis cineva ochii unui orb din naștere - a zis?",
    "answers": [
      "Phariseii",
      "Orbul vindecat",
      "Irozii",
      "Cărturarii"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Orbul vindecat."
  },
  {
    "card": "Cartea lui Ioan · Card 4.09",
    "question": "Despre cine se spunea: „Are drac, este nebun. De ce-L ascultați?'",
    "answers": [
      "Domnul Isus",
      "Ioan Botezătorul",
      "Iuda Iscariotul",
      "Gherasia din Gadareni"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 10:20)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.09",
    "question": "„Până când ne tot ții sufletele în încordare? Dacă ești Hristosul, spune-ne, i-au zis iudeii lui:",
    "answers": [
      "Ioan Botezătorul",
      "Isus",
      "Varianta c"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este b (Ioan 10:24)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.09",
    "question": "Lazăr a ieșit cu mâinile și picioarele legate cu _ și cu fața înfășurată cu un ștergar.",
    "answers": [
      "Haine albastru închis",
      "Matase și in",
      "Fâșii de pânză",
      "Pânză de in"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Fâșii de pânză (Ioan 11:4)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.09",
    "question": "Cine era mare preot în Israel, în anul în care a fost înviat Lazăr?",
    "answers": [
      "Ioan 2:13",
      "Caiafa - Ioan ll:qg",
      "Ioan 5:1",
      "Ioan 7:37"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Caiafa - Ioan ll:qg."
  },
  {
    "card": "Cartea lui Ioan · Card 4.09",
    "question": "Ostașii au zdrobit fluierele picioarelor ta toți cei răstigniți.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 19:32)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.10",
    "question": "IAW ”M-au urât fără temei” - Cine a făcut această afirmație?",
    "answers": [
      "Ucenicii",
      "Fariseii",
      "Soldații romani",
      "Domnul Isus"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 15:25)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.10",
    "question": "În ce toc, Isus se adunase de multe ori cu ucenicii Săi?",
    "answers": [
      "În templu",
      "Pe Muntele Măslinilor",
      "În grădină",
      "La Betania"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: În grădină (Ioan 16:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.10",
    "question": "Grupul de oameni care a venit cu Iuda, să prindă pe Isus",
    "answers": [
      "preoți",
      "ostași",
      "aprozi era format din:"
    ],
    "correct": [
      1,
      2
    ],
    "feedback": "Răspunsul corect este b, c (Ioan 18:3)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.10",
    "question": "Marele preot a întrebat pe Isus despre _ Lui și despre _ Lui.",
    "answers": [
      "Ucenicii, învățătura",
      "Părinții, faptele",
      "Fratele, cuvintele",
      "Chipul, cunoștințele"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Ucenicii, învățătura (Ioan 18:19)."
  },
  {
    "card": "Cartea lui Ioan · Card 4.10",
    "question": "Cine și cui a zis: ”Mai am să vă spun multe lucruri, dar acum nu te puteți purta\"?",
    "answers": [
      "Domnul Isus, Săi însuși",
      "Petru, celorlalți",
      "Duhul Sfânt, credincioșilor",
      "Domnul Isus, ucenicilor"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Domnul Isus, ucenicilor."
  },
  {
    "card": "Cartea lui Ioan · Card 4.10",
    "question": "Isus a fost dus de la marele preot Caiafa, în odaia de judecată.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 18:28)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.01",
    "question": "'Atie Cine a văzut doi îngeri în alb șezând în locut unde fus culcat trupul lui Isus; unul la cap și altul la picioare?",
    "answers": [
      "Maria, sora lui Lazăr",
      "Maria Magdalena",
      "Maria, mama lui Isus",
      "Salomeea"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Maria Magdalena."
  },
  {
    "card": "Cartea lui Ioan · Card 5.01",
    "question": "Când L-a întâlnit pe Isus Cel înviat, Maria a crezut că es",
    "answers": [
      "Grădinarul",
      "Un profet",
      "Un măiastru",
      "Un înger"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Grădinarul (Ioan 20:15)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.01",
    "question": "„Au luat pe Domnul din Mormânt și nu știu unde pusl” - a spus Maria lui:",
    "answers": [
      "Petru",
      "Andrei",
      "ucenicului pe care-l iubea Isus"
    ],
    "correct": [
      0,
      2
    ],
    "feedback": "Răspunsul corect este a, c (Ioan 20:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.01",
    "question": "Celălalt ucenic, care ajunsese cel dintâi la mormân intrat și el; și a văzut și a _.",
    "answers": [
      "Văzut",
      "Ascultat",
      "Crezut",
      "Înțeles"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Crezut (Ioan 20:8)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.01",
    "question": "După câte zile, de la înviere, Isus S-a arătat și tui Toma?",
    "answers": [
      "Trei zile",
      "Opt zile",
      "Zece zile",
      "Patruzeci de zile"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Opt zile (Ioan 20:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.01",
    "question": "Isus i-a arătat doar lui Toma, mâinile și coasta Sa.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 20:20)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.02",
    "question": "se Cine a zis despre Isus: \" Eu nu găsesc nicio vină în El\"? I",
    "answers": [
      "Irod",
      "Caiafa",
      "Anania",
      "Pilat"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Pilat (Ioan 18:38)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.02",
    "question": "Cine a împletit o cunună de spini și a pus-o pe capul lui te? Isus?",
    "answers": [
      "Fariseii",
      "Sumo-preotul",
      "Ostasii",
      "Soldații gărzii"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Ostasii (Ioan 19:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.02",
    "question": "Pilat l-a zis lui Isus: \"Cei care Te-au dat în mâna mea sunt\":",
    "answers": [
      "preoții cei mai de seamă",
      "ucenicii Tăi",
      "neamul Tău"
    ],
    "correct": [
      0,
      2
    ],
    "feedback": "Răspunsul corect este a, c (Ioan 16:35)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.02",
    "question": "„Noi avem o Lege, și după Legea aceasta, El trebuie să moară, pentru că S-a făcut pe Sine _ a",
    "answers": [
      "Fiul lui Dumnezeu",
      "Măiastrul",
      "Profetul",
      "Împăratul Israelului"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Fiul lui Dumnezeu (Ioan 19:7)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.02",
    "question": "Cine și cui a zis: „N-ai avea nici o putere asupra Mea, dacă nu ți-ar fi fost dată de sus\"?",
    "answers": [
      "Pilat lui Isus",
      "Isus lui Iuda",
      "Caiafa lui Pilat",
      "Isus lui Pilat"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Isus lui Pilat (Ioan 19:)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.02",
    "question": "Caiafa a șezut pe scaunul de judecător, în locul numit ”Gabata”.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 19:13)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.03",
    "question": "Cu câți Iei spunea Iuda că s-ar fi putut vinde mirul nard folosit de Maria?",
    "answers": [
      "Cinci sute de lei",
      "Trei sute de lei",
      "Două sute de lei",
      "Patru sute de lei"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Trei sute de lei (Ioan 12:4)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.03",
    "question": "Despre care ucenic, din cei doisprezece, se spune că er un hot?",
    "answers": [
      "Iuda Iscarioteanu",
      "Petru",
      "Ioan",
      "Andrei"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Iuda Iscarioteanu (Ioan 12:4)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.03",
    "question": "„Vedeți că nu câștigați nimic. lată că lumea se duce dup El!\"- si-au zis între ei:",
    "answers": [
      "preoții",
      "cărturarii",
      "farise"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c (Ioan 12:19)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.03",
    "question": "„Osana! Binecuvântat este Cel ce vine în __, Împăratul ľ Israel!'",
    "answers": [
      "Numele tău",
      "Numele Tatălui",
      "Numele Domnului",
      "Numele Sfântului Duh"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Numele Domnului (Ioan 12:13)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.03",
    "question": "Care mare preot, contemporan cu Isus, a prorocit că ISU va muri pentru neam?",
    "answers": [
      "Anania",
      "Caiafa",
      "Nichodemus",
      "Iosif din Arimateea"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Caiafa (Ioan 11:)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.03",
    "question": "Au întâmpinat pe Isus ta intrarea în Ierusalim, cu ramu de smochin.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 12:12)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.04",
    "question": "Cum i s-a mai spus tui Iuda Iscarioteanul, potrivit misiunii le",
    "answers": [
      "Trădător",
      "Ucenic",
      "Apostol",
      "Vânzător"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Vânzător (Ioan 18:5)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.04",
    "question": "Care dintre ucenici, când a mers cu Isus în grădină avea ta el sabie?",
    "answers": [
      "Ioan",
      "Iacov",
      "Simon Petru",
      "Andrei"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Simon Petru (Ioan 18:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.04",
    "question": "Când au mers să prindă pe Isus, oamenii din grupul bă condus de Iuda aveau asupra lor: ii a, arme v. felinare",
    "answers": [
      "Varianta a",
      "Varianta b",
      "făclii"
    ],
    "correct": [
      0,
      1,
      2
    ],
    "feedback": "Răspunsul corect este a, b, c (Ioan 18:3)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.04",
    "question": "„Și-au Împărțit hainele Mele între ei, și pentru cămașa Mea au _.' Iată ce au făcut ostașii.",
    "answers": [
      "Tras la sorți",
      "Împărțit între ei",
      "Luat de ostași",
      "Dat la moșie"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Tras la sorți (Ioan 19:24)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.04",
    "question": "Cine și cui a zis: „Bagă-ti sabia în teacă\"?",
    "answers": [
      "Petru către ucenici",
      "Iacov către Ioan",
      "Andrei către Pilat",
      "Domnul Isus lui Petru"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Domnul Isus lui Petru."
  },
  {
    "card": "Cartea lui Ioan · Card 5.04",
    "question": "Isus a fost acuzat de iudei, înaintea lui Pilat, ca făcător de",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 18:30)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.05",
    "question": "Cum se numea, în evreiește, locul de judecată?",
    "answers": [
      "Siloam",
      "Gabata",
      "Betesda",
      "Golgotha"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Gabata (Ioan 19:13)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.05",
    "question": "Ce obiect vestimentar, îmbrăcat de Isus, nu avea nici cusătură?",
    "answers": [
      "Cămașă",
      "Haină",
      "Manta",
      "Hol"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Cămașă (Ioan 19:23)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.05",
    "question": "Însemnarea de deasupra crucii lui Isus, a fost scrisă Pilat în limba:",
    "answers": [
      "latină",
      "greacă",
      "evreiască"
    ],
    "correct": [
      0,
      1,
      2
    ],
    "feedback": "Răspunsul corect este a, b, c (Ioan 19:20)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.05",
    "question": "„Dacă dai drumul Omului acestuia, nu ești _ cu Cezarul.",
    "answers": [
      "Dușman",
      "Vrăjmaș",
      "Prieten",
      "Răzvrătit"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Prieten (Ioan 19:12)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.05",
    "question": "Ce răspuns au dat iudeii, lui Pilat, când acesta i- întrebat: 'Să răstignesc pe împăratul vostru\"?",
    "answers": [
      "'Răstignește-l pe acesta'",
      "\"Noi n-avem alt împărat decât pe cezarul \"",
      "'Noi îl vom condamna'",
      "'Ia-l și judecă-l tu'"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: \"Noi n-avem alt împărat decât pe cezarul \"."
  },
  {
    "card": "Cartea lui Ioan · Card 5.05",
    "question": "Locut unde a fost răstignit Isus era departe de cetate.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 19:20)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.06",
    "question": "Cum a fost numită mana pe care israeliții au mâncat-o în pustiu?",
    "answers": [
      "Pâine a părinților",
      "Pâine a prorocilor",
      "Pâine a sfinților",
      "Pâine din cer"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Pâine din cer (Ioan 6:31)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.06",
    "question": "Cu ce scop a trimis Dumnezeu în lume pe Domnul Isus? o",
    "answers": [
      "Ca oamenii să-L cunoască pe Dumnezeu",
      "Ca să condamne lumea",
      "Ca lumea să fie mântuită",
      "Ca să înțeleagă oamenii legea"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Ca lumea să fie mântuită (Ioan 3:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.06",
    "question": "Inchinătorii adevărați se vor închina Tatălui în:",
    "answers": [
      "Ierusalim",
      "sinagogă",
      "duh și în adevăr"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c (Ioan 14:23)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.06",
    "question": "Isus nu Se încredea în ei, pentru că _ pe toți.",
    "answers": [
      "Cunoștea",
      "Iubea",
      "Judeca",
      "Vedea"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Cunoștea (Ioan 2:214)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.06",
    "question": "Oamenii au întrebat pe Isus: ace să facem ca să săvârșim tucrările lui Dumnezeu\", El a zis: Lucrarea pe care o cere Dumnezeu este să?",
    "answers": [
      "Să păziți poruncile legii",
      "Să vă pocăiți din păcate",
      "Să iubiți pe aproapele vostru",
      "Credeți în Acela pe care L-a trimis El"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Credeți în Acela pe care L-a trimis El (Ioan 6:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.06",
    "question": "Acela peste care vei vedea Duhul coborându-Se și oprindu-Se este Cel ce botează cu Duhul Sfânt.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 1:33)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.07",
    "question": "Pentru ce fel de mâncare a zis Domnul Isus, să nu lucrăn",
    "answers": [
      "Temporară",
      "Pieritoare",
      "Omenească",
      "Trecătoare"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Pieritoare (Ioan 6:27)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.07",
    "question": "Cine a zis lui Isus: Tu ai cuvintele vieții veșnice?",
    "answers": [
      "Simon Petru",
      "Ioan Botezătorul",
      "Filipi",
      "Natanael"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Simon Petru (Ioan 6:68)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.07",
    "question": "„Învățătorule, cine a păcătuit de s-a -născut orb?\"- Isus zis:",
    "answers": [
      "părinții lui",
      "el însuși",
      "nici el, nici părinții lui"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c (Ioan 9:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.07",
    "question": "„... oricui va bea din apa pe care i-o voi da Eu în veac, ba încă, apa pe care i-o voi da Eu se va preface în într-un izvor de apă,....”",
    "answers": [
      "Se va sfinții în vecie",
      "Va ajunge la viață veșnică",
      "Nu-i va fi sete",
      "Se va cufunda în adâncuri"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Nu-i va fi sete (Ioan 4:13)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.07",
    "question": "Cum poți fi, în adevăr, ucenicul Domnului Isus?",
    "answers": [
      "Dacă Te iubești pe tine însuți",
      "Dacă rămâi în cuvântul Său",
      "Dacă urci pe muntele sfânt",
      "Dacă păzești toate lucurile"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Dacă rămâi în cuvântul Său."
  },
  {
    "card": "Cartea lui Ioan · Card 5.07",
    "question": "Dumnezeu este Duh; și cine se închină Lui trebuie să I s închine în duh și în adevăr",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 8:31)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.08",
    "question": "Pe cine a citat Ioan când a zis: ”Neteziți calea Domnului\"?",
    "answers": [
      "Cartea lui Iezechiel",
      "Cartea lui Ieremia",
      "Cartea lui Maleahi",
      "C-a,tbo lui Isaia"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: C-a,tbo lui Isaia (Ioan 1:23)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.08",
    "question": "Cine a zis lui Isus: „Este aici un băiețel care are cinci pâini de orz și doi pești\"?",
    "answers": [
      "Filip",
      "Bartolomeu",
      "Andrei",
      "Iacov"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Andrei (Ioan 6:8)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.08",
    "question": "Din ce motive iudeii căutau să-L omoare pe Domnul Isus: a",
    "answers": [
      "dezlega ziua Sabatului",
      "n-aveau motiv",
      "zicea că Dumnezeu e Tatăl Său"
    ],
    "correct": [
      0,
      2
    ],
    "feedback": "Răspunsul corect este a, c (Ioan 5:18)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.08",
    "question": "„Pâinile pe care le-am putea cumpăra cu __ __ n-ar ajunge ca fiecare să capete puțintel din ele.”",
    "answers": [
      "Două sute de lei",
      "Trei sute de lei",
      "Cinci sute de lei",
      "O sută de lei"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Două sute de lei."
  },
  {
    "card": "Cartea lui Ioan · Card 5.08",
    "question": "Câți din ucenicii pe care l-a avut Isus au fost aleși de El?",
    "answers": [
      "Zece",
      "Treisprezece",
      "Patrusprezece",
      "Doisprezece"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Doisprezece (Ioan 6:70)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.08",
    "question": "Filip a zis: ”Strângeți firimiturile rămase, ca să nu se piardă",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 6:12)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.09",
    "question": "Cine și cui a zis: „Să faceți orice 'vă va zice\"?",
    "answers": [
      "Marta, lui Lazăr",
      "Mama Domnului Isus, slugilor",
      "Maria Magdalena, discipulilor",
      "Samarineanca, în familie"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Mama Domnului Isus, slugilor."
  },
  {
    "card": "Cartea lui Ioan · Card 5.09",
    "question": "Din ce partid religios făcea parte Nicodim?",
    "answers": [
      "Ioan 2:5 Fariseilor",
      "Ioan 3:1 Saducheilor",
      "Ioan 7:45 Perușilor",
      "Ioan 1:24 Leviților"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Ioan 2:5 Fariseilor (Ioan 3:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.09",
    "question": "Ce minune făcută de Isus e considerată un început semnelor Lui:",
    "answers": [
      "Învierea lui Lazăr",
      "vindecarea Bartimeu",
      "apa în vin"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c (Ioan 2:11)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.09",
    "question": "Ioan a mărturisit: „Am văzut Duhul coborându-Se din c ca un - și oprindu-Se peste El\".",
    "answers": [
      "Cerb",
      "Vultur",
      "Porumbel",
      "Pasăre"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Porumbel (Ioan 1:32)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.09",
    "question": "Cui i-a zis Isus: „după cum a înălțat Moise șarpele pustiu, tot așa trebuie să fie înălțat și Fiul omului\"?",
    "answers": [
      "Petru",
      "Nicodim",
      "Natanael",
      "Iacob"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Nicodim (Ioan 3:10)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.09",
    "question": "Ioan a venit să boteze cu apă, ca Isus să fie făc cunoscut lui Israel.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 1:31)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.10",
    "question": "Cine a zis: „lată Mielul lui Dumnezeu, care ridică păcatul lumii\"?",
    "answers": [
      "Andrei",
      "Filip",
      "Toma",
      "Ioan"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Ioan (Ioan 1:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.10",
    "question": "De ce nu vine la (urnină cel care face răul, ci o urăște?",
    "answers": [
      "Ca să nu fie pedepsit",
      "Ca să nu fie descoperit",
      "Ca să nu i se vadă faptele",
      "Ca să nu fie acuzat"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Ca să nu i se vadă faptele."
  },
  {
    "card": "Cartea lui Ioan · Card 5.10",
    "question": "al Dumnezeu, a trimis pe Fiul Său în lume ca să...: [ui",
    "answers": [
      "judece lumea",
      "ca lumea să fie mântuită",
      "fie condamnată"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este Ioan 3: 20 b (Ioan 3:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.10",
    "question": "er n-avea trebuință să-t facă cineva mărturisiri Isus despre niciun om, fiindcă El Însuși _ ce este în om.",
    "answers": [
      "Știa",
      "Vedea",
      "Cunoștea",
      "Simțea"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Știa (Ioan 2:25)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.10",
    "question": "în La ce templu S-a referit Domnul Isus când a zis: „Stricați-l și în trei zile îl voi ridica.\"",
    "answers": [
      "La templul lui Herodes",
      "La sinagoga din Nazaret",
      "La altarul cu tămâie",
      "La templul trupuluî Său"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: La templul trupuluî Său (Ioan 2:19)."
  },
  {
    "card": "Cartea lui Ioan · Card 5.10",
    "question": "ut Nimeni n-a văzut vreodată pe Dumnezeu.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 1:16)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.01",
    "question": "Căruia dintre ucenici i se mai spunea și Geamăn?",
    "answers": [
      "Filip",
      "Toma",
      "Bartolomeu",
      "Matei"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Toma (Ioan 11:16)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.01",
    "question": "Cum se numeau surorile lui Lazăr, omul înviat de Isus? Cum au reacționat iudeii când Isus a zis: ”Sunt Fiut lui",
    "answers": [
      "Maria și Marta",
      "Eva și Sara",
      "Elisabeta și Anna",
      "Ioana și Saloma"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Maria și Marta (Ioan 11.1)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.01",
    "question": "Dumnezeu\"?",
    "answers": [
      "I s-au închinat",
      "au zis că hulește",
      "au luat pietre ca să-L ucidă"
    ],
    "correct": [
      1,
      2
    ],
    "feedback": "Răspunsul corect este b, c (Ioan 10:31)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.01",
    "question": "Isus nu mai umbla pe față printre iudei, a plecat din Betania în ținutul de lângă pustiu, într-o cetate numită _ și a rămas acolo cu ucenicii Săi.",
    "answers": [
      "Ierihon",
      "Samaria",
      "Efraim",
      "Galilea"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Efraim (Ioan 11:54)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.01",
    "question": "Când a auzit că vine Isus în Betania, care din surorile lui Lazăr, I-a ieșit înainte?",
    "answers": [
      "Maria",
      "Marta",
      "Eva",
      "Saloma"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Marta (Ioan 11:20)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.01",
    "question": "De când l-a înviat pe Lazăr, iudeii s-au hotărât să ucidă pe Isus.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 11:53)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.02",
    "question": "Cine a zis lui Isus: „Doamne, vino până nu moare micuțul meu\"?",
    "answers": [
      "Un fariseul",
      "Un satrap",
      "Un magistrat",
      "Un slujbaș împărătesc"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Un slujbaș împărătesc."
  },
  {
    "card": "Cartea lui Ioan · Card 6.02",
    "question": "Cine a zis: „Stiu, că are să vină Mesia (căruia I se zice Hristos); când va veni El, are să ne spună toate lucrurile\"?",
    "answers": [
      "Femeia adulteră",
      "Femeia canaanită",
      "Femeia samariteancă",
      "Femeia din Magdala"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Femeia samariteancă."
  },
  {
    "card": "Cartea lui Ioan · Card 6.02",
    "question": "Fiul slujbașului împărătesc, vindecat de Isus, suferise de:",
    "answers": [
      "epilepsie",
      "chinuit de un duh rău",
      "friguri"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este 4:25 c (Ioan 4:)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.02",
    "question": "Isus a zis: „Mâncarea Mea este să __ Celui ce M-a trimis și să _ lucrarea Lui.”",
    "answers": [
      "Fac voia, împlinesc",
      "Cred în, mărturisesc",
      "Iubesc pe, slujesc",
      "Ascult de, respect"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Fac voia, împlinesc."
  },
  {
    "card": "Cartea lui Ioan · Card 6.02",
    "question": "Unde a întâlnit Isus, o multime de bolnavi, orbi, șchiopi, uscați, care așteptau mișcarea apei?",
    "answers": [
      "La fântâna Iacov",
      "La Iordanul Betania",
      "La liniștea din Getsimani",
      "La scăldătoarea Betesda"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: La scăldătoarea Betesda (Ioan 5:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.02",
    "question": "Când Mă veți lăsa singur, Tatăl este cu Mine, le-a zis Isus ucenicilor.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 16:32)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.03",
    "question": "'Atie În ce sat locuiau surorile, Maria și Marta?",
    "answers": [
      "În Nazaret",
      "Weie Betania",
      "În Capernaum",
      "În Ieriho"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Weie Betania (Ioan 11:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.03",
    "question": "Cum i-a numit Isus pe cei care au venit înainte de El, de care oile n-au ascultat?",
    "answers": [
      "Hoți și tâlhari",
      "Păcătoși și păgâni",
      "Vrăjitori și demonizați",
      "Mincinoși și sedători"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Hoți și tâlhari (Ioan 10:8)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.03",
    "question": "Întrebați de iudei cum de vede fiul vostru, părinții orbului vindecat, au răspuns:",
    "answers": [
      "Isus l-a vindecat",
      "nu știm",
      "întrebați-l pe el"
    ],
    "correct": [
      1,
      2
    ],
    "feedback": "Răspunsul corect este b, c (Ioan 9:18)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.03",
    "question": "Când a luat Isus, a zis: „S-a isprăvit!' Apoi Și-a plecat capul și Și-a dat duhul.",
    "answers": [
      "Apa",
      "Vinul",
      "Oțetul",
      "Sângele"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Oțetul (Ioan 19:30)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.03",
    "question": "Cui a zis orbul, despre Isus: „Dacă este un- păcătos, ne-' știu; eu una știu: că eram orb, și acum văd\"?",
    "answers": [
      "Iudeilor",
      "Fariseilor",
      "Ucenicilor",
      "Părinților"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Fariseilor (Ioan 9:214)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.03",
    "question": "Când a venit, Isus a- aflat că Lazăr era de două zile în Mormânt.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 11:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.04",
    "question": "Cine a zis: ”Nu judecați după înfățișare, ci judecați după dreptate\"?",
    "answers": [
      "Moise",
      "Ioan Botezătorul",
      "Petru",
      "Domnul Isus"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 7:24)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.04",
    "question": "Cui îi este rob, cel care trăiește în păcat?",
    "answers": [
      "Diavolului",
      "Legii",
      "Păcatului",
      "Lumii"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Păcatului (Ioan 8:34)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.04",
    "question": "Voia Tatălui Meu este ca oricine vede pe Fiul și... să aibă viata veșnică; și Eu îl voi învia în ziua de apoi.\"",
    "answers": [
      "vine după El",
      "crede în El",
      "aude cuvintele Lui"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este b (Ioan 6:40)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.04",
    "question": "că te-ai făcut sănătos; de acum să nu mai, ca să nu ti se întâmple ceva mai rău.”",
    "answers": [
      "Păcătuiești",
      "Minți",
      "Ucigi",
      "Furi"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Păcătuiești (Ioan 5:14)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.04",
    "question": "Cine crede în Mine, din inima lui vor curge râuri de apă vie, cum zice?",
    "answers": [
      "Legea",
      "Prorocii",
      "Cartea",
      "Scriptura"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Scriptura (Ioan 7:36)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.04",
    "question": "Cine nu cinstește pe Fiul, nu cinstește pe Tatăl, care L-a trimis.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 5:23)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.05",
    "question": "Ce înseamnă scăldătoarea Siloamului?",
    "answers": [
      "Betesda",
      "Trimis",
      "Gihon",
      "Cedron"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Trimis (Ioan 9:)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.05",
    "question": "Cui i-au fost adresate întrebările: \"Unde sunt pârâșii tăi? Nimeni nu te-a osândit?\"",
    "answers": [
      "Femeii prinsă în preacurvie",
      "Mariei Magdalenei",
      "Samarinei",
      "Mărturei"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Femeii prinsă în preacurvie."
  },
  {
    "card": "Cartea lui Ioan · Card 6.05",
    "question": "El s-a dus, s-a spălat și s-a întors văzând bine.",
    "answers": [
      "omul orb din naștere",
      "Bartimeu",
      "un oarecare orb"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este Ioan 8:10 a (Ioan 9:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.05",
    "question": "Isus a zis: -”nu fac nimic de la Mine însumi, ci vorbes după cum M—a învățât _",
    "answers": [
      "Duhul Sfânt",
      "Legea",
      "Tatăl Meu",
      "Profetul"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Tatăl Meu (Ioan 8:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.05",
    "question": "Cui i-au recomandat colegii de partid: ”Cercetează bine ș vei vedea că din Galileea nu s-a ridicat nici un proroc\"?",
    "answers": [
      "Iosifului din Arimateea",
      "Nicodim",
      "Iosefului din Nazaret",
      "Gamalielului"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Nicodim (Ioan 7:50)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.05",
    "question": "Ioan n-a făcut niciun semn, dar tot ce a spus despre Isu era adevărat.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 10:41)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.06",
    "question": "Cine a zis: „Trebuie să vă nașteți din nou\"?",
    "answers": [
      "Ioan Botezătorul",
      "Petru",
      "Nicodim",
      "Domnul Isus"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 3:7)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.06",
    "question": "La cine a făcut Ioan referire când a zis: „După mine vine un Om, care este înaintea mea, căci era înainte de mine\"?",
    "answers": [
      "Ioan Botezătorul",
      "Sfântul Duh",
      "Domnul Isus",
      "Mesia"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 1:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.06",
    "question": "nu poate vedea Împărăția lui „Dacă un om nu Dumnezeu.\"",
    "answers": [
      "dă zeciuială",
      "merge la sinagogă",
      "se naște din nou"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c (Ioan 33)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.06",
    "question": "\" Cuvântul S-a făcut - și a locuit printre noi, plin de har și de adevăr. i'",
    "answers": [
      "Trup",
      "Duh",
      "Inimă",
      "Minte"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Trup (Ioan 1:14)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.06",
    "question": "Cine a zis: „Stricați templul acesta și în 3 zile îl voi ridicaâ?",
    "answers": [
      "Petru",
      "Ioan",
      "Iuda",
      "Domnul Isus"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 2:19)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.06",
    "question": "Lumina luminează în întuneric, și întunericul a biruit-o.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 1:5)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.07",
    "question": "Câți ucenici au mers la pescuit pe Marea Tiberiadei, după",
    "answers": [
      "Doisprezece",
      "Șapte",
      "Cinci",
      "Zece"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Șapte (Ioan 21:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.07",
    "question": "Cine și cui a zis: ”Copii aveți ceva de mâncare\"?",
    "answers": [
      "Isus, ucenicilor",
      "Isus, lui Petru",
      "Isus, lui Ioan",
      "Isus, mulțimii"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Isus, ucenicilor (Ioan 21:4)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.07",
    "question": "Ioan a zis lui Petru: „Este Domnull” Când a auzit Simon Petru că este Domnul:",
    "answers": [
      "și-a pus haina pe el",
      "s-a încins",
      "s-a încălțat"
    ],
    "correct": [
      0,
      1
    ],
    "feedback": "Răspunsul corect este a, b (Ioan 21:7)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.07",
    "question": "„Aruncați mreaja în partea _ a corăbiei și veți găsi le a zis Isus, ucenicilor.",
    "answers": [
      "Stângă",
      "Dinspre apus",
      "Dreaptă",
      "Dinspre răsărit"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Dreaptă (Ioan 21:6)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.07",
    "question": "Câți pești au prins ucenicii când au aruncat mreaja în apă, la porunca lui Isus?",
    "answers": [
      "Două sute",
      "O sută cincizeci și trei",
      "Optzeci și opt",
      "Nouăzeci și nouă"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: O sută cincizeci și trei."
  },
  {
    "card": "Cartea lui Ioan · Card 6.07",
    "question": "unii din farisei I-au zis lui Isus: „Doar n-om fi și noi orbil”",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 9:140)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.08",
    "question": "Cine a pus în inima lui Iuda, gândul să vândă pe Isus?",
    "answers": [
      "Petru",
      "Pilat",
      "Sinedriul",
      "Diavolul"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Diavolul (Ioan 13:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.08",
    "question": "Isus fiindcă iubea pe ai Săi, care erau în i-a iubit până la capăt.",
    "answers": [
      "Cer",
      "Biserică",
      "Lume",
      "Israel"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Lume (Ioan 13:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.08",
    "question": "Fruntașii care credeau în Isus, nu-L mărturiseau pe față:",
    "answers": [
      "de frica fariseilor",
      "ca să nu fie dați afară din sinagogă",
      "iubeau mai mult slava oamenilor"
    ],
    "correct": [
      0,
      1,
      2
    ],
    "feedback": "Răspunsul corect este a, b, c (Ioan 12:42)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.08",
    "question": "ca să nu vă cuprindă Umblați ca unii care aveți _, întunericul: cine umblă în întuneric nu știe unde merge.",
    "answers": [
      "Lumina",
      "Adevărul",
      "Viața",
      "Cuvântul"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Lumina (Ioan 12:35)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.08",
    "question": "Cine și cui a zis: „Ce ai să faci, fă repede\"?",
    "answers": [
      "Petru, lui Isus",
      "Maria, lui Ioan",
      "Pilat, lui Isus",
      "Domnul Isus, lui Iuda"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Domnul Isus, lui Iuda."
  },
  {
    "card": "Cartea lui Ioan · Card 6.08",
    "question": "Isus a spălat de multe ori picioarele ucenicilor Săi.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 13:27)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.09",
    "question": "Cine a fost ucenic al lui Isus, pe ascuns?",
    "answers": [
      "Nicodim",
      "Iosif din Arimateea",
      "Lazar",
      "Bartimeu"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Iosif din Arimateea (Ioan 19:)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.09",
    "question": "În ce locuri, i-a spus Isus preotului Ana, că i-a învățat pe oameni? A adus aproape 100 de litri, smirnă amestecată cu aloe",
    "answers": [
      "Sinagogi și Templu",
      "În Templu și Iotapata",
      "În Betania și Sfat",
      "La Cana și Ierusalim"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Sinagogi și Templu."
  },
  {
    "card": "Cartea lui Ioan · Card 6.09",
    "question": "pentru îngroparea lui Isus:",
    "answers": [
      "Nicodim",
      "Iosif",
      "Maria Magdalena"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este 18:13, 19-20 3- a (Ioan 19:3)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.09",
    "question": "Unul din robii marelui preot, rudă cu acela căruia îi _ urechea, a zis: „Nu te-am văzut eu cu El în grădină?\"",
    "answers": [
      "Rănise Ioan",
      "Lovise Andrei",
      "Tăiase Petru",
      "Omorâse Matei"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Tăiase Petru (Ioan 18:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.09",
    "question": "Când a zis Isus: „N-am pierdut pe niciunul din aceia p care Mi i-ai dat\"?",
    "answers": [
      "La Cina cea de Taină",
      "În rugăciunea Sa",
      "La Răstignire",
      "După Înviere"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: În rugăciunea Sa (Ioan 18:)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.09",
    "question": "Rabuni, în evreiește înseamnă învățătorule.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă."
  },
  {
    "card": "Cartea lui Ioan · Card 6.10",
    "question": "Prin cine a fost dată Legea?",
    "answers": [
      "Aaron",
      "David",
      "Avraam",
      "Moise"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Moise (Ioan 1:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.10",
    "question": "A fost întrebat de niște preoți și 'eviți din Ierusalim trimiși de farisei: 'Tu cine esti”?",
    "answers": [
      "Isus din Nazaret",
      "Bartimeu orbul",
      "Ioan Botezătorul",
      "Lazar din Betania"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Ioan Botezătorul (Ioan 1:19)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.10",
    "question": "Era din cetatea Betsaida:",
    "answers": [
      "Petru",
      "Andrei",
      "Varianta c"
    ],
    "correct": [
      0,
      1,
      2
    ],
    "feedback": "Răspunsul corect este a, b, c (Ioan 1:144)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.10",
    "question": "__ _, Tu ești Împăratul Natanael l-a răspuns: „Rabi, Tu ești [ui Israell”",
    "answers": [
      "Fiul lui Dumnezeu",
      "Împăratul Israelului",
      "Mesia, Fiul celui Preaînalt",
      "Regele Iudeilor"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Fiul lui Dumnezeu (Ioan 1:14)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.10",
    "question": "Cine când l-a auzit pe Ioan spunând: „lată Mielul lui Dumnezeu\", L-a urmat pe Isus?",
    "answers": [
      "Trei din ucenicii săi",
      "Patru oameni",
      "Cinci credincioși",
      "Doi din ucenicii lui"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Doi din ucenicii lui (Ioan 1:35)."
  },
  {
    "card": "Cartea lui Ioan · Card 6.10",
    "question": "A venit la ai Săi, și ai Săi nu L-au primit.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 1:11)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.01",
    "question": "Cine a zis: „De ce nu s-a vândut acest mir cu trei sute de Iei și să se fi dat săracilor”?",
    "answers": [
      "Petru",
      "Iuda Iscarioteanu",
      "Ioan",
      "Toma"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Iuda Iscarioteanu."
  },
  {
    "card": "Cartea lui Ioan · Card 7.01",
    "question": "De unde s-a auzit glasul care zicea: „L-am proslăvit și-L voi mai proslăvi\"?",
    "answers": [
      "Din cer",
      "Din templu",
      "Din nori",
      "Din apă"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Din cer (Ioan 12:28)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.01",
    "question": "”El Însuși Mi-a poruncit ce trebuie să spun și cum trebuie să vorbesc\", a zis:",
    "answers": [
      "Ioan Botezătorul",
      "Moise",
      "Isus"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c (Ioan 12:)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.01",
    "question": "Și după ce voi fi înălțat de pe pământ, voi atrage la Mine pe _",
    "answers": [
      "Poporul ales",
      "Credincioșii",
      "Toți oamenii",
      "Iudeii"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Toți oamenii (Ioan 12:32)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.01",
    "question": "Cine a zis lui Isus: „de curând, căutau iudeii să Te ucidă cu pietre, și Te întorci în Iudeea\"?",
    "answers": [
      "Fratele lui Isus",
      "Ucenicii lui",
      "Mama lui Isus",
      "Toți fariseii"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Ucenicii lui (Ioan 11:8)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.01",
    "question": "Cel ce mănâncă pâine cu Mine, a ridicat călcâiul împotrivi Mea.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 13:18)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.02",
    "question": "Cine a zis: ”Ce am scris, am scris\"?",
    "answers": [
      "Caiafa",
      "Anania",
      "Irod",
      "Icaw Pilat"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Icaw Pilat (Ioan 19:22)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.02",
    "question": "Cum este numit în evreiește, locul zis al ”Căpățânii”?",
    "answers": [
      "Betsaida",
      "Sihon",
      "Golgota",
      "Gat"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Golgota (Ioan 19:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.02",
    "question": "În însemnarea de deasupra crucii,... i-au sugerat lui Pilat să scrie că El a zis: sunt Împăratul iudeilor”:",
    "answers": [
      "fariseii",
      "preoții cei mai de seamă",
      "ostașii"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este b (Ioan 19:21)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.02",
    "question": "”0stașii, după ce au răstignit pe Isus, l-au luat hainele si le-au făcut _ părți, câte o parte pentru fiecare ostaș\".",
    "answers": [
      "Patru",
      "Trei",
      "Opt",
      "Doisprezece"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Patru (Ioan 19:23)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.02",
    "question": "Ce a scris Pilat, pe însemnarea pusă deasupra crucii lui",
    "answers": [
      "Regele iudeilor",
      "Acesta este Isus",
      "Isus Hristosul",
      "Isus din Nazaret, Împăratul iudeilor"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Isus din Nazaret, Împăratul iudeilor (Ioan 19:19)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.02",
    "question": "Cămașa lui Isus, n-avea nici o cusătură era dint-o singură țesătură.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 1923)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.03",
    "question": "Cine a zis: ”Am putere să-Mi dau viața și am putere s-o iau iarăși\"?",
    "answers": [
      "Părintele",
      "Domnul Isus",
      "Duhul Sfânt",
      "Iacov"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 10:18)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.03",
    "question": "Cine a uns picioarele lui Isus cu mir și le-a șters cu părul capului ei?",
    "answers": [
      "Maria",
      "Marta",
      "Salome",
      "Ioana"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Maria (Ioan 11:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.03",
    "question": "Maria, sora Martei când a ajuns unde era Isus și L-a văzut:",
    "answers": [
      "s-a aruncat la picioarele Lui",
      "I s-a închinat",
      "I-a zis: \"dacă ai fi fost aici, n-ar fi murit fratele meu”"
    ],
    "correct": [
      0,
      2
    ],
    "feedback": "Răspunsul corect este a, c (Ioan 11:32)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.03",
    "question": "Isus a murit - ca să adune într-un singur trup pe _ Dumnezeu cei risipiți.",
    "answers": [
      "Păstorii",
      "Apostolii",
      "Copiii",
      "Oile"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Copiii (Ioan 11:52)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.03",
    "question": "Cine și cui a zis: „A venit Învățătorul și te cheamă\"?",
    "answers": [
      "Maria, Marthei",
      "Marta, Mariei",
      "Plouton, Mariei",
      "Ioan, Petrei"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Marta, Mariei (Ioan 11:28)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.03",
    "question": "Cine Mă vede pe Mine vede pe Cel ce M-a trimis pe Mine.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 12:45)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.04",
    "question": "De ce nu merg oile după un străin, ci fug de el?",
    "answers": [
      "Pentru că nu au auzit niciodată glas uman",
      "Pentru că sunt frica lor naturală",
      "Pentru că le-a poruncit păstorul",
      "Nu cunosc glasul străinilor"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Nu cunosc glasul străinilor (Ioan 10:5)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.04",
    "question": "Cine este un mincinos și tată al minciunii?",
    "answers": [
      "Cain",
      "Iuda",
      "Diavolul",
      "Pilat"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Diavolul (Ioan 6:44)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.04",
    "question": "Cât este ziuă, trebuie să...; vine noaptea, când nimeni nu mai poate' lucra.",
    "answers": [
      "a. veghez",
      "lucrez lucrările Celui ce M-a trimis",
      "vindec"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este b."
  },
  {
    "card": "Cartea lui Ioan · Card 7.04",
    "question": "nu va umbla în „Eu sunt Lumina lumii; cine _ întuneric, ci va avea lumina vieții.”",
    "answers": [
      "Mă urmează pe Mine",
      "Te întorci la sine",
      "Cazi în păcat",
      "Rătăcești în umbră"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Mă urmează pe Mine (Ioan 6:12)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.04",
    "question": "La ce S-a referit Isus când a zis: ”Cine crede în Mine, din inima lui vor curge râuri de apă vie\"?",
    "answers": [
      "Ca apa curată din fântâni se va revărsa",
      "Ca vorbirea în limbi străine se va răspândi",
      "Ca tămâia mirului se va adinci în temple",
      "Ca Duhul Sfânt, pe care aveau să-L primească cei ce vor crede în El"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Ca Duhul Sfânt, pe care aveau să-L primească cei ce vor crede în El."
  },
  {
    "card": "Cartea lui Ioan · Card 7.04",
    "question": "Adevărul te va face slobod/ liber.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 8:32)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.05",
    "question": "Cine a zis: ”Nu voi bea paharul pe care Mi I-a dat Tatăl să-t beau\"?",
    "answers": [
      "Petru",
      "Domnul Isus",
      "Ioan",
      "Iacov"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 18:11)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.05",
    "question": "Cui i-a zis Isus: 'Eu am vorbit lumii pe față, n-am spus nimic în ascuns\"?",
    "answers": [
      "Marelui preot Ana",
      "Prefectului Pilat",
      "Irodului Antipa",
      "Sinedriuluişi"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Marelui preot Ana."
  },
  {
    "card": "Cartea lui Ioan · Card 7.05",
    "question": "„lată că vi-L aduc afară, ca să știți că nu găsesc nici o vină în El” - a zis:",
    "answers": [
      "Caiafa",
      "Pilat",
      "preotul Ana"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este 18:13, 19-20 b (Ioan 19:14)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.05",
    "question": "Va veni vremea când oricine vă __ să creadă că aduce o slujbă lui Dumnezeu.",
    "answers": [
      "Va izgoni",
      "Va chinui",
      "Va ucide",
      "Va robia"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Va ucide (Ioan 16:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.05",
    "question": "Din ce motiv, Iosif din Arimateea era ucenic al lui Isus, pe ascuns?",
    "answers": [
      "Din invidie",
      "De frica iudeilor",
      "Din mândrie",
      "Din îndoială"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: De frica iudeilor (Ioan 19:36)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.05",
    "question": "Iosif, fiul lui Iacov, a cerut lui Pilat trupul lui Isus pentru îngropare.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 19:38)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.06",
    "question": "Isus i-a zis: „Eu sunt Calea, Adevărul și...?",
    "answers": [
      "Puterea",
      "Mântuirea",
      "Cuvântul",
      "Viata"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Viata (Ioan 14:6)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.06",
    "question": "La cine a făcut Isus referire când a zis ucenicilor Săi: voi sunteți curați, dar! nu toți?",
    "answers": [
      "La Petru",
      "La Ioan",
      "La cel ce avea să-L vândă",
      "La Iuda"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: La cel ce avea să-L vândă."
  },
  {
    "card": "Cartea lui Ioan · Card 7.06",
    "question": "După ce Mă voi duce și vă voi pregăti un loc:",
    "answers": [
      "Mă voi întoarce",
      "vă voi lua cu Mine",
      "unde sunt Eu, să fiti și voi"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 13:10)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.06",
    "question": "\"Cine își iubește viața o va _ și cine își urăște viața în lumea aceasta o va păstra pentru viața veșnică\" - a spus Isus.",
    "answers": [
      "Pierde",
      "Găsește",
      "Păstrează",
      "Mântuiește"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Pierde (Ioan 12:25)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.06",
    "question": "De cine este osândit cel care-L nesocotește pe Isus și nu primește cuvintele Lui?",
    "answers": [
      "Legea Moisei",
      "Duhul Sfânt",
      "Tatăl Ceresc",
      "Cuvântul vestit de El"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Cuvântul vestit de El."
  },
  {
    "card": "Cartea lui Ioan · Card 7.06",
    "question": "Porunca lui Dumnezeu este viața veșnică.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 12:50)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.07",
    "question": "Cine a mers dis-de-dimineață, la Mormânt și a văzut piatrz luată?",
    "answers": [
      "Ioan și Petru",
      "Maria Magdalena",
      "Marta și Maria",
      "Nicodim"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Maria Magdalena (Ioan 20:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.07",
    "question": "Care ucenic a lipsit când Isus li S-a arătat, după înviere?",
    "answers": [
      "Toma",
      "Petru",
      "Ioan",
      "Iuda"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Toma (Ioan 20:24)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.07",
    "question": "De frica iudeilor:",
    "answers": [
      "nimeni nu vorbea de Isus, pe față",
      "Iosif era ucenic pe ascuns",
      "ucenicii erau adunați cu usile încuiate"
    ],
    "correct": [
      0,
      1,
      2
    ],
    "feedback": "Răspunsul corect este a, b, c (Ioan 7:13)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.07",
    "question": "Isus le-a zis din nou: „Pace vouă! Cum M-a _ pe Mine Tatăl, așa vă _ și Eu pe voi.”",
    "answers": [
      "Iubit, iubesc",
      "Cunoscut, cunosc",
      "Trimis, trimit",
      "Ales, aleg"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Trimis, trimit (Ioan 20:21)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.07",
    "question": "Cine si cui a zis: nu fi necredincios, ci credincios?",
    "answers": [
      "Isus lui Petru",
      "Isus lui Toma",
      "Petru lui Ioan",
      "Ioan lui Andrei"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Isus lui Toma (Ioan 20:27)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.07",
    "question": "Ucenicii stăteau cu usile încuiate când a venit Isus la ei, după înviere.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă."
  },
  {
    "card": "Cartea lui Ioan · Card 7.08",
    "question": "Cine a zis tui Isus: „Doamne, văd că ești proroc\"?",
    "answers": [
      "Maria din Betania",
      "Elisabeta",
      "Maria, mama lui Isus",
      "Femeia samariteancă"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Femeia samariteancă."
  },
  {
    "card": "Cartea lui Ioan · Card 7.08",
    "question": "Cine a spus că un proroc nu este prețuit în patria sa? „Credem din pricină că L-am auzit noi înșine și știm că",
    "answers": [
      "Ioan Botezătorul",
      "Moise",
      "Domnul Isus",
      "Evreiil"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Domnul Isus."
  },
  {
    "card": "Cartea lui Ioan · Card 7.08",
    "question": "Acesta este în adevăr Hristosul, Mântuitorul lumii.” - au",
    "answers": [
      "samaritenii",
      "fariseii spus:",
      "ucenicii"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 4:142)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.08",
    "question": "__ _ și cine este Cel ce-ți zice: „Dacă ai fi cunoscut tu ttDă-Mi să beau!” tu singură ai fi cerut să bei, și El ți-ar fi dat apă vie.”",
    "answers": [
      "Darul lui Dumnezeu",
      "Apa vie",
      "Pâinea cerului",
      "Lumina lumii"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Darul lui Dumnezeu (Ioan 14:10)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.08",
    "question": "Cui i-a vorbit Domnul Isus, Într-o cetate păgână, despre apa vieții?",
    "answers": [
      "Unui bărbat",
      "Unui fariseu",
      "Unui scriitor",
      "Unei femei"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Unei femei (Ioan 4:7)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.08",
    "question": "Mă duc la Tatăl și nu Mă veți mai vedea, te-a zis Isus, preoților.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 16:16)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.09",
    "question": "Cui a spus Maria, am văzut pe Domnul?",
    "answers": [
      "Fariseilor",
      "Ucenicilor",
      "Soldaților",
      "Maicii Sale"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Ucenicilor (Ioan 20:18)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.09",
    "question": "Cine a întrebat-o pe Maria: „Femeie\", „de ce plângi? Pe cint",
    "answers": [
      "Domnul Isus",
      "Petru",
      "Iacov",
      "Un înger"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 20:15)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.09",
    "question": "După înviere, când li S-a arătat Isus, ucenicii, când L-au văzut s-au:",
    "answers": [
      "bucurat",
      "speriat",
      "cutremurat"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 20:20)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.09",
    "question": "„Du-te la și spune-le că Mă sui la Tatăl Meu și Tatăl vostru, la Dumnezeul Meu și Dumnezeul vostru.”",
    "answers": [
      "Ucenicilor",
      "Apostolilor",
      "Frații Mei",
      "Discipolilor Mei"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Frații Mei (Ioan 20:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.09",
    "question": "Ce a zis Toma, după ce Isus l-a invitat să-l vadă și s verifice rănile?",
    "answers": [
      "„Aceasta este învierea!'",
      "„Domnul meu și Dumnezeul meu!\"",
      "„Te cred, Doamne!'",
      "„Nu pot să cred!'"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: „Domnul meu și Dumnezeul meu!\" (Ioan 20:28)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.09",
    "question": "După Înviere, Isus S-a arătat ucenicilor și lui Toma, care erau în Templu.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 20:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.10",
    "question": "Pâinea lui Dumnezeu este aceea care Se coboară din cer I și dă lumii?",
    "answers": [
      "Lumina",
      "Adevărul",
      "Calea",
      "Viața"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Viața (Ioan 6:33)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.10",
    "question": "De ce oamenii au iubit mai mult întunericul decât lumina?",
    "answers": [
      "Erau întunecate",
      "Erau ascunse",
      "Pentru că faptele lor erau",
      "Erau pedepsite"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Pentru că faptele lor erau."
  },
  {
    "card": "Cartea lui Ioan · Card 7.10",
    "question": "Cine ascultă cuvintele Mele și crede în Cel ce M-a trimis:",
    "answers": [
      "are viața veșnică",
      "nu vine la judecată",
      "a trecut din moarte ta viață."
    ],
    "correct": [
      0,
      1,
      2
    ],
    "feedback": "Răspunsul corect este a, b, c (Ioan 5:)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.10",
    "question": "Dacă nu se _ cineva din apă și din Duh, nu poate să intre în Împărăția lui Dumnezeu.",
    "answers": [
      "Naște",
      "Cântă",
      "Crede",
      "Pocăiește"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Naște (Ioan 3:5)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.10",
    "question": "Domnul Isus spune că Tatăl nu judecă pe nimeni, ci toată judecata a dat-o?",
    "answers": [
      "Tatălui",
      "Sfântului Duh",
      "Lumii",
      "Fiului"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Fiului (Ioan 5:22)."
  },
  {
    "card": "Cartea lui Ioan · Card 7.10",
    "question": "Ce este născut din Duh este duh.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 3:)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.01",
    "question": "Despre care ucenic a ieșit zvonul că nu va muri deloc?",
    "answers": [
      "Petru (Cephas)",
      "Ioan (ucenicul pe care-L iubea Isus)",
      "Toma (Didim)",
      "Filip"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Ioan (ucenicul pe care-L iubea Isus) (Ioan 21:20)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.01",
    "question": "Ce i-a spus Isus lui Petru, ca să arate cu ce fel de moarte va proslăvi el pe Dumnezeu?",
    "answers": [
      "„Mergi și nu mai păcătui de acum încolo'",
      "„când vei îmbătrâni, altul te va încinge și te va duce unde nu vei vrea\"",
      "„Iacă, Mielul lui Dumnezeu, care ridică păcatul lumii'",
      "„Cine crede în Mine, chiar dacă ar muri, va trăi'"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: „când vei îmbătrâni, altul te va încinge și te va duce unde nu vei vrea\" (Ioan 21:18)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.01",
    "question": "„Dacă vreau ca el să rămână până voi veni Eu, ce-ți pasă",
    "answers": [
      "Petru",
      "Ioan ție? - i-a zis Isus lui:",
      "Iuda"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este încinge și te va duce unde nu vei voi.” (Ioan 21:16)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.01",
    "question": "_ să creadă că aduce o Va veni vremea când oricine __ slujbă tui Dumnezeu.",
    "answers": [
      "Vă va izgoni din sinagogi",
      "Vă va osândi",
      "Vă va ucide",
      "Vă va întemniță"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Vă va ucide (Ioan 16:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.01",
    "question": "Despre ce subiect, spune Ioan că dacă s-ar fi scris cu de-amănuntul, cred că nici chiar în tumea aceasta n-ar fi putut încăpea cărțile?",
    "answers": [
      "Învățăturile lui Isus",
      "Lucruri pe care le-a făcut Isus",
      "Miracolele făcute pentru nenorociți",
      "Cuvintele pe care le-a rostit Isus"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Lucruri pe care le-a făcut Isus (Ioan 21:25)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.01",
    "question": "Eu Însumi Mă sfințesc pentru ei, ca și ei să fie sfințiți prin adevăr.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 17:19)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.02",
    "question": "Despre care om din vechime, Isus a zis: 'tel a scris despre Mine\"?",
    "answers": [
      "Avraam",
      "Iosif",
      "Davide",
      "Moise"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Moise (Ioan 5:46)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.02",
    "question": "Câți oameni au fost prezenți la prima înmulțire a pâinilor și peștilor?",
    "answers": [
      "Patru mii",
      "Trei mii cinci sute",
      "Aproape cinci mii",
      "Șase mii"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Aproape cinci mii (Ioan 6:10)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.02",
    "question": "Minunea vindecării omului bolnav de 38 de ani, Isus a făcut-o în:",
    "answers": [
      "Capernaum",
      "Ierusalim",
      "Samaria"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este b (Ioan 5:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.02",
    "question": "\"Tatăl nu M-a lăsat _, pentru că totdeauna fac ce-l este plăcut.”",
    "answers": [
      "Singur",
      "Niciodată",
      "În întuneric",
      "Fără apărare"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Singur (Ioan 8:29)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.02",
    "question": "Despre cine ni se spune: \"Voi nu l-ați auzit niciodată glasul, nu l-ați văzut deloc fata\"?",
    "answers": [
      "Duhul Sfânt",
      "Fiul lui Dumnezeu",
      "Isus Hristos",
      "Dumnezeu Tatăl"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Dumnezeu Tatăl (Ioan 5:37)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.02",
    "question": "Cuvântul Lui nu rămâne în voi, pentru că nu credeți în Acela pe care L-a trimis El.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 5:38)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.03",
    "question": "Ce face diferit Păstorul cel bun, față de cel plătit - în privința oilor?",
    "answers": [
      "Păstorul cel bun cunoaște pe fiecare oaie",
      "Păstorul cel bun își dă viața pentru oi",
      "Păstorul cel bun le numește pe nume",
      "Păstorul cel bun nu le lasă să rătăcească"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Păstorul cel bun își dă viața pentru oi (Ioan 10:11)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.03",
    "question": "Ce face păstorul plătit atunci când oile sunt în pericol de a fi prădate de fiare sălbatice, potrivit spuselor lui Isus?",
    "answers": [
      "Fuge",
      "Se gândește la nevoile oilor",
      "Încearcă să apere oile",
      "Rămâne lângă oile"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Fuge (Ioan 10:12)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.03",
    "question": "Despre Isus, Împăratul care va veni călare pe mânz, a",
    "answers": [
      "Zaharia",
      "Isaia profețit:",
      "Maleahi"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 12:14)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.03",
    "question": "„Boala aceasta nu este spre moarte, ci spre _, pentru ca Fiul lui Dumnezeu să fie proslăvit prin ea.\"",
    "answers": [
      "Viața eternă",
      "Împărăția cerurilor",
      "Slava lui Dumnezeu",
      "Mântuirea oamenilor"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Slava lui Dumnezeu (Ioan 11:4)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.03",
    "question": "Cine si cui a zis: „Dacă ai fi fost aici, n-ar fi murit fratele meu\"?",
    "answers": [
      "Maria, Domnului Isus",
      "Atât Marta, cât și Maria, Domnului Isus",
      "Marta, fratelui ei Lazăr",
      "Discipolii, Domnului Isus"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Atât Marta, cât și Maria, Domnului Isus (Ioan 11:21)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.03",
    "question": "Pe săraci îi aveți totdeauna cu voi, dar pe Mine nu, a zis",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 12:8)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.04",
    "question": "Cine a zis: vă dau pacea... nu v-o dau cum o dă lume?",
    "answers": [
      "Ioan Botezătorul",
      "Duhul Sfânt",
      "Părintele Ceresc",
      "Domnul Isus"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 14:27)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.04",
    "question": "Cine - va aduce aminte de tot ce v-am spus Eu/lsus\"?",
    "answers": [
      "Tatăl",
      "Ioan Botezătorul",
      "Mângâietorul/ Duhul Sfânt",
      "Petru"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Mângâietorul/ Duhul Sfânt."
  },
  {
    "card": "Cartea lui Ioan · Card 8.04",
    "question": "Domnul Isus a zis despre cellcei care ÎI iubesc, valvor păzi:",
    "answers": [
      "poruncile Mele",
      "cuvântul Meu",
      "căile Mele"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 114:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.04",
    "question": "Nu vă voi lăsa _, Mă voi întoarce la voi.",
    "answers": [
      "Orfani",
      "Singuri",
      "Înfricoșați",
      "Triști"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Orfani (Ioan 114:18)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.04",
    "question": "La început s-a împotrivit spălarii picioarelor iar după ce a înțeles gestul Lui Isus, a vrut să-i fie spălate și?",
    "answers": [
      "Picioarele și brațele",
      "Capul și brațele",
      "Întregul trup",
      "Mâinile și capul"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Mâinile și capul (Ioan 13:)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.04",
    "question": "Cine crede în Mine va face și el lucrările pe care le fac Eu",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 1)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.05",
    "question": "La rugămintea oamenilor, unde a rămas Isus două zite?",
    "answers": [
      "În Iudeea",
      "I. Samaria",
      "La Nazaret",
      "În Galileea"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: I. Samaria (Ioan 4:40)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.05",
    "question": "Unde a întâlnit Isus pe un om bolnav de 38 de ani?",
    "answers": [
      "La scăldătoarea Betesda",
      "La mormântul lui Lazăr",
      "La pescăriile din Galileea",
      "La poarta unei case din Ierusalim"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: La scăldătoarea Betesda."
  },
  {
    "card": "Cartea lui Ioan · Card 8.05",
    "question": "Venea din când în când, În scăldătoarea Betesda și tulbura apa, pentru vindecarea botnavilor:",
    "answers": [
      "Isus",
      "un preot",
      "un înger"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c (Ioan 5:4)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.05",
    "question": "Tocmai lucrările acestea pe care le fac Eu, mărturisesc despre Mine că _ M-a trimis.",
    "answers": [
      "Duhul Sfânt",
      "Cuvântul",
      "Tatăl",
      "Cuvântul lui Dumnezeu"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Tatăl (Ioan 5:36)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.05",
    "question": "Despre ce om, Domnul Isus a zis că este o lumină aprinsă?",
    "answers": [
      "Petru",
      "Ioan Botezătorul",
      "Natanael",
      "Filipi"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Ioan Botezătorul (Ioan 5:35)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.05",
    "question": "Tatăl înviază morții și Ie dă viață, tot așa și Fiul dă viață cui vrea.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 5:21)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.06",
    "question": "La cine s-a referit Domnul Isus, când a zis: 'Tn el nu este adevăr\"?",
    "answers": [
      "Fariseii",
      "Iuda",
      "Satana",
      "Diavol"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Diavol (Ioan 6:44)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.06",
    "question": "Cine ne poate face cu adevărat slobozi/ liberi? Când făcuse Isus tină și-i deschisese ochii orbului din",
    "answers": [
      "Tatăl",
      "Împăratul",
      "Domnul Isus/ Fiul",
      "Mântuitorul"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Domnul Isus/ Fiul."
  },
  {
    "card": "Cartea lui Ioan · Card 8.06",
    "question": "naștere era:",
    "answers": [
      "sărbătoarea Corturilor",
      "o zi normală",
      "o zi de Sabat"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este 8:36 c (Ioan 9:114)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.06",
    "question": ", pe care îl voi da „Pâinea pe care o voi da Eu este _ pentru viața lumii.”",
    "answers": [
      "Trupul Meu",
      "Sângele Meu",
      "Pâinea cerului",
      "Viața aeternă"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Trupul Meu (Ioan 6:51)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.06",
    "question": "La ce învățătură dată de Isus, unii dintre ucenici, au zis: „Vorbirea aceasta este prea de to.t: cine poate s-o sufere?”",
    "answers": [
      "Despre a se născu din nou",
      "Despre să iubești pe inamici",
      "Despre pocăință și iertare",
      "Despre a mânca trupul și sângele Lui"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Despre a mânca trupul și sângele Lui."
  },
  {
    "card": "Cartea lui Ioan · Card 8.06",
    "question": "Vine ceasul când toți cei din morminte vor auzi glasul Lui.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 5:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.07",
    "question": "Cine a zis: ”Trebuie ca El să crească, iar eu să mă micșorez\"?",
    "answers": [
      "Petru",
      "Ioan Botezătorul",
      "Pavel",
      "Andrei"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Ioan Botezătorul (Ioan 3:30)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.07",
    "question": "Cui i-a zis Isus: „Oricui bea din apa aceasta îi va fi iarăși",
    "answers": [
      "Femeii samaritence",
      "Nicodim",
      "Marta",
      "Maria Magdalena"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Femeii samaritence (Ioan 4:13)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.07",
    "question": "Este clasificat de Ioan, a' doilea semn făcut de Isus în Galileea:",
    "answers": [
      "vindecarea fiului unui slujbaș",
      "apa în vin",
      "orbul Bartimeu"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 4:54)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.07",
    "question": "Acela pe care L-a trimis Dumnezeu vorbește cuvintele lui Dumnezeu, pentru că Dumnezeu nu-l dă Duhul cu _.",
    "answers": [
      "Tărie",
      "Gracie",
      "Măsură",
      "Chip"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Măsură (Ioan 3:34)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.07",
    "question": "Cine și cui a zis: „Veniți de vedeți un Om care mi-a spus tot ce am făcut\"?",
    "answers": [
      "Marta, surorilor ei",
      "Samariteanca, oamenilor din",
      "Maria, prietenilor ei",
      "Femeya adulterei, vecinilor ei"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Samariteanca, oamenilor din."
  },
  {
    "card": "Cartea lui Ioan · Card 8.07",
    "question": "Cum M-ai trimis Tu pe Mine în lume, așa i-am trimis și Eu pe ei în tume.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 4:28)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.08",
    "question": "Cine a zis lui Isus: „Nu știi că am putere să Te răstignesc și am putere să-Ți dau drumul\"?",
    "answers": [
      "Caiafa",
      "Irod",
      "Garda romană",
      "Pilat"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Pilat (Ioan 19:)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.08",
    "question": "Ce se afla în locul numit: ”Pardosit cu pietre\"?",
    "answers": [
      "Altarul pentru jertfe",
      "Poarta ogrăzii",
      "Scaunul de judecător",
      "Masa pregătită pentru cină"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Scaunul de judecător."
  },
  {
    "card": "Cartea lui Ioan · Card 8.08",
    "question": "' - a zis „Cine Mă dă în mâinile tale are un mai mare păca„",
    "answers": [
      "Pilat",
      "Caiafa Isus lui:",
      "Ana"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este a (Ioan 19:)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.08",
    "question": "”0ricine se face pe sine _ este împotriva Cezarului.”",
    "answers": [
      "Împărat",
      "Rege",
      "Conducător",
      "Domnitor"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Împărat (Ioan 19:12)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.08",
    "question": "Cine și cui a zis: „Noi n-avem alt împărat decât pe cezarul”",
    "answers": [
      "Fariseii, lui Isus",
      "Ucenicii, lui Petru",
      "Gunoierii, lui Cezar",
      "Preoții cei mai de seamă, lui Pilat"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Preoții cei mai de seamă, lui Pilat (Ioan 1915)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.08",
    "question": "Împreună cu Isus, au fost răstigniți alți doi, unul deoparte și altul de alta, iar Isus la mijloc.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 19:18)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.09",
    "question": "La ce S-a referit Isus când a zis: ”sunt strânse, aruncate în foc și ard\"?",
    "answers": [
      "Pietrele Templului",
      "Mlădițele uscate",
      "Vesmintele păgânilor",
      "Cărțile diavolului"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Mlădițele uscate (Ioan 15:6)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.09",
    "question": "Cum se numea robul marelui preot?",
    "answers": [
      "Malhu",
      "Simeon",
      "Iacov",
      "Ioan"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Malhu (Ioan 16:10)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.09",
    "question": "Cine a zis: ”acum las lumea și Mă duc la Tatăl\"?",
    "answers": [
      "fiul risipitor",
      "Iacov",
      "Domnul Isus"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c (Ioan 16:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.09",
    "question": "„Eu am vorbit lumii pe față; totdeauna am învățat pe norod în _ și în _, unde se adună toți iudeii, și n-am spus nimic în ascuns.\"",
    "answers": [
      "Piaț, Casă",
      "Cărări, Cărări",
      "Sinagogă, Templu",
      "Grădini, Cărări"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Sinagogă, Templu (Ioan 16:20)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.09",
    "question": "Ce a spus Domnul Isus că se întâmplă cu mlădița care aduce rod?",
    "answers": [
      "O taie de pe vie",
      "O curățește ca să aducă și mai",
      "O lasă să veștejeascǎ",
      "O aruncă în iad"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: O curățește ca să aducă și mai."
  },
  {
    "card": "Cartea lui Ioan · Card 8.09",
    "question": "Trupul mort a lui Isus, l-au înfășurat în fâșii de pânză de in.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 15:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.10",
    "question": "Unde stătea mama lui Isus, când El a fost răstignit?",
    "answers": [
      "În grădina mormântului",
      "Pe muntele Golgota",
      "La poarta orașului",
      "Lângă crucea lui Isus"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Lângă crucea lui Isus."
  },
  {
    "card": "Cartea lui Ioan · Card 8.10",
    "question": "Când Isus a zis: ”Mi-e sete!\" Ce i-au dat ostașii să bea?",
    "answers": [
      "Apă",
      "Vin",
      "Oțet",
      "Lapte"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Oțet (Ioan 19:8)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.10",
    "question": "Ostașii au pus într-o ramură de... un burete plin cu oțet",
    "answers": [
      "trestie",
      "finic și t l-au dus la gură.",
      "isop"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c (Ioan 19:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.10",
    "question": "După aceea, Isus, care știa că acum totul s-a sfârșit, ca să împlinească Scriptura, a zis: „ -- - -",
    "answers": [
      "Mi-e sete!",
      "Tatăl, în mâinile Tale mă încredințez!",
      "Iată fiul tău și iată mama ta!",
      "S-a sfârșit!"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Mi-e sete! (Ioan 19:28)."
  },
  {
    "card": "Cartea lui Ioan · Card 8.10",
    "question": "Cine se afla lângă mama lui Isus când El i-a zis: \"Femeie, iată fiul tău\"?",
    "answers": [
      "Petru, Prietenul lui Isus",
      "Andrei, Fratele lui Petru",
      "Filip, Apostolul credincios",
      "Ucenicul pe care-l iubea Isus"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Ucenicul pe care-l iubea Isus."
  },
  {
    "card": "Cartea lui Ioan · Card 8.10",
    "question": "Isus a fost îngropat într-un Mormânt nou, în care nu mai fusese pus nimeni.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 19:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.01",
    "question": "Cine a zis: „despărțiți de Mine, nu puteți face nimic\"?",
    "answers": [
      "Duhul Sfânt",
      "Domnul Isus",
      "Tatăl Ceresc",
      "Ioan Botezătorul"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 15:5)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.01",
    "question": "Ce iubește lumea, potrivit celor spuse de Domnul Isus?",
    "answers": [
      "Ce este al ei",
      "Ce este rău",
      "Ce este spiritual",
      "Ce este etern"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Ce este al ei (Ioan 15:19)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.01",
    "question": "a. vă va învăț Duhul Sfânt, pe care-L va trimite Tatăl: toate lucrurile",
    "answers": [
      "vă va învăț Duhul Sfânt, pe care-L va trimite Tatăl: toate lucrurile",
      "vă va aduce aminte de tot ce a spu",
      "rămâne cu voi în veac"
    ],
    "correct": [
      0,
      1,
      2
    ],
    "feedback": "Răspunsul corect este a, b, c."
  },
  {
    "card": "Cartea lui Ioan · Card 9.01",
    "question": "Până acum n-ați cerut nimic în __: cereți și veți căpăti pentru ca bucuria voastră să fie deplină.",
    "answers": [
      "Numele Tatălui",
      "Numele Duhului Sfânt",
      "Numele Meu",
      "Numele meu"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Numele Meu (Ioan 16:24)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.01",
    "question": "În ce condiții se împlinește promisiunea: ”Cereți orice ve vrea și vi se va dai'?",
    "answers": [
      "Dacă aveți credință tare",
      "Dacă rămâneți în Mine și dacă rămân în voi",
      "Dacă sunteți curai de inimă",
      "Dacă respectați poruncile"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Dacă rămâneți în Mine și dacă rămân în voi."
  },
  {
    "card": "Cartea lui Ioan · Card 9.01",
    "question": "Domnul Isus ne-a ates din mijlocul lumii.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 15:7)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.02",
    "question": "Cum trebuie să-l cerem lui Dumnezeu orice lucru pe care vrem să-l primim?",
    "answers": [
      "În rugăciune des",
      "Cu glas tare",
      "Cu inima curată",
      "În Numele Domnului Isus"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: În Numele Domnului Isus."
  },
  {
    "card": "Cartea lui Ioan · Card 9.02",
    "question": "Cine - ”vă va învăța toate lucrurile și vă va aduce aminte de tot ce v-am spus Eu”/tsus?",
    "answers": [
      "Ioan Evanghelistul",
      "Duhul Sfânt",
      "Petru",
      "Pavel"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Duhul Sfânt (Ioan 14:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.02",
    "question": "Eu voi ruga pe Tatăl, și El vă va da un alt..., care să rămână cu voi în veac;",
    "answers": [
      "Învățător",
      "Îndrumător",
      "Mângâietor"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c (Ioan 14:16)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.02",
    "question": "Cum M-a iubit pe Mine _, așa v-am iubit și Eu pe voi.",
    "answers": [
      "Tatăl",
      "Fiul",
      "Duhul Sfânt",
      "Poporul Israel"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Tatăl (Ioan 15:)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.02",
    "question": "Isus a spus că putem aduce roadă, cu condiția - Care?",
    "answers": [
      "Să ascultăm cuvântul Lui",
      "Să ne lepădăm de păcat",
      "Să mergem la templu",
      "Să rămânem în El si El în •"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Să rămânem în El si El în • (Ioan 15:5)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.02",
    "question": "Acum voi sunteți curati din pricina Cuvântului pe care vi l-am spus.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 15:3)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.03",
    "question": "Cine a zis: Eu Mă duc să vă pregătesc un loc?",
    "answers": [
      "Petru",
      "Domnul Isus",
      "Iacov",
      "Ioan Botezătorul"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Domnul Isus (Ioan 14:2)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.03",
    "question": "Ce i s-a întâmplat lui Iuda, după ce a luat bucățica de Ii Isus?",
    "answers": [
      "A intrat Satana în el",
      "A plecat de la cină",
      "A trădat pe Isus imediat",
      "A plâns amar"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: A intrat Satana în el."
  },
  {
    "card": "Cartea lui Ioan · Card 9.03",
    "question": "Vor cunoaște toți că sunteți ucenicii Mei, dacă veți avea:",
    "answers": [
      "teamă de Domnul",
      "dragoste unii pentru alții",
      "bani"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este 13:26,27 b (Ioan 13:35)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.03",
    "question": "Isus a zis despre oile Lui: le dau viața veșnică. îr veac nu vor pieri și _ nu le va smulge din mâna Mea\".",
    "answers": [
      "Diavolul",
      "Păstorul",
      "Nimeni",
      "Vrăjmașul"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Nimeni (Ioan 10:27)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.03",
    "question": "De la cine vor primi cinste, cei ce-L slujesc pe ISUS?",
    "answers": [
      "De la Duhul Sfânt",
      "De la Tatăl",
      "De la popor",
      "De la îngerul Domnului"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: De la Tatăl (Ioan 12:26)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.03",
    "question": "Domnul Isus a venit ca săjudece lumea.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 12:47)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.04",
    "question": "Cine a zis: „Cât sunt în lume, sunt Lumina lumii\"?",
    "answers": [
      "Ioan Botezătorul",
      "Petru",
      "Filip",
      "Law Domnul Isus"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Law Domnul Isus."
  },
  {
    "card": "Cartea lui Ioan · Card 9.04",
    "question": "Câte coșuri cu firimituri au fost strânse la prima înmulțire a pâinilor?",
    "answers": [
      "Șapte",
      "Cinci",
      "Douăsprezece",
      "Nouă"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Douăsprezece (Ioan 6:13)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.04",
    "question": "Pe cine a întrebat Isus: „De unde avem să cumpărăm pâini ca să mănânce oamenii aceștia?” - doar ca să-l încerce: e",
    "answers": [
      "Petru",
      "Andrei",
      "Filip"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este Răspunsul indicat pe card."
  },
  {
    "card": "Cartea lui Ioan · Card 9.04",
    "question": "„Omul acesta nu vine de la Dumnezeu, pentru că nu ține _”",
    "answers": [
      "Sabatul",
      "Legea lui Moise",
      "Porunca predecesorilor",
      "Tradiția bătrânilor"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Sabatul (Ioan 9:16)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.04",
    "question": "Unde se afla Isus când a spus despre trupul și sângele Său care dau lumii viață?",
    "answers": [
      "Pe malul Mării Galileii",
      "La Betaniei",
      "La Ierusalim în templu",
      "În sinagoga din Capernaum"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: În sinagoga din Capernaum."
  },
  {
    "card": "Cartea lui Ioan · Card 9.04",
    "question": "Păstorul după ce și-a scos toate oile, din staul merge în urma lor.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 10:4)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.05",
    "question": "Cum se numea tatăl lui Simon Petru?",
    "answers": [
      "Zebedeu",
      "Iona",
      "Andreia",
      "Alexandru"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Iona (Ioan 21:15)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.05",
    "question": "După o noapte în care nu au pescuit nimic, cine i-ë așteptat la țărm pe ucenici, cu jăratic de cărbuni, pește pus deasupra și pâine?",
    "answers": [
      "Isus",
      "Petru",
      "Ioan",
      "Nathanael"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Isus (Ioan 21:)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.05",
    "question": "După ce Petru a confirmat că Îl iubește pe Isus, El i-a zis",
    "answers": [
      "oițele Mele",
      "mieluseii Mei paște:",
      "oile Mele"
    ],
    "correct": [
      0,
      1,
      2
    ],
    "feedback": "Răspunsul corect este a, b, c (Ioan 21:15)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.05",
    "question": "Petru s-a întristat că-i zisese a _ oară: „Mă iubești?”",
    "answers": [
      "Doua",
      "Patra",
      "Treia",
      "Cinci"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Treia (Ioan 21:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.05",
    "question": "Ce l-a întrebat Isus pe Petru, la Marea Tiberiadei, după a au prânzit?",
    "answers": [
      "Dacă crede în El",
      "Dacă Îl iubește",
      "Dacă va rămâne cu El",
      "Dacă-L va urma"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Dacă Îl iubește (Ioan 21:15)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.05",
    "question": "Tată, proslăvește-mă la Tine însuți cu slava pe care aveam la Tine, înainte de a fi lumea.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 17:5)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.06",
    "question": "Care ucenic a zis: „arată-ne pe Tatăl și ne este de ajuns\"?",
    "answers": [
      "Andrei",
      "Ioan",
      "Natanael",
      "Filip"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Filip (Ioan 14:8)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.06",
    "question": "Care profet a văzut slava lui Isus și a vorbit despre E'?",
    "answers": [
      "Ieremia",
      "Ezechiel",
      "Isaia",
      "Daniel"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Isaia (Ioan 12:41)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.06",
    "question": "„De atâta vreme sunt cu voi și nu M-ai cunoscut” - a zis Isus lui:",
    "answers": [
      "Andrei",
      "Petru",
      "Varianta c"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c (Ioan 14:)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.06",
    "question": "Robii și aprozii făcuseră un foc de _, căci era frig, și se încălzeau. Petru stătea și el cu ei și se încălzea.",
    "answers": [
      "Cărbuni",
      "Lemne",
      "Tăciuni",
      "Scânduri"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Cărbuni (Ioan 18:18)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.06",
    "question": "La afirmația lui Isus: 'Știți unde Mă duc și știți și calea într-acolo”, cine a zis: „Nu știm unde Te duci\"?",
    "answers": [
      "Filip",
      "Iuda Tadeul",
      "Bartolomeu",
      "Toma"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Toma (Ioan 14:4)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.06",
    "question": "Simon Petru i-a tăiat urechea stângă, lui Malhu.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 18:10)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.07",
    "question": "Care ucenic l-a confirmat lui Isus: „știi că Te iubesc”l?",
    "answers": [
      "Ioan",
      "Petru",
      "Iacov",
      "Andrei"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Petru (Ioan 21:16)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.07",
    "question": "De câte ori l-a întrebat Isus, pe Petru: iubesti”?",
    "answers": [
      "De 3 ori",
      "De 2 ori",
      "De 4 ori",
      "De 5 ori"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: De 3 ori (Ioan 21:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.07",
    "question": "„Doamne, dar cu acesta ce va fi\"? A întrebat Petru pe Isu* despre:",
    "answers": [
      "Toma",
      "Iuda",
      "Ioan"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este c (Ioan 21:20)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.07",
    "question": "„Doamne, Tu toate le știi, știi că",
    "answers": [
      "Te urăsc",
      "Te urmez",
      "Te iubesc",
      "Te negi"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Te iubesc (Ioan 21:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.07",
    "question": "Cine și cui a zis: „când vei îmbătrâni, îți vei întinde mâinile și altul te va încinge și te va duce unde nu vei voi\"?",
    "answers": [
      "Domnul Isus lui Ioan",
      "Domnul Isus lui Petru",
      "Domnul Isus lui Iacov",
      "Domnul Isus lui Filip"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Domnul Isus lui Petru."
  },
  {
    "card": "Cartea lui Ioan · Card 9.07",
    "question": "Și viața veșnică este aceasta: să Te cunoască pe Tine, singurul Dumnezeu adevărat, și pe Isus Hristos, pe care L-ai trimis Tu.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 17:3)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.08",
    "question": "Isus S-a rugat: „Nu Te rog să-i iei din lume, ci să-i păzești",
    "answers": [
      "Păcatul",
      "Moartea",
      "Pădurea",
      "Cel rău"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Cel rău (Ioan 17:15)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.08",
    "question": "Cum putem să avem viața în Numele Domnului Isus?",
    "answers": [
      "Iertând",
      "Ascultând",
      "Crezând",
      "Lucrând"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Crezând (Ioan 20:31)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.08",
    "question": "Pentrucă ziceți: „Vedem\". Tocmai de aceea păcatul vostru",
    "answers": [
      "preoților",
      "unor farisei rămâne - le-a zis Isus:",
      "saducheilor"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este b (Ioan 9:)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.08",
    "question": "Lucrurile acestea au fost scrise pentru ca voi să _ că Isus este Hristosul, Fiul lui Dumnezeu",
    "answers": [
      "Credeti",
      "Înțelegeți",
      "Ascultați",
      "Păziți"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Credeti (Ioan 20)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.08",
    "question": "În rugăciunea Sa, Isus a spus: ”Sfințește-i prin adevărul Tăun— Cine este adevărul?",
    "answers": [
      "'Legea Ta'",
      "'Duhul Tău'",
      "'Numele Tău'",
      "”Cuvântul Tău\""
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: ”Cuvântul Tău\" (Ioan 17:)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.08",
    "question": "Isus a zis: \"Mai am să vă spun multe lucruri, dar acum nu le puteți purta.”",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 16:12)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.09",
    "question": "Cine L-a întrebat pe Domnul Isus: „De unde mă cunoști?'",
    "answers": [
      "Filip",
      "Natanael",
      "Petru",
      "Iacov"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Natanael (Ioan 1:48)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.09",
    "question": "Cine a gustat primul din apa pe care Isus, a făcut-o vin?",
    "answers": [
      "Nunul",
      "Mărturia lui Ioan",
      "Un servitor",
      "Maria, mama lui Isus"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Nunul (Ioan 2:7)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.09",
    "question": "În cetatea... din ținutul Samariei, se afla fântâna lui laco'u",
    "answers": [
      "Sidon",
      "Sihar",
      "Siloam"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este b (Ioan 4:5)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.09",
    "question": "În Templu a găsit pe cei ce vindeau _ și și schimbătorii de bani șezând jos.",
    "answers": [
      "Peștii și păsările",
      "Mieii și vitele",
      "Boi, oi și porumbei",
      "Pasul și cămilele"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Boi, oi și porumbei (Ioan 2:14)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.09",
    "question": "Ce a răspuns Filip, la întrebarea ironică a lui Natanae „Poate ieși ceva bun din Nazar*'?",
    "answers": [
      "Nu pot crede!",
      "Vino și vezi!",
      "E mult prea departe!",
      "Rămâi aici cu noi!"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Vino și vezi! (Ioan 1:46)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.09",
    "question": "Ogorul pe care l-a dat Iacov, fiului său Iosif, se afla tinutul Samariei.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 4:5)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.10",
    "question": "Care este - porunca nouă - despre care a vorbit Isus ucenicilor Săi?",
    "answers": [
      "Să vă iertați unii pe alții",
      "Să vă slujiti unii pe alții",
      "Să vă mărturisiți unii altuia",
      "Să vă iubiți unii pe alții"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Să vă iubiți unii pe alții."
  },
  {
    "card": "Cartea lui Ioan · Card 9.10",
    "question": "Cu ce scop a venit Domnul Isus în lumea noastră?",
    "answers": [
      "Ca să condamne lumea",
      "Ca să ne mântuiască",
      "Ca să dobândească împărăția",
      "Ca să-și împlinească voia"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Ca să ne mântuiască (Ioan 3:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.10",
    "question": "Când Isus i-a zis lui Iuda: „Ce ai să faci, fă repede\", unii ucenici credeau că:",
    "answers": [
      "l-a trimis să cumpere ce trebuie pentru praznic",
      "să dea ceva săracilor",
      "Varianta c"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este Ioan 12:47 a, b (Ioan 13:27)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.10",
    "question": "e 'Eu sunt Ușa. Dacă intră cineva prin _, va fi mântuit; va intra și va ieși și va găsi pășune.”",
    "answers": [
      "Mine",
      "Ușă",
      "Purtă",
      "Gard"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Mine (Ioan 10:)."
  },
  {
    "card": "Cartea lui Ioan · Card 9.10",
    "question": "\"Ca să ajungeți să cunoașteți și să știți că Tatăl este în Mine și Eu sunt în Tatăl\", ce a zis Domnul Isus să facem?",
    "answers": [
      "Să ne rugăm zilnic",
      "Să păzim poruncile Lui",
      "Să prevestim cuvintele Lui",
      "Să credem lucrările Lui"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Să credem lucrările Lui."
  },
  {
    "card": "Cartea lui Ioan · Card 9.10",
    "question": "Credeți în Lumină, ca să fiți fii ai luminii - a zis Domnul Isus.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 1,
    "feedback": "Afirmația este falsă (Ioan 12:36)."
  },
  {
    "card": "Cartea lui Ioan · Card 10.01",
    "question": "Ce făcea Isus, de turbau iudeii de mânieîmpotriva Lui?",
    "answers": [
      "A vorbit cu femeia samarineană la fântâna lui Iacov",
      "A însănătoșit un om în ziua Sabatului",
      "A întors apa în vin la nunta din Cana",
      "A înviat pe Lazăr din morți"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: A însănătoșit un om în ziua Sabatului (Ioan 7:23)."
  },
  {
    "card": "Cartea lui Ioan · Card 10.01",
    "question": "Cine și cui a zis: „Doar n-ați fi fost duși și voi în rătăcire\"?",
    "answers": [
      "Fariseii, aprozilor",
      "Iudeii, discipolilor",
      "Oamenii din mulțime, fariseilor",
      "Întocmai ai săi, întregului popor"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Fariseii, aprozilor."
  },
  {
    "card": "Cartea lui Ioan · Card 10.01",
    "question": "Isus le-a zis: „Dacă ar fi Dumnezeu Tatăl vostru, M-ați... și pe Mine, căci Eu vin de la Dumnezeu\".",
    "answers": [
      "cinsti",
      "asculta",
      "Varianta c"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este 7:45-147 c (Ioan 8:42)."
  },
  {
    "card": "Cartea lui Ioan · Card 10.01",
    "question": "Isus i-a zis: „Nici Eu nu te osândesc. Du-te și să nu mai",
    "answers": [
      "Minți",
      "Furi",
      "Păcătuiești",
      "Omori"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Păcătuiești (Ioan 8:11)."
  },
  {
    "card": "Cartea lui Ioan · Card 10.01",
    "question": "Cine au fost trimiși să-L prindă pe Isus, dar n-au făcut-o iar când s-au întors au zis: „Niciodată n-a vorbit vreun om ca Omul acesta\"?",
    "answers": [
      "Fariseii",
      "Aprozii",
      "Oamenii din mulțime",
      "Scribii"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Aprozii (Ioan 7:)."
  },
  {
    "card": "Cartea lui Ioan · Card 10.01",
    "question": "Avraam a săltat de bucurie că are -să vadă ziua Mea - a zis Isus.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 8:56)."
  },
  {
    "card": "Cartea lui Ioan · Card 10.02",
    "question": "Cine a fost trimis ca martor, ca să mărturisescă despre Lumină?",
    "answers": [
      "Andrei",
      "Natan",
      "Filip",
      "Ioan"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Ioan (Ioan 1:6)."
  },
  {
    "card": "Cartea lui Ioan · Card 10.02",
    "question": "Ce drept li s-a dat celor ce-L primesc pe Domnul Isus și cred în Numele Lui?",
    "answers": [
      "Să moștenească împărăția lui Dumnezeu",
      "Să fie mântuiți de păcate",
      "Să se facă copii ai lui",
      "Să poarte troita Lui"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Să se facă copii ai lui."
  },
  {
    "card": "Cartea lui Ioan · Card 10.02",
    "question": "”Cel ce vine după mine este înaintea mea, pentru că era înainte de mine.” - a zis:",
    "answers": [
      "Moise",
      "Isus",
      "Ioan"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este Dumnezeu (Ioan 1:12)."
  },
  {
    "card": "Cartea lui Ioan · Card 10.02",
    "question": "\"Toate lucrurile au fost făcute __; și nimic din ce a fost făcut, n-a fost făcut __",
    "answers": [
      "Prin El, fără El",
      "De Dumnezeu, cu Dumnezeu",
      "Cu cuvântul, de cuvânt",
      "În Dumnezeu, împotriva Lui"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Prin El, fără El (Ioan 1:3)."
  },
  {
    "card": "Cartea lui Ioan · Card 10.02",
    "question": "Prin cine au venit harul și adevărul, la noi oamenii?",
    "answers": [
      "Duhul Sfânt și Tatăl",
      "Ioan Botezătorul",
      "Moise și Ilie",
      "Isus Hristos"
    ],
    "correct": 3,
    "feedback": "Răspunsul corect este: Isus Hristos (Ioan 1:17)."
  },
  {
    "card": "Cartea lui Ioan · Card 10.02",
    "question": "Cuvântul era Dumnezeu.",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 1:1)."
  },
  {
    "card": "Cartea lui Ioan · Card 10.03",
    "question": "Cum se numea tatăl lui Iuda, ucenicul care L-a vândut pe Isus?",
    "answers": [
      "Iuda din Galileea",
      "Simon Iscarioteanul",
      "Iuda fiul lui Iacov",
      "Iuda fiul lui Alfeu"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: Simon Iscarioteanul (Ioan 6:71)."
  },
  {
    "card": "Cartea lui Ioan · Card 10.03",
    "question": "Cui a zis Isus: „Vremea Mea n-a sosit încă, dar vouă vremea totdeauna vă este prielnică\"?",
    "answers": [
      "Fraților Lui",
      "Discipolilor Săi",
      "Fariseilor",
      "Oamenilor din Ierusalim"
    ],
    "correct": 0,
    "feedback": "Răspunsul corect este: Fraților Lui (Ioan 7:3)."
  },
  {
    "card": "Cartea lui Ioan · Card 10.03",
    "question": "Când Isus a zis: „mai înainte ca să se nască Avraam, sunt Eu\" iudeii:",
    "answers": [
      "L-au scuipat",
      "au luat pietre să arunce în El",
      "L-au părăsit"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este b (Ioan 8:58)."
  },
  {
    "card": "Cartea lui Ioan · Card 10.03",
    "question": "Când au auzit ei cuvintele acestea, s-au simțit _ de cugetul lor și au ieșit afară, unul câte unul, începând de la cei mai bătrâni.",
    "answers": [
      "Surprinși Ioan",
      "Înspăimântați Ioan",
      "Mustrați Ioan",
      "Condamnați Ioan"
    ],
    "correct": 2,
    "feedback": "Răspunsul corect este: Mustrați Ioan."
  },
  {
    "card": "Cartea lui Ioan · Card 10.03",
    "question": "De la cine era venită porunca tăierii împrejur?",
    "answers": [
      "De la profeți",
      "De la patriarhi",
      "De la moise",
      "De la preoți"
    ],
    "correct": 1,
    "feedback": "Răspunsul corect este: De la patriarhi (Ioan 7:22)."
  },
  {
    "card": "Cartea lui Ioan · Card 10.03",
    "question": "Isus a zis: „Eu nu sunt din lumea aceasta\".",
    "answers": [
      "Adevărat",
      "Fals"
    ],
    "correct": 0,
    "feedback": "Afirmația este adevărată (Ioan 8:23)."
  }
];
