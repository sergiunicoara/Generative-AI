/* Fundaluri animate — scene biblice drăguțe pentru copii de 10 ani.
   Fiecare scenă: init(w,h)->stare, draw(ctx,w,h,age,stare). */

/* ── utilitare ─────────────────────────────────────────────── */
function rnd(a, b) { return a + Math.random() * (b - a); }
function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

function skyGrad(ctx, w, h, stops) {
  const g = ctx.createLinearGradient(0, 0, 0, h);
  stops.forEach(([p, c]) => g.addColorStop(p, c));
  ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
}

function star5(ctx, x, y, r, col, alpha) {
  ctx.save(); ctx.globalAlpha = alpha; ctx.fillStyle = col; ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const ang = i * Math.PI / 5 - Math.PI / 2, rad = i % 2 === 0 ? r : r * 0.42;
    i === 0 ? ctx.moveTo(x + rad * Math.cos(ang), y + rad * Math.sin(ang))
            : ctx.lineTo(x + rad * Math.cos(ang), y + rad * Math.sin(ang));
  }
  ctx.closePath(); ctx.fill(); ctx.restore();
}

/* Soare zâmbitor cu raze (stil Talant) */
function smileSun(ctx, cx, cy, r, t) {
  ctx.save();
  for (let i = 0; i < 12; i++) {
    const a = i / 12 * Math.PI * 2 + t * 0.4;
    const r1 = r * 1.25, r2 = r * 1.78 + Math.sin(t * 2 + i) * r * 0.07;
    ctx.strokeStyle = '#FFD700'; ctx.lineWidth = r * 0.1; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(cx + r1 * Math.cos(a), cy + r1 * Math.sin(a));
    ctx.lineTo(cx + r2 * Math.cos(a), cy + r2 * Math.sin(a)); ctx.stroke();
  }
  ctx.fillStyle = '#FFE840'; ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = 'rgba(255,255,255,.28)';
  ctx.beginPath(); ctx.ellipse(cx - r * .22, cy - r * .25, r * .36, r * .2, -.5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#7a4500';
  [-.28, .28].forEach(dx => { ctx.beginPath(); ctx.arc(cx + dx * r, cy - r * .08, r * .09, 0, Math.PI * 2); ctx.fill(); });
  ctx.strokeStyle = '#7a4500'; ctx.lineWidth = r * .09; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.arc(cx, cy + r * .12, r * .28, .15, Math.PI - .15); ctx.stroke();
  ctx.restore();
}

/* Nor pufos cu 6 cercuri (stil Talant) */
function fluffyCloud(ctx, cx, cy, cw, ch, col, alpha) {
  ctx.save(); ctx.globalAlpha = alpha; ctx.fillStyle = col;
  [[0,0,.55],[-.28,.12,.38],[.28,.12,.38],[-.48,.28,.28],[.48,.28,.28],[0,.32,.32]]
    .forEach(([dx, dy, fr]) => {
      ctx.beginPath(); ctx.arc(cx + dx * cw, cy + dy * ch * 2, fr * ch * 2, 0, Math.PI * 2); ctx.fill();
    });
  ctx.restore();
}

function hillBand(ctx, w, h, baseY, amp, freq, phase, col) {
  ctx.fillStyle = col; ctx.beginPath();
  for (let x = 0; x <= w; x += 8) {
    const y = baseY + amp * Math.sin(x / w * Math.PI * freq + phase);
    x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath(); ctx.fill();
}

function flame(ctx, x, y, s, t, seed = 0) {
  const f = 1 + .18 * Math.sin(t * 9 + seed) + .1 * Math.sin(t * 23 + seed * 2);
  [[s,'#FF7043',.9],[s*.66,'#FFA726',.95],[s*.38,'#FFEE58',1]].forEach(([r,c,a]) => {
    ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = c; ctx.beginPath();
    ctx.ellipse(x + Math.sin(t * 7 + seed) * s * .08, y - r * .5, r * .5 * f, r * f, 0, 0, Math.PI * 2);
    ctx.fill(); ctx.restore();
  });
}

function smokePuffs(ctx, x, y, s, age, n = 3) {
  for (let i = 0; i < n; i++) {
    const p = ((age + i * 1100) % 3300) / 3300;
    ctx.save(); ctx.globalAlpha = .35 * (1 - p); ctx.fillStyle = '#B0BEC5';
    ctx.beginPath(); ctx.ellipse(x + Math.sin(p * 5 + i) * s * .4, y - p * s * 3.2, s * (.3 + p * .7), s * (.22 + p * .5), 0, 0, Math.PI * 2);
    ctx.fill(); ctx.restore();
  }
}

function figure(ctx, x, y, s, robe, skin = '#FFCC80') {
  ctx.fillStyle = skin; ctx.beginPath(); ctx.arc(x, y - s * .85, s * .18, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = robe; ctx.beginPath();
  ctx.moveTo(x - s * .3, y); ctx.quadraticCurveTo(x - s * .32, y - s * .72, x, y - s * .7);
  ctx.quadraticCurveTo(x + s * .32, y - s * .72, x + s * .3, y); ctx.closePath(); ctx.fill();
}

function bird(ctx, x, y, s, col = 'rgba(60,60,80,.7)') {
  ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = Math.max(1.4, s * .12); ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(x - s, y); ctx.quadraticCurveTo(x - s * .4, y - s * .7, x, y);
  ctx.quadraticCurveTo(x + s * .4, y - s * .7, x + s, y); ctx.stroke(); ctx.restore();
}

function banner(ctx, x, y, len, t, col) {
  ctx.strokeStyle = '#6D4C41'; ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y - len); ctx.stroke();
  ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(x, y - len);
  const sway = Math.sin(t * 2.4 + x) * len * .1;
  ctx.quadraticCurveTo(x + len * .3, y - len + sway, x + len * .55, y - len + len * .1 + sway);
  ctx.lineTo(x, y - len + len * .24); ctx.closePath(); ctx.fill();
}

/* Oaie drăguță și pufoasă */
function sheep(ctx, x, y, s, t, ph) {
  const bob = Math.sin(t * 2 + ph) * s * .05;
  ctx.fillStyle = '#FAFAFA';
  [[0,0,1],[-.5,-.3,.7],[.5,-.3,.7],[-.8,.1,.55],[.8,.1,.55]]
    .forEach(([dx,dy,r]) => { ctx.beginPath(); ctx.arc(x + dx*s, y + dy*s + bob, s*r*.65, 0, Math.PI*2); ctx.fill(); });
  ctx.fillStyle = '#6D4C41'; ctx.beginPath(); ctx.arc(x - s, y - s*.2 + bob, s*.42, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(x - s - s*.12, y - s*.3 + bob, s*.12, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = '#222'; ctx.beginPath(); ctx.arc(x - s - s*.1, y - s*.28 + bob, s*.06, 0, Math.PI*2); ctx.fill();
  ctx.strokeStyle = '#6D4C41'; ctx.lineWidth = s*.22; ctx.lineCap = 'round';
  [-.45,-.15,.18,.48].forEach(dx => {
    ctx.beginPath(); ctx.moveTo(x + dx*s*1.2, y + s*.5 + bob); ctx.lineTo(x + dx*s*1.2, y + s + bob); ctx.stroke();
  });
}

/* Floare colorată pe tulpină */
function flower(ctx, cx, by, sz, col) {
  ctx.strokeStyle = '#4a8a20'; ctx.lineWidth = sz*.1; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(cx, by); ctx.lineTo(cx, by - sz*.9); ctx.stroke();
  ctx.fillStyle = col;
  for (let i = 0; i < 6; i++) {
    const a = i / 6 * Math.PI * 2;
    ctx.beginPath(); ctx.ellipse(cx + Math.cos(a)*sz*.32, by - sz*.9 + Math.sin(a)*sz*.32, sz*.2, sz*.12, a, 0, Math.PI*2); ctx.fill();
  }
  ctx.fillStyle = '#FFE840'; ctx.beginPath(); ctx.arc(cx, by - sz*.9, sz*.16, 0, Math.PI*2); ctx.fill();
}

/* Fluture cu aripi bătând */
function butterfly(ctx, cx, cy, sz, col, t, ph) {
  ctx.save(); ctx.globalAlpha = .88;
  const flap = Math.abs(Math.sin(t * 8 + ph));
  [1, -1].forEach(s => {
    ctx.fillStyle = col; ctx.beginPath();
    ctx.ellipse(cx + s*flap*sz*.55, cy - sz*.28, flap*sz*.6, sz*.45, s*.5, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.ellipse(cx + s*flap*sz*.42, cy + sz*.15, flap*sz*.45, sz*.3, -s*.4, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,.45)'; ctx.beginPath(); ctx.arc(cx + s*flap*sz*.45, cy - sz*.3, sz*.1, 0, Math.PI*2); ctx.fill();
  });
  ctx.fillStyle = '#333'; ctx.globalAlpha = 1;
  ctx.beginPath(); ctx.ellipse(cx, cy, sz*.07, sz*.48, 0, 0, Math.PI*2); ctx.fill();
  ctx.restore();
}

/* Copac rotund drăguț */
function roundTree(ctx, x, by, sz, tc, lc) {
  ctx.fillStyle = tc; ctx.fillRect(x - sz*.08, by - sz*.22, sz*.16, sz*.22);
  ctx.fillStyle = lc; ctx.beginPath(); ctx.arc(x, by - sz*.52, sz*.45, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = 'rgba(255,255,255,.2)';
  ctx.beginPath(); ctx.ellipse(x - sz*.1, by - sz*.62, sz*.2, sz*.14, -.5, 0, Math.PI*2); ctx.fill();
}

/* Brad */
function pineTree(ctx, x, by, sz) {
  ctx.fillStyle = '#5D4037'; ctx.fillRect(x - sz*.06, by - sz*.2, sz*.12, sz*.2);
  [[0,3,.7],[1,2,.55],[2,1,.4]].forEach(([i]) => {
    const lw = sz*(.7 - i*.15), ly = by - sz*(.35 + i*.25);
    ctx.fillStyle = i===0?'#2E7D32':i===1?'#388E3C':'#43A047';
    ctx.beginPath(); ctx.moveTo(x, by - sz*(.55+i*.25)); ctx.lineTo(x-lw/2,ly); ctx.lineTo(x+lw/2,ly); ctx.closePath(); ctx.fill();
  });
}

/* Curcubeu */
function rainbow(ctx, cx, cy, rad, alpha) {
  const cols = ['#FF0000','#FF7700','#FFEE00','#00CC00','#0088FF','#9900CC'];
  ctx.save(); ctx.globalAlpha = alpha;
  cols.forEach((c,i) => {
    ctx.strokeStyle = c; ctx.lineWidth = rad*.045;
    ctx.beginPath(); ctx.arc(cx, cy, rad*(1-i*.05), Math.PI, 0); ctx.stroke();
  });
  ctx.restore();
}

/* Palmier cu cocos */
function palmTree(ctx, x, y, s) {
  ctx.strokeStyle = '#8D6E63'; ctx.lineWidth = s*.12; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(x,y); ctx.quadraticCurveTo(x+s*.18,y-s*.6,x+s*.08,y-s*1.1); ctx.stroke();
  ctx.strokeStyle = '#43A047'; ctx.lineWidth = s*.09;
  for (let i = 0; i < 6; i++) {
    const ang = -Math.PI/2 + (i-2.5)*.45;
    ctx.beginPath(); ctx.moveTo(x+s*.08,y-s*1.1);
    ctx.quadraticCurveTo(x+s*.08+Math.cos(ang)*s*.5,y-s*1.1+Math.sin(ang)*s*.5-s*.1,x+s*.08+Math.cos(ang)*s*.85,y-s*1.1+Math.sin(ang)*s*.55);
    ctx.stroke();
  }
  ctx.fillStyle = '#FF8F00';
  for (let i = 0; i < 3; i++) {
    const a = -Math.PI/2+(i-1)*.35;
    ctx.beginPath(); ctx.arc(x+s*.08+Math.cos(a)*s*.2,y-s*1.05+Math.sin(a)*s*.2,s*.09,0,Math.PI*2); ctx.fill();
  }
}

/* ── SCENE ──────────────────────────────────────────────────── */
const SCENES = [
  /* 1. TEMPLU */
  {
    id: 'templu',
    keywords: ['altar','jertfă','jertfele','jertfa','templu','chivot','preot','preoți',
      'arderi','sfânt','slujbă','Silo','Casa Domnului','tămâie','efod'],
    init(w, h) {
      return {
        clouds: Array.from({length:3},()=>({x:rnd(0,1),y:rnd(.04,.18),cw:rnd(60,130),ch:rnd(28,55),sp:rnd(.1,.3)})),
        embers: Array.from({length:14},()=>({x:rnd(-30,30),period:rnd(2200,4200),off:rnd(0,5000),r:rnd(1.5,3.5)})),
        flowers: Array.from({length:8},()=>({x:rnd(0,1),sz:rnd(12,20),col:pick(['#FF6B9D','#FFCA28','#FF7043','#7C4DFF','#29B6F6'])})),
      };
    },
    draw(ctx, w, h, age, s) {
      const t = age / 1000;
      skyGrad(ctx, w, h, [[0,'#FFF9C4'],[.4,'#FFE082'],[.7,'#FFCC80'],[1,'#FFB74D']]);
      smileSun(ctx, w*.15, h*.14, 36, t);
      s.clouds.forEach(c => {
        c.x += .0002*c.sp*60; if(c.x>1.3) c.x=-0.3;
        fluffyCloud(ctx, w*c.x, h*c.y, c.cw, c.ch, '#fff', .88);
      });
      // deal verde
      hillBand(ctx, w, h, h*.72, h*.03, 2.2, 0, '#A5D6A7');
      // podeaua
      ctx.fillStyle='#D7CCC8'; ctx.fillRect(0,h*.82,w,h*.18);
      ctx.fillStyle='#BCAAA4'; ctx.fillRect(w*.08,h*.79,w*.84,h*.035);
      ctx.fillStyle='#A1887F'; ctx.fillRect(w*.13,h*.755,w*.74,h*.035);
      // templu
      const tx=w/2, ty=h*.755, tw=Math.min(w*.62,380), th=h*.38;
      ctx.fillStyle='#FFF9C4'; ctx.fillRect(tx-tw/2,ty-th,tw,th);
      // coloane colorate
      const colColors=['#FFD54F','#FF8A65','#CE93D8','#80DEEA','#A5D6A7','#FFD54F'];
      for(let i=0;i<6;i++){
        const cx2=tx-tw/2+tw*(.08+i*.168);
        ctx.fillStyle=colColors[i]; ctx.fillRect(cx2,ty-th+th*.18,tw*.07,th*.82);
        ctx.fillStyle='rgba(255,255,255,.4)'; ctx.fillRect(cx2+2,ty-th+th*.18,tw*.02,th*.82);
      }
      // fronton
      ctx.fillStyle='#FFB74D'; ctx.beginPath();
      ctx.moveTo(tx-tw/2-tw*.06,ty-th); ctx.lineTo(tx,ty-th-h*.12); ctx.lineTo(tx+tw/2+tw*.06,ty-th); ctx.closePath(); ctx.fill();
      star5(ctx,tx,ty-th-h*.06,10,'#FFD600',1);
      // altar
      ctx.fillStyle='#8D6E63'; ctx.fillRect(tx-46,h*.88-36,92,36);
      ctx.fillStyle='#6D4C41'; ctx.fillRect(tx-56,h*.88-8,112,10);
      flame(ctx,tx,h*.88-36,36,t);
      smokePuffs(ctx,tx,h*.88-78,16,age);
      s.embers.forEach(e=>{
        const p=((age+e.off)%e.period)/e.period;
        ctx.save(); ctx.globalAlpha=(1-p)*.8; ctx.fillStyle='#FFB300';
        ctx.beginPath(); ctx.arc(tx+e.x+Math.sin(p*8)*10,h*.88-40-p*130,e.r*(1-p*.5),0,Math.PI*2); ctx.fill(); ctx.restore();
      });
      // flori în față
      s.flowers.forEach((f,i)=>flower(ctx,w*(0.04+i*.13),h*.84,f.sz,f.col));
    },
  },

  /* 2. RĂZBOI */
  {
    id: 'razboi',
    keywords: ['război','luptă','sabie','suliță','oaste','bătălie','arme','vitejii',
      'filisteni','oștire','înfrângere','biruit','tabără'],
    init(w, h) {
      return {
        swords: Array.from({length:6},(_,i)=>({x:.1+i*.15,ph:rnd(0,6),sp:rnd(.8,1.6)})),
        sparks: Array.from({length:12},()=>({x:rnd(.3,.7),period:rnd(800,1600),off:rnd(0,3000),col:pick(['#FFD54F','#FF7043','#FFEE58'])})),
        shields: [{x:.18,col:'#E53935'},{x:.38,col:'#1E88E5'},{x:.62,col:'#E53935'},{x:.82,col:'#1E88E5'}],
        birds: Array.from({length:3},()=>({x:rnd(0,1),y:rnd(.06,.2),sp:rnd(.5,1)})),
      };
    },
    draw(ctx, w, h, age, s) {
      const t = age / 1000;
      skyGrad(ctx, w, h, [[0,'#FF8F00'],[.4,'#FFA726'],[.7,'#FFB74D'],[1,'#FFCC80']]);
      smileSun(ctx, w*.5, h*.22, 30, t);
      s.birds.forEach(b=>{ b.x+=.001*b.sp; if(b.x>1.1)b.x=-0.1; bird(ctx,w*b.x,h*b.y+Math.sin(t*2+b.x*9)*5,10,'rgba(80,40,20,.7)'); });
      hillBand(ctx,w,h,h*.6,h*.04,3,0,'#A5D6A7');
      hillBand(ctx,w,h,h*.72,h*.03,4,2,'#81C784');
      // scuturi rotunde drăguțe
      s.shields.forEach(sh=>{
        const sx=w*sh.x, sy=h*.74, r=28+Math.sin(t*1.5+sh.x*10)*3;
        ctx.fillStyle=sh.col; ctx.beginPath(); ctx.arc(sx,sy,r,0,Math.PI*2); ctx.fill();
        ctx.strokeStyle='#FFD700'; ctx.lineWidth=4; ctx.beginPath(); ctx.arc(sx,sy,r,0,Math.PI*2); ctx.stroke();
        star5(ctx,sx,sy,10,'#FFD700',.9);
        // sulița / sabia
        ctx.strokeStyle='#90A4AE'; ctx.lineWidth=5; ctx.lineCap='round';
        ctx.beginPath(); ctx.moveTo(sx,sy-r); ctx.lineTo(sx+Math.sin(t*.9+sh.x*7)*8,sy-r-60); ctx.stroke();
        ctx.fillStyle='#CFD8DC'; ctx.beginPath();
        ctx.moveTo(sx+Math.sin(t*.9+sh.x*7)*8,sy-r-60);
        ctx.lineTo(sx+Math.sin(t*.9+sh.x*7)*8-6,sy-r-46);
        ctx.lineTo(sx+Math.sin(t*.9+sh.x*7)*8+6,sy-r-46); ctx.closePath(); ctx.fill();
      });
      // scântei de ciocnire în centru
      s.sparks.forEach(sp=>{
        const p=((age+sp.off)%sp.period)/sp.period;
        star5(ctx,w*sp.x+Math.sin(p*10)*18,h*.55-p*40,5+p*5,sp.col,(1-p)*.9);
      });
      banner(ctx,w*.06,h*.72,80,t,'#E53935');
      banner(ctx,w*.94,h*.72,80,t+1,'#1E88E5');
      ctx.fillStyle='#6D4C41'; ctx.fillRect(0,h*.86,w,h*.14);
    },
  },

  /* 3. ÎMPĂRAT */
  {
    id: 'imparat',
    keywords: ['împărat','împăratul','tron','palat','domnie','coroană','sceptru','uns','rege','împărăție'],
    init() {
      return { sparks: Array.from({length:20},()=>({x:rnd(.05,.95),y:rnd(.05,.7),sp:rnd(.8,2.4),ph:rnd(0,6),r:rnd(2,5)})) };
    },
    draw(ctx, w, h, age, s) {
      const t = age / 1000;
      skyGrad(ctx,w,h,[[0,'#7B1FA2'],[.5,'#9C27B0'],[.85,'#CE93D8'],[1,'#F8BBD0']]);
      // stele strălucitoare
      s.sparks.forEach(sp=>star5(ctx,w*sp.x,h*sp.y,sp.r,'#FFE082',.3+.65*(0.5+.5*Math.sin(t*sp.sp+sp.ph))));
      // raze aurii
      for(let i=0;i<10;i++){
        const ang=-Math.PI/2+(i-4.5)*.22+Math.sin(t*.4)*.03;
        ctx.save(); ctx.globalAlpha=.18+.1*Math.sin(t*1.2+i);
        ctx.strokeStyle='#FFD54F'; ctx.lineWidth=12;
        ctx.beginPath(); ctx.moveTo(w/2,h*.42); ctx.lineTo(w/2+Math.cos(ang)*h*.6,h*.42+Math.sin(ang)*h*.6); ctx.stroke(); ctx.restore();
      }
      // covor roșu cu stele
      ctx.fillStyle='#C62828'; ctx.beginPath();
      ctx.moveTo(w*.36,h*.62); ctx.lineTo(w*.64,h*.62); ctx.lineTo(w*.82,h); ctx.lineTo(w*.18,h); ctx.closePath(); ctx.fill();
      ctx.fillStyle='#FFD54F'; ctx.fillRect(w*.18,h*.98,w*.64,h*.02);
      for(let i=0;i<8;i++) star5(ctx,w*(.22+i*.08),h*.85,5,'#FFD54F',.7);
      // trepte
      ctx.fillStyle='#8E24AA'; ctx.fillRect(w*.28,h*.66,w*.44,h*.05);
      ctx.fillStyle='#7B1FA2'; ctx.fillRect(w*.32,h*.62,w*.36,h*.05);
      // tron auriu
      const cx=w/2;
      ctx.fillStyle='#F9A825'; ctx.fillRect(cx-64,h*.38,128,h*.24);
      ctx.fillStyle='#FBC02D'; ctx.fillRect(cx-80,h*.5,22,h*.13); ctx.fillRect(cx+58,h*.5,22,h*.13);
      ctx.beginPath(); ctx.arc(cx,h*.38,64,Math.PI,0); ctx.fill();
      ctx.fillStyle='#FFD600'; ctx.beginPath(); ctx.arc(cx,h*.38,24,0,Math.PI*2); ctx.fill();
      star5(ctx,cx,h*.38,14,'#FFF9C4',1);
      // împăratul
      figure(ctx,cx,h*.62,88,'#6A1B9A','#FFCC80');
      // coroană mare
      const crY=h*.62-88*.85-14;
      ctx.fillStyle='#FFD600'; ctx.beginPath();
      ctx.moveTo(cx-20,crY+12); ctx.lineTo(cx-20,crY-2); ctx.lineTo(cx-10,crY+8);
      ctx.lineTo(cx,crY-8); ctx.lineTo(cx+10,crY+8); ctx.lineTo(cx+20,crY-2); ctx.lineTo(cx+20,crY+12); ctx.closePath(); ctx.fill();
      [cx-20,cx,cx+20].forEach(x=>{ ctx.fillStyle='#F44336'; ctx.beginPath(); ctx.arc(x,crY+(x===cx?-8:x<cx?-2:-2),4,0,Math.PI*2); ctx.fill(); });
      // sceptru
      ctx.strokeStyle='#FFD600'; ctx.lineWidth=5; ctx.lineCap='round';
      ctx.beginPath(); ctx.moveTo(cx+32,h*.52); ctx.lineTo(cx+48,h*.62); ctx.stroke();
      star5(ctx,cx+32,h*.52,8,'#FFF176',1);
      // torțe colorate
      [[w*.12,0],[w*.88,3]].forEach(([tx2,seed])=>{ ctx.fillStyle='#5D4037'; ctx.fillRect(tx2-5,h*.5,10,h*.16); flame(ctx,tx2,h*.5,24,t,seed); });
    },
  },

  /* 4. PUSTIE */
  {
    id: 'pustie',
    keywords: ['pustie','pustiul','cort','nisip','tabăra','drum','călătorie','Zif','câmpii'],
    init(w, h) {
      return {
        sand: Array.from({length:10},()=>({x:rnd(0,1),y:rnd(.72,.95),sp:rnd(.3,.9),ph:rnd(0,6)})),
        camelOff: rnd(0,20000),
        clouds: Array.from({length:2},()=>({x:rnd(0,1),y:rnd(.04,.14),cw:rnd(55,100),ch:rnd(25,45),sp:rnd(.1,.25)})),
      };
    },
    draw(ctx, w, h, age, s) {
      const t = age / 1000;
      skyGrad(ctx,w,h,[[0,'#4FC3F7'],[.4,'#81D4FA'],[.65,'#FFE0B2'],[1,'#FFCC80']]);
      smileSun(ctx,w*.78,h*.14,38,t);
      s.clouds.forEach(c=>{ c.x+=.0002*c.sp*60; if(c.x>1.3)c.x=-0.3; fluffyCloud(ctx,w*c.x,h*c.y,c.cw,c.ch,'#fff',.85); });
      // dune colorate
      hillBand(ctx,w,h,h*.6,h*.04,2.5,.5,'#FFD54F');
      hillBand(ctx,w,h,h*.7,h*.045,3.2,2.2,'#FFCA28');
      hillBand(ctx,w,h,h*.8,h*.05,2.2,4.1,'#FFA726');
      // palmieri
      palmTree(ctx,w*.1,h*.76,72);
      palmTree(ctx,w*.88,h*.8,55);
      // corturi colorate
      const tent=(tx,ty,ts,c1,c2)=>{
        ctx.fillStyle=c1; ctx.beginPath(); ctx.moveTo(tx,ty-ts); ctx.lineTo(tx-ts*1.1,ty); ctx.lineTo(tx+ts*1.1,ty); ctx.closePath(); ctx.fill();
        ctx.fillStyle=c2; ctx.beginPath(); ctx.moveTo(tx,ty-ts); ctx.lineTo(tx-ts*.25,ty); ctx.lineTo(tx+ts*.25,ty); ctx.closePath(); ctx.fill();
        // model geometric simplu
        ctx.strokeStyle='rgba(255,255,255,.5)'; ctx.lineWidth=2;
        ctx.beginPath(); ctx.moveTo(tx-ts*.6,ty-ts*.3); ctx.lineTo(tx+ts*.6,ty-ts*.3); ctx.stroke();
      };
      tent(w*.32,h*.8,62,'#EF5350','#B71C1C');
      tent(w*.62,h*.76,46,'#FF8F00','#E65100');
      // cămilă cartoon
      const cp=((age+s.camelOff)%28000)/28000;
      const camX=-70+(w+140)*cp, camY=h*.85+Math.sin(cp*40)*2;
      ctx.fillStyle='#A1887F';
      ctx.beginPath(); ctx.ellipse(camX,camY,30,16,0,0,Math.PI*2); ctx.fill();
      // cocoașe
      ctx.beginPath(); ctx.arc(camX-8,camY-16,10,0,Math.PI*2); ctx.fill();
      ctx.beginPath(); ctx.arc(camX+8,camY-14,9,0,Math.PI*2); ctx.fill();
      // gât + cap
      ctx.strokeStyle='#A1887F'; ctx.lineWidth=12; ctx.lineCap='round';
      ctx.beginPath(); ctx.moveTo(camX+24,camY-8); ctx.quadraticCurveTo(camX+38,camY-26,camX+36,camY-40); ctx.stroke();
      ctx.fillStyle='#8D6E63'; ctx.beginPath(); ctx.arc(camX+36,camY-42,7,0,Math.PI*2); ctx.fill();
      // ochișor
      ctx.fillStyle='#fff'; ctx.beginPath(); ctx.arc(camX+39,camY-44,2.5,0,Math.PI*2); ctx.fill();
      ctx.fillStyle='#333'; ctx.beginPath(); ctx.arc(camX+39.5,camY-44,1.2,0,Math.PI*2); ctx.fill();
      // picioare
      ctx.strokeStyle='#8D6E63'; ctx.lineWidth=5; ctx.lineCap='round';
      const legPh=t*6;
      for(let i=0;i<4;i++){ const lx=camX-16+i*11,sw=Math.sin(legPh+i*1.6)*5; ctx.beginPath(); ctx.moveTo(lx,camY+10); ctx.lineTo(lx+sw,camY+28); ctx.stroke(); }
      // nisip purtat de vânt
      s.sand.forEach(sd=>{ const x=(sd.x+t*.02*sd.sp)%1.05; ctx.save(); ctx.globalAlpha=.35; ctx.strokeStyle='#FFE0B2'; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(w*x,h*sd.y+Math.sin(t+sd.ph)*4); ctx.quadraticCurveTo(w*x+22,h*sd.y-4,w*x+44,h*sd.y+Math.sin(t+sd.ph+1)*4); ctx.stroke(); ctx.restore(); });
    },
  },

  /* 5. PĂSTOR */
  {
    id: 'pastor',
    keywords: ['oi','oile','turmă','păstor','oaie','miei','mielușea','pășune','Betleem'],
    init(w, h) {
      return {
        clouds: Array.from({length:4},()=>({x:rnd(0,1),y:rnd(.04,.2),cw:rnd(65,140),ch:rnd(30,58),sp:rnd(.1,.3)})),
        sheepArr: Array.from({length:5},(_,i)=>({x:.15+i*.16+rnd(-.04,.04),y:rnd(.78,.88),s:rnd(16,26),ph:rnd(0,6)})),
        flowers: Array.from({length:16},()=>({x:rnd(.05,.95),y:rnd(.72,.96),sz:rnd(10,18),col:pick(['#EC407A','#FFEE58','#AB47BC','#EF5350','#42A5F5','#FF7043'])})),
        bfly: Array.from({length:4},()=>({x:rnd(.1,.9),y:rnd(.38,.62),ph:rnd(0,6),col:pick(['#FF7043','#FFCA28','#AB47BC','#26C6DA'])})),
      };
    },
    draw(ctx, w, h, age, s) {
      const t = age / 1000;
      skyGrad(ctx,w,h,[[0,'#64B5F6'],[.5,'#90CAF9'],[.78,'#C5E1A5'],[1,'#AED581']]);
      smileSun(ctx,w*.16,h*.13,32,t);
      s.clouds.forEach(c=>{ c.x+=.0002*c.sp*60; if(c.x>1.3)c.x=-0.3; fluffyCloud(ctx,w*c.x,h*c.y,c.cw,c.ch,'#fff',.92); });
      // dealuri verzi
      hillBand(ctx,w,h,h*.55,h*.05,2,.4,'#81C784');
      hillBand(ctx,w,h,h*.65,h*.05,2.6,2.4,'#66BB6A');
      hillBand(ctx,w,h,h*.78,h*.045,2.1,4.4,'#4CAF50');
      // copaci
      roundTree(ctx,w*.82,h*.62,80,'#6D4C41','#388E3C');
      roundTree(ctx,w*.72,h*.67,60,'#5D4037','#43A047');
      // flori
      s.flowers.forEach(f=>flower(ctx,w*f.x,h*f.y,f.sz,f.col));
      // fluturi
      s.bfly.forEach(b=>{ const bx=w*b.x+Math.sin(t*.7+b.ph)*32,by=h*b.y+Math.cos(t*.9+b.ph)*18; butterfly(ctx,bx,by,8,b.col,t,b.ph); });
      // păstorul
      const px=w*.24, py=h*.82;
      figure(ctx,px,py,80,'#8D6E63','#FFCC80');
      ctx.strokeStyle='#5D4037'; ctx.lineWidth=5; ctx.lineCap='round';
      ctx.beginPath(); ctx.moveTo(px+26,py-60); ctx.lineTo(px+36,py);
      ctx.moveTo(px+26,py-60); ctx.quadraticCurveTo(px+14,py-76,px+26,py-84); ctx.stroke();
      // oi drăguțe
      s.sheepArr.forEach(sh=>sheep(ctx,w*sh.x,h*sh.y,sh.s,t,sh.ph));
    },
  },

  /* 6. APĂ */
  {
    id: 'apa',
    keywords: ['apă','apele','râu','mare','Iordan','fântână','pârâu','setea','vad'],
    init(w, h) {
      return {
        fish: Array.from({length:4},(_,i)=>({x:rnd(.2,.8),period:rnd(4000,7000),off:i*2000,col:pick(['#FF7043','#FFD740','#40C4FF','#69F0AE'])})),
        birds: Array.from({length:4},()=>({x:rnd(0,1),y:rnd(.06,.22),sp:rnd(.4,.9)})),
        reeds: Array.from({length:7},(_,i)=>({x:i<4?rnd(.02,.14):rnd(.85,.98),hgt:rnd(.1,.16),ph:rnd(0,6)})),
      };
    },
    draw(ctx, w, h, age, s) {
      const t = age / 1000;
      skyGrad(ctx,w,h,[[0,'#E1F5FE'],[.35,'#B3E5FC'],[.5,'#81D4FA'],[1,'#4FC3F7']]);
      smileSun(ctx,w*.82,h*.14,32,t);
      // curcubeu
      rainbow(ctx,w*.5,h*.45,Math.min(w*.42,200),.55+.15*Math.sin(t*.5));
      s.birds.forEach(b=>{ b.x+=.001*b.sp; if(b.x>1.1)b.x=-0.1; bird(ctx,w*b.x,h*b.y+Math.sin(t*2+b.x*8)*6,10,'rgba(30,90,150,.7)'); });
      // mal verde
      hillBand(ctx,w,h,h*.48,h*.03,2.4,1,'#A5D6A7');
      roundTree(ctx,w*.7,h*.5,50,'#5D4037','#43A047');
      // apă animată în benzi
      [[h*.55,8,3,.8,'#0277BD'],[h*.63,9,4,1.1,'#0288D1'],[h*.71,8,5,1.5,'#039BE5'],[h*.79,7,6,1.9,'#29B6F6'],[h*.87,6,7,2.3,'#4FC3F7']]
        .forEach(([yBase,amp,freq,spd,col])=>{
          ctx.fillStyle=col; ctx.beginPath();
          for(let x=0;x<=w;x+=6){ const y=yBase+amp*Math.sin(freq*x/w*Math.PI*2+t*spd); x===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }
          ctx.lineTo(w,h); ctx.lineTo(0,h); ctx.closePath(); ctx.fill();
        });
      // sclipiri pe apă
      for(let i=0;i<10;i++){
        const sx=(i/10+t*.015)%1; ctx.save(); ctx.globalAlpha=.4+.3*Math.sin(t*2+i);
        ctx.strokeStyle='#E1F5FE'; ctx.lineWidth=2;
        ctx.beginPath(); ctx.moveTo(w*sx,h*(.58+(i%4)*.09)); ctx.lineTo(w*sx+18,h*(.58+(i%4)*.09)); ctx.stroke(); ctx.restore();
      }
      // pești colorați sărind
      s.fish.forEach(f=>{
        const p=((age+f.off)%f.period)/f.period; if(p>.3)return;
        const jp=p/.3, fx2=w*f.x+jp*50, fy2=h*.62-Math.sin(jp*Math.PI)*h*.1;
        ctx.save(); ctx.translate(fx2,fy2); ctx.rotate(jp<.5?-.6:.6);
        ctx.fillStyle=f.col; ctx.beginPath(); ctx.ellipse(0,0,18,8,0,0,Math.PI*2); ctx.fill();
        ctx.beginPath(); ctx.moveTo(16,0); ctx.lineTo(26,-8); ctx.lineTo(26,8); ctx.closePath(); ctx.fill();
        ctx.fillStyle='rgba(255,255,255,.6)'; ctx.beginPath(); ctx.arc(-4,0,4,0,Math.PI*2); ctx.fill();
        ctx.restore();
      });
      // trestii
      s.reeds.forEach(r=>{
        const sway=Math.sin(t*1.4+r.ph)*6;
        ctx.strokeStyle='#33691E'; ctx.lineWidth=4; ctx.lineCap='round';
        ctx.beginPath(); ctx.moveTo(w*r.x,h*.98); ctx.quadraticCurveTo(w*r.x+sway*.5,h*(1-r.hgt/2),w*r.x+sway,h*(1-r.hgt)); ctx.stroke();
        ctx.fillStyle='#795548'; ctx.beginPath(); ctx.ellipse(w*r.x+sway,h*(1-r.hgt)-8,4,11,0,0,Math.PI*2); ctx.fill();
      });
    },
  },

  /* 7. CETATE */
  {
    id: 'cetate',
    keywords: ['cetate','cetatea','poartă','porți','zid','ziduri','Ierusalim','turn','Hebron','Gat'],
    init() {
      return { stars: Array.from({length:18},()=>({x:rnd(0,1),y:rnd(0,.28),r:rnd(1.5,3),sp:rnd(.6,2),ph:rnd(0,6)})) };
    },
    draw(ctx, w, h, age, s) {
      const t = age / 1000;
      skyGrad(ctx,w,h,[[0,'#FF8F00'],[.4,'#FFA726'],[.65,'#FFCC80'],[.85,'#FFE082'],[1,'#FFF9C4']]);
      // soare portocaliu la apus
      smileSun(ctx,w*.5,h*.3,38,t);
      // nori
      fluffyCloud(ctx,w*.2,h*.1,90,38,'#fff',.7);
      fluffyCloud(ctx,w*.78,h*.14,70,30,'#fff',.65);
      s.stars.forEach(st=>star5(ctx,w*st.x,h*st.y,st.r,'#FFF9C4',.2+.5*(0.5+.5*Math.sin(t*st.sp+st.ph))));
      hillBand(ctx,w,h,h*.6,h*.03,2,1,'#A1887F');
      // zid principal
      const wallY=h*.55, wallH=h*.3;
      ctx.fillStyle='#FFCC80'; ctx.fillRect(0,wallY,w,wallH);
      // creneluri colorate
      for(let x=0;x<w;x+=44){ ctx.fillStyle=x%88===0?'#FFB74D':'#FFA726'; ctx.fillRect(x,wallY-18,28,18); }
      // rosturi
      ctx.strokeStyle='rgba(141,110,99,.3)'; ctx.lineWidth=1.5;
      for(let y=wallY+24;y<wallY+wallH;y+=26){ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }
      // turnuri
      const tower=(tx,tw2,th2,col)=>{
        ctx.fillStyle=col; ctx.fillRect(tx-tw2/2,wallY-th2,tw2,th2+wallH);
        ctx.fillStyle='rgba(0,0,0,.1)';
        for(let x=tx-tw2/2;x<tx+tw2/2;x+=22){ ctx.fillRect(x,wallY-th2-16,14,16); }
        ctx.fillStyle='#3E2723'; ctx.fillRect(tx-10,wallY-th2+24,20,28);
      };
      tower(w*.13,80,h*.18,'#FFCA28');
      tower(w*.87,80,h*.18,'#FFA726');
      // steaguri colorate
      banner(ctx,w*.13,h*.55-h*.18-16,66,t,'#E53935');
      banner(ctx,w*.87,h*.55-h*.18-16,66,t+2,'#1565C0');
      // poartă arcuită cu aur
      ctx.fillStyle='#4E342E'; ctx.beginPath();
      ctx.moveTo(w*.43,wallY+wallH); ctx.lineTo(w*.43,wallY+wallH*.38);
      ctx.quadraticCurveTo(w*.5,wallY+wallH*.1,w*.57,wallY+wallH*.38);
      ctx.lineTo(w*.57,wallY+wallH); ctx.closePath(); ctx.fill();
      ctx.strokeStyle='#FFD700'; ctx.lineWidth=4; ctx.beginPath();
      ctx.moveTo(w*.43,wallY+wallH*.38); ctx.quadraticCurveTo(w*.5,wallY+wallH*.1,w*.57,wallY+wallH*.38); ctx.stroke();
      // torțe
      flame(ctx,w*.4,wallY+wallH*.4,18,t,1);
      flame(ctx,w*.6,wallY+wallH*.4,18,t,4);
      ctx.fillStyle='#5D4037'; ctx.fillRect(0,h*.86,w,h*.14);
    },
  },

  /* 8. NOAPTE */
  {
    id: 'noapte',
    keywords: ['noapte','noaptea','stele','întuneric','vis','somn','lună','a vorbit Domnul'],
    init(w, h) {
      return {
        stars: Array.from({length:55},()=>({x:rnd(0,1),y:rnd(0,.55),r:rnd(1,3.4),sp:rnd(.5,2.2),ph:rnd(0,6)})),
        shoots: Array.from({length:3},(_,i)=>({trig:2000+i*5000,x0:rnd(.1,.6),y0:rnd(.05,.3),dur:rnd(600,900)})),
        flies: Array.from({length:10},()=>({x:rnd(.1,.9),y:rnd(.58,.86),sp:rnd(.8,2),ph:rnd(0,6)})),
      };
    },
    draw(ctx, w, h, age, s) {
      const t = age / 1000;
      skyGrad(ctx,w,h,[[0,'#0D1B4C'],[.5,'#1A2C6B'],[.85,'#283A80'],[1,'#1A2C6B']]);
      // lună cu față zâmbitoare
      const mx=w*.74, my=h*.18;
      const halo=ctx.createRadialGradient(mx,my,0,mx,my,110);
      halo.addColorStop(0,`rgba(255,249,196,${.35+.12*Math.sin(t*.8)})`); halo.addColorStop(1,'rgba(255,249,196,0)');
      ctx.fillStyle=halo; ctx.fillRect(mx-110,my-110,220,220);
      ctx.fillStyle='#FFF9C4'; ctx.beginPath(); ctx.arc(mx,my,42,0,Math.PI*2); ctx.fill();
      // fața lunii
      ctx.fillStyle='rgba(180,170,120,.4)';
      ctx.beginPath(); ctx.arc(mx-10,my-8,7,0,Math.PI*2); ctx.fill();
      ctx.beginPath(); ctx.arc(mx+12,my+10,5,0,Math.PI*2); ctx.fill();
      ctx.fillStyle='#8B6914';
      [-.25,.25].forEach(dx=>{ ctx.beginPath(); ctx.arc(mx+dx*42,my-42*.08,42*.09,0,Math.PI*2); ctx.fill(); });
      ctx.strokeStyle='#8B6914'; ctx.lineWidth=42*.09; ctx.lineCap='round';
      ctx.beginPath(); ctx.arc(mx,my+42*.12,42*.28,.15,Math.PI-.15); ctx.stroke();
      // stele
      s.stars.forEach(st=>star5(ctx,w*st.x,h*st.y,st.r,'#FFFDE7',.25+.65*(0.5+.5*Math.sin(t*st.sp+st.ph))));
      // stele căzătoare
      s.shoots.forEach(sh=>{
        const el=(age-sh.trig)%15000; if(el<0||el>sh.dur)return;
        const p=el/sh.dur, x=w*sh.x0+p*190, y=h*sh.y0+p*80;
        ctx.save(); ctx.globalAlpha=p<.3?p/.3:1-(p-.3)/.7;
        const g=ctx.createLinearGradient(x,y,x-70,y-30); g.addColorStop(0,'#fff'); g.addColorStop(1,'rgba(255,255,255,0)');
        ctx.strokeStyle=g; ctx.lineWidth=2.5; ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(x-70,y-30); ctx.stroke();
        star5(ctx,x,y,4,'#fff',1); ctx.restore();
      });
      hillBand(ctx,w,h,h*.66,h*.04,2.2,1.2,'#152258');
      hillBand(ctx,w,h,h*.78,h*.04,1.8,3.6,'#0F1A45');
      // foc de tabără
      const fx=w*.35, fy=h*.88;
      ctx.strokeStyle='#4E342E'; ctx.lineWidth=5; ctx.lineCap='round';
      ctx.beginPath(); ctx.moveTo(fx-14,fy); ctx.lineTo(fx+14,fy-6); ctx.moveTo(fx-14,fy-6); ctx.lineTo(fx+14,fy); ctx.stroke();
      flame(ctx,fx,fy-4,24,t);
      // siluetă dormind
      ctx.fillStyle='#37474F'; ctx.beginPath(); ctx.ellipse(w*.55,h*.9,46,13,0,0,Math.PI*2); ctx.fill();
      ctx.beginPath(); ctx.arc(w*.55+42,h*.885,11,0,Math.PI*2); ctx.fill();
      // licurici colorați
      s.flies.forEach(f=>{
        const gl=.3+.7*(0.5+.5*Math.sin(t*f.sp+f.ph));
        ctx.save(); ctx.globalAlpha=gl; ctx.fillStyle='#CDDC39';
        ctx.beginPath(); ctx.arc(w*f.x+Math.sin(t*.6+f.ph)*16,h*f.y+Math.cos(t*.5+f.ph)*10,3,0,Math.PI*2); ctx.fill(); ctx.restore();
      });
    },
  },

  /* 9. MUNTE */
  {
    id: 'munte',
    keywords: ['munte','muntele','deal','stâncă','vârf','Ghilboa','Carmel','înălțimi','suit'],
    init(w, h) {
      return {
        clouds: Array.from({length:5},()=>({x:rnd(0,1),y:rnd(.06,.32),cw:rnd(75,180),ch:rnd(34,78),sp:rnd(.1,.35)})),
        eagleOff: 0,
      };
    },
    draw(ctx, w, h, age, s) {
      const t = age / 1000;
      skyGrad(ctx,w,h,[[0,'#64B5F6'],[.4,'#90CAF9'],[.68,'#BBDEFB'],[1,'#E3F2FD']]);
      smileSun(ctx,w*.82,h*.13,34,t);
      // munți colorați în straturi
      const mtn=(baseX,peakY,half,col,snow)=>{
        ctx.fillStyle=col; ctx.beginPath();
        ctx.moveTo(baseX-half,h*.8); ctx.lineTo(baseX,peakY); ctx.lineTo(baseX+half,h*.8); ctx.closePath(); ctx.fill();
        if(snow){
          ctx.fillStyle='#FFFFFF'; ctx.beginPath();
          ctx.moveTo(baseX,peakY); ctx.lineTo(baseX-half*.22,peakY+(h*.8-peakY)*.24);
          ctx.lineTo(baseX-half*.1,peakY+(h*.8-peakY)*.19); ctx.lineTo(baseX,peakY+(h*.8-peakY)*.27);
          ctx.lineTo(baseX+half*.1,peakY+(h*.8-peakY)*.18); ctx.lineTo(baseX+half*.22,peakY+(h*.8-peakY)*.24); ctx.closePath(); ctx.fill();
        }
      };
      mtn(w*.85,h*.32,w*.44,'#9FA8DA',false);
      mtn(w*.18,h*.22,w*.48,'#7986CB',true);
      mtn(w*.54,h*.1,w*.6,'#5C6BC0',true);
      // nori
      s.clouds.forEach(c=>{ c.x+=.00025*c.sp*60; if(c.x>1.3)c.x=-0.3; fluffyCloud(ctx,w*c.x,h*c.y,c.cw,c.ch,'#fff',.88); });
      // vultur
      const ea=t*.4, ex=w*.4+Math.cos(ea)*w*.16, ey=h*.24+Math.sin(ea)*h*.06;
      ctx.save(); ctx.strokeStyle='#4E342E'; ctx.lineWidth=3.5; ctx.lineCap='round';
      const flap=Math.sin(t*4)*7;
      ctx.beginPath(); ctx.moveTo(ex-22,ey-flap); ctx.quadraticCurveTo(ex,ey+7,ex+22,ey-flap); ctx.stroke(); ctx.restore();
      // poale verzi + brazi
      hillBand(ctx,w,h,h*.8,h*.03,2,1,'#66BB6A');
      pineTree(ctx,w*.1,h*.92,62);
      pineTree(ctx,w*.22,h*.96,46);
      pineTree(ctx,w*.88,h*.94,56);
      pineTree(ctx,w*.76,h*.97,40);
    },
  },

  /* 10. RUGĂCIUNE */
  {
    id: 'rugaciune',
    keywords: ['rugăciune','rugat','plâns','suflet','jelit','postit','Doamne','cerut','juruință','binecuvânt'],
    init() {
      return {
        motes: Array.from({length:18},()=>({x:rnd(.3,.7),y0:rnd(.5,.95),period:rnd(4000,9000),off:rnd(0,9000),r:rnd(1.5,3.5)})),
        doves: Array.from({length:3},(_,i)=>({off:i*4500})),
        flowers: Array.from({length:10},()=>({x:rnd(.02,.98),sz:rnd(11,18),col:pick(['#FF80AB','#FFCA28','#CE93D8','#FF7043','#80CBC4'])})),
      };
    },
    draw(ctx, w, h, age, s) {
      const t = age / 1000;
      skyGrad(ctx,w,h,[[0,'#FCE4EC'],[.4,'#F8BBD0'],[.7,'#FFE0B2'],[1,'#FFF3E0']]);
      // lumină divină
      const gx=w/2, gy=h*.08;
      const glow=ctx.createRadialGradient(gx,gy,0,gx,gy,h*.35);
      glow.addColorStop(0,`rgba(255,241,118,${.7+.2*Math.sin(t)})`); glow.addColorStop(1,'rgba(255,241,118,0)');
      ctx.fillStyle=glow; ctx.fillRect(0,0,w,h*.55);
      // raze
      for(let i=0;i<8;i++){
        const ang=Math.PI/2+(i-3.5)*.17;
        ctx.save(); ctx.globalAlpha=.18+.14*Math.sin(t*1.1+i); ctx.fillStyle='#FFF59D'; ctx.beginPath();
        ctx.moveTo(gx,gy); ctx.lineTo(gx+Math.cos(ang-.04)*h,gy+Math.sin(ang-.04)*h); ctx.lineTo(gx+Math.cos(ang+.04)*h,gy+Math.sin(ang+.04)*h); ctx.closePath(); ctx.fill(); ctx.restore();
      }
      // curcubeu subtil
      rainbow(ctx,w*.5,h*.6,Math.min(w*.38,180),.35+.1*Math.sin(t*.4));
      // porumbei albi
      s.doves.forEach(d=>{ const p=((age+d.off)%12000)/12000; bird(ctx,w*(.1+p*.8),h*.3-Math.sin(p*Math.PI)*h*.12+Math.sin(t*3)*4,13,'rgba(255,255,255,.95)'); });
      hillBand(ctx,w,h,h*.76,h*.04,1.8,.8,'#A5D6A7');
      hillBand(ctx,w,h,h*.9,h*.03,2.4,2.8,'#81C784');
      // flori colorate
      s.flowers.forEach((f,i)=>flower(ctx,w*(0.04+i*.1),h*.92,f.sz,f.col));
      // siluetă îngenuncheată
      const kx=w/2, ky=h*.84;
      ctx.fillStyle='#5D4037';
      ctx.beginPath(); ctx.ellipse(kx,ky-26,24,32,-.35,0,Math.PI*2); ctx.fill();
      ctx.beginPath(); ctx.arc(kx-15,ky-52,15,0,Math.PI*2); ctx.fill();
      ctx.beginPath(); ctx.ellipse(kx+12,ky+2,22,12,0,0,Math.PI*2); ctx.fill();
      ctx.fillStyle='#FFCC80'; ctx.beginPath(); ctx.ellipse(kx-20,ky-24,8,12,-.4,0,Math.PI*2); ctx.fill();
      // particule de lumină urcând
      s.motes.forEach(m=>{ const p=((age+m.off)%m.period)/m.period; ctx.save(); ctx.globalAlpha=.7*(1-p); ctx.fillStyle='#FFF176'; ctx.beginPath(); ctx.arc(w*m.x+Math.sin(p*6+m.off)*14,h*m.y0-p*h*.4,m.r,0,Math.PI*2); ctx.fill(); ctx.restore(); });
    },
  },

  /* 11. FUGĂ */
  {
    id: 'fuga',
    keywords: ['fugi','fugit','fuga','urmărit','ascuns','scăpat','pribeag','peștera','Adulam'],
    init() {
      return {
        dust: Array.from({length:6},(_,i)=>({off:i*380})),
        stars: Array.from({length:12},()=>({x:rnd(0,1),y:rnd(0,.25),r:rnd(1.5,3),ph:rnd(0,6)})),
      };
    },
    draw(ctx, w, h, age, s) {
      const t = age / 1000;
      skyGrad(ctx,w,h,[[0,'#FF6F00'],[.4,'#FF8F00'],[.7,'#FFA726'],[1,'#FFB74D']]);
      smileSun(ctx,w*.85,h*.55,28,t);
      s.stars.forEach(st=>star5(ctx,w*st.x,h*st.y,st.r,'#FFF9C4',.4+.4*Math.sin(t+st.ph)));
      hillBand(ctx,w,h,h*.55,h*.05,1.6,.6,'#A1887F');
      hillBand(ctx,w,h,h*.7,h*.03,2,.8,'#8D6E63');
      // gura peșterii
      ctx.fillStyle='#6D4C41'; ctx.beginPath(); ctx.moveTo(w*.04,h*.8); ctx.quadraticCurveTo(w*.17,h*.44,w*.32,h*.8); ctx.closePath(); ctx.fill();
      ctx.fillStyle='#37474F'; ctx.beginPath(); ctx.moveTo(w*.09,h*.8); ctx.quadraticCurveTo(w*.17,h*.52,w*.27,h*.8); ctx.closePath(); ctx.fill();
      // lilieci cartoon (drăguți)
      for(let i=0;i<3;i++){
        const bx=w*.12+Math.sin(t*1.4+i*2)*28, by=h*.48+Math.cos(t*1.8+i*2)*14+i*14;
        ctx.fillStyle='#7E57C2'; ctx.save(); ctx.globalAlpha=.8;
        ctx.beginPath(); ctx.ellipse(bx,by,9,6,0,0,Math.PI*2); ctx.fill();
        const flap=Math.abs(Math.sin(t*8+i))*12;
        ctx.beginPath(); ctx.ellipse(bx-flap,by-2,flap,4,-.3,0,Math.PI*2); ctx.fill();
        ctx.beginPath(); ctx.ellipse(bx+flap,by-2,flap,4,.3,0,Math.PI*2); ctx.fill();
        ctx.fillStyle='#fff'; ctx.beginPath(); ctx.arc(bx-3,by-2,2,0,Math.PI*2); ctx.fill();
        ctx.restore();
      }
      ctx.fillStyle='#6D4C41'; ctx.fillRect(0,h*.8,w,h*.2);
      // siluetă alergând
      const runX=w*.6, runY=h*.8+Math.abs(Math.sin(t*5))*-8;
      ctx.save(); ctx.translate(runX,runY); ctx.rotate(.1);
      figure(ctx,0,0,66,'#E65100','#FFCC80');
      ctx.fillStyle='rgba(255,143,0,.7)'; ctx.beginPath();
      ctx.moveTo(-8,-40); ctx.quadraticCurveTo(-48-Math.sin(t*6)*8,-22,-60,6); ctx.quadraticCurveTo(-30,-4,-6,-6); ctx.closePath(); ctx.fill();
      const st2=Math.sin(t*10)*16;
      ctx.strokeStyle='#E65100'; ctx.lineWidth=9; ctx.lineCap='round';
      ctx.beginPath(); ctx.moveTo(0,-6); ctx.lineTo(st2,22); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0,-6); ctx.lineTo(-st2,20); ctx.stroke();
      ctx.restore();
      s.dust.forEach(d=>{ const p=((age+d.off)%1400)/1400; ctx.save(); ctx.globalAlpha=.4*(1-p); ctx.fillStyle='#BCAAA4'; ctx.beginPath(); ctx.ellipse(runX-30-p*44,h*.82-p*12,9*(1+p),5*(1+p),0,0,Math.PI*2); ctx.fill(); ctx.restore(); });
    },
  },

  /* 12. VICTORIE */
  {
    id: 'victorie',
    keywords: ['biruință','slavă','laudă','izbăvit','izbăvire','triumf','Goliat','înfrânt','bucurie','cântat'],
    init() {
      return {
        // câțiva fluturași care plutesc lin prin scenă, nu stele care cad
        // (o scenă de zi senină, cu soare, nu are ce căuta cu stele căzătoare)
        bflies: Array.from({length:4},()=>({x:rnd(.1,.9),y:rnd(.15,.5),ph:rnd(0,6),sp:rnd(.15,.3),col:pick(['#FFD54F','#FF7043','#4FC3F7','#AED581','#F06292','#CE93D8'])})),
        // culorile florilor se aleg o singură dată aici, nu în draw() (care rulează
        // la 60fps) — altfel fiecare cadru re-alegea culoarea și florile păreau
        // că clipesc/își schimbă culoarea.
        flowerCols: Array.from({length:6},()=>pick(['#FF6B9D','#FFCA28','#7C4DFF','#FF7043'])),
      };
    },
    draw(ctx, w, h, age, s) {
      const t = age / 1000;
      skyGrad(ctx,w,h,[[0,'#E1F5FE'],[.35,'#B3E5FC'],[.6,'#FFF9C4'],[1,'#FFF176']]);
      smileSun(ctx,w*.14,h*.14,36,t);
      // curcubeu mare
      rainbow(ctx,w*.5,h*.55,Math.min(w*.46,230),.75+.1*Math.sin(t*.5));
      // raze rotitoare
      ctx.save(); ctx.translate(w/2,h*.4); ctx.rotate(t*.1);
      for(let i=0;i<12;i++){ ctx.rotate(Math.PI/6); ctx.globalAlpha=.12; ctx.fillStyle='#FFF59D'; ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(-w*.06,-h); ctx.lineTo(w*.06,-h); ctx.closePath(); ctx.fill(); }
      ctx.restore();
      // fluturași care plutesc lin, cu traiectorie ondulată, lentă
      s.bflies.forEach(b=>{
        const bx=w*b.x+Math.sin(t*b.sp+b.ph)*w*.12;
        const by=h*b.y+Math.cos(t*b.sp*.8+b.ph)*h*.06;
        butterfly(ctx,bx,by,9,b.col,t,b.ph);
      });
      hillBand(ctx,w,h,h*.7,h*.05,1.6,1,'#A5D6A7');
      hillBand(ctx,w,h,h*.84,h*.04,2.2,3,'#81C784');
      // flori
      for(let i=0;i<6;i++) flower(ctx,w*(.06+i*.18),h*.88,16,s.flowerCols[i]);
      // învingătorul
      const vx=w/2, vy=h*.72;
      figure(ctx,vx,vy,90,'#1565C0','#FFCC80');
      const raise=Math.sin(t*2.4)*.1;
      ctx.save(); ctx.translate(vx+22,vy-62); ctx.rotate(-.7+raise);
      ctx.strokeStyle='#FFD700'; ctx.lineWidth=7; ctx.lineCap='round'; ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(0,-60); ctx.stroke();
      ctx.strokeStyle='#CFD8DC'; ctx.lineWidth=8; ctx.beginPath(); ctx.moveTo(0,-14); ctx.lineTo(0,-72); ctx.stroke();
      ctx.restore();
      star5(ctx,vx+22+Math.cos(-.7+raise+Math.PI/2)*72,vy-62-Math.sin(.7-raise+Math.PI/2)*68,10,'#FFF176',.7+.3*Math.sin(t*5));
      banner(ctx,vx-64,vy,104,t,'#F9A825');
    },
  },

  /* 13. PERGAMENT */
  {
    id: 'pergament',
    keywords: [],
    init() {
      return { sparks: Array.from({length:14},()=>({x:rnd(.12,.88),y:rnd(.12,.82),sp:rnd(.7,1.8),ph:rnd(0,6),r:rnd(2,4.5)})) };
    },
    draw(ctx, w, h, age, s) {
      const t = age / 1000;
      skyGrad(ctx,w,h,[[0,'#FFF8E1'],[.5,'#FFECB3'],[1,'#FFE082']]);
      // stele aurii
      s.sparks.forEach(sp=>star5(ctx,w*sp.x,h*sp.y,sp.r,'#FFB300',.25+.5*(0.5+.5*Math.sin(t*sp.sp+sp.ph))));
      // flori de colț
      flower(ctx,w*.06,h*.88,16,'#FF80AB');
      flower(ctx,w*.94,h*.88,16,'#FFCA28');
      flower(ctx,w*.06,h*.2,14,'#CE93D8');
      flower(ctx,w*.94,h*.2,14,'#80DEEA');
      // sul de pergament
      const bob=Math.sin(t*.9)*6;
      const sx=w/2, sy=h*.5+bob, sw=Math.min(w*.6,320), sh=h*.44;
      ctx.save(); ctx.globalAlpha=.18; ctx.fillStyle='#795548';
      ctx.beginPath(); ctx.ellipse(sx,h*.87,sw*.55,16,0,0,Math.PI*2); ctx.fill(); ctx.restore();
      // corp
      ctx.fillStyle='#FFF3E0'; ctx.fillRect(sx-sw/2,sy-sh/2,sw,sh);
      ctx.strokeStyle='#D7CCC8'; ctx.lineWidth=2; ctx.strokeRect(sx-sw/2,sy-sh/2,sw,sh);
      // suluri sus/jos
      const roll=(ry)=>{
        ctx.fillStyle='#A1887F'; ctx.beginPath(); ctx.ellipse(sx,ry,sw*.56,16,0,0,Math.PI*2); ctx.fill();
        ctx.fillStyle='#BCAAA4'; ctx.beginPath(); ctx.ellipse(sx,ry,sw*.48,10,0,0,Math.PI*2); ctx.fill();
        ctx.fillStyle='#8D6E63'; [sx-sw*.56,sx+sw*.56].forEach(x=>{ ctx.beginPath(); ctx.arc(x,ry,10,0,Math.PI*2); ctx.fill(); });
      };
      roll(sy-sh/2); roll(sy+sh/2);
      // rânduri de text animate
      for(let i=0;i<7;i++){
        const ly=sy-sh/2+36+i*(sh-64)/7;
        const lit=.35+.5*(.5+.5*Math.sin(t*1.4-i*.7));
        ctx.save(); ctx.globalAlpha=lit; ctx.strokeStyle='#8D6E63'; ctx.lineWidth=5; ctx.lineCap='round';
        ctx.beginPath(); ctx.moveTo(sx-sw*.36,ly); ctx.lineTo(sx+sw*(.18+(i%3)*.08),ly); ctx.stroke(); ctx.restore();
      }
      // pană de scris
      const qx=sx+sw*.42, qy=sy-sh*.34+Math.sin(t*1.6)*8;
      ctx.save(); ctx.translate(qx,qy); ctx.rotate(-.5+Math.sin(t)*.06);
      ctx.fillStyle='#ECEFF1'; ctx.beginPath(); ctx.moveTo(0,0); ctx.quadraticCurveTo(10,-26,4,-46); ctx.quadraticCurveTo(-2,-26,0,0); ctx.fill();
      ctx.strokeStyle='#90A4AE'; ctx.lineWidth=1.4; ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(3,-44); ctx.stroke();
      ctx.restore();
    },
  },
];

/* ── alegerea scenei ─────────────────────────────────────────── */
const NEUTRAL_ROTATION = ['pastor', 'munte', 'apa', 'noapte', 'pergament'];

function pickScene(pageVerses, pageIndex = 0, avoidId = null) {
  const text = pageVerses.map(v => v.text.replace('{0}', v.blanks[0].answer)).join(' ').toLowerCase();
  let matches = SCENES.filter(sc => sc.keywords.length && sc.keywords.some(kw => text.includes(kw.toLowerCase())));

  if (matches.length === 0) {
    const pool = NEUTRAL_ROTATION.map(id => SCENES.find(sc => sc.id === id)).filter(Boolean);
    const rotated = pool[pageIndex % pool.length];
    if (rotated.id !== avoidId || pool.length === 1) return rotated;
    return pool[(pageIndex + 1) % pool.length];
  }

  if (avoidId && matches.length > 1) {
    const filtered = matches.filter(sc => sc.id !== avoidId);
    if (filtered.length > 0) matches = filtered;
  }
  return matches[pageIndex % matches.length];
}

/* ── motor de randare: canvas fix, tranziție slide+fade ────── */
const SceneEngine = (() => {
  let canvas = null, ctx = null, w = 0, h = 0;
  let cur = null, prev = null, fadeStart = 0;
  const FADE_MS = 1400;
  let mirror = false, slideDir = 1;

  function ease(t) { return t < .5 ? 4*t*t*t : 1 - Math.pow(-2*t+2,3)/2; }

  function resize() {
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = window.innerWidth; h = window.innerHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (cur) cur.state = cur.scene.init(w, h);
    if (prev) prev.state = prev.scene.init(w, h);
  }

  function drawScene(entry, now, alpha, offsetX) {
    ctx.save(); ctx.globalAlpha = alpha;
    if (offsetX) ctx.translate(offsetX, 0);
    if (mirror) { ctx.translate(w, 0); ctx.scale(-1, 1); }
    try { entry.scene.draw(ctx, w, h, now - entry.start, entry.state); } catch(e) {}
    ctx.restore();
  }

  function frame(now) {
    if (!canvas || !cur) return;
    ctx.clearRect(0, 0, w, h);
    if (prev && now - fadeStart < FADE_MS) {
      const p = ease((now - fadeStart) / FADE_MS);
      drawScene(prev, now, 1 - p * .7, -slideDir * w * .35 * p);
      drawScene(cur,  now, p,           slideDir * w * (1 - p) * .35);
    } else {
      prev = null; drawScene(cur, now, 1, 0);
    }
  }

  function tick(now) { frame(now); requestAnimationFrame(tick); }

  function ensureCanvas() {
    if (canvas) return true;
    canvas = document.getElementById('scene-canvas');
    if (!canvas) return false;
    ctx = canvas.getContext('2d');
    window.addEventListener('resize', resize);
    resize(); requestAnimationFrame(tick);
    return true;
  }

  function show(scene, mirrored) {
    if (!ensureCanvas()) return;
    mirror = !!mirrored;
    if (cur && cur.scene.id === scene.id) return;
    slideDir *= -1;
    prev = cur;
    cur = { scene, state: scene.init(w, h), start: performance.now() };
    fadeStart = performance.now();
  }

  return { show, frame };
})();
