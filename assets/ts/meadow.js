let AC;
function initAudio() {
  AC = AC || new (window.AudioContext || window.webkitAudioContext)();
  if (AC.state === "suspended") AC.resume();
}
document.addEventListener("click",     initAudio, { once: true });
document.addEventListener("mousemove", initAudio, { once: true });

function note(freq, duration = 0.4, delay = 0, type = "sine", vol = 0.1) {
  if (!AC) return;
  const osc   = AC.createOscillator();
  const gain  = AC.createGain();
  const t     = AC.currentTime + delay;
  osc.type = type;
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0, t);
  gain.gain.linearRampToValueAtTime(vol, t + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.001, t + duration);
  osc.connect(gain);
  gain.connect(AC.destination);
  osc.start(t);
  osc.stop(t + duration + 0.05);
}

function sndLadybug() {
  [880, 988, 1047].forEach((f, i) => note(f, 0.2, i * 0.07, "sine", 0.08));
}

function sndBee() {
  for (let i = 0; i < 6; i++) 
    note(180 + Math.random() * 40, 0.03, i * 0.04, "sawtooth", 0.04);
}

function sndFox() {
  [523, 659, 587].forEach((f, i) => note(f, 0.35, i * 0.12, "triangle",  0.09));
}

function sndRabbit() { 
  note(659, 0.08, 0, "square", 0.07);
  note(784, 0.08, 0.07, "square", 0.07);
  note(880, 0.12, 0.14, "square", 0.06);
}

function sndOwl() {
  note(330, 0.6, 0, "triangle", 0.1);
  note(262, 0.8, 0.2, "triangle", 0.08);
}

function sndMush() {
  note(220, 0.25, 0, "square", 0.08);
}

function sndStar() {
  const nbs = [523,659,784,1047,659,784,1047,1319,784,1047,1319,1568];
  nbs.forEach((f,i) => note(f, 0.12, i * 0.055, "square", 0.07));
}

function sndHedge() {
  [440, 520, 600].forEach((f, i) => note(f, 0.15, i * 0.08, "triangle", 0.08));
}

function sndSnail() {
  [330, 294, 262].forEach((f, i) => note(f, 0.5,  i * 0.18, "triangle", 0.07));
}

function sndFish() {
  [400, 300, 200].forEach((f, i) => note(f, 0.25, i * 0.06, "sine",     0.07));
}

const tip = document.getElementById("critter-tip");
let tipTimer;

function critterClick(e, msg) {
  e.stopPropagation();
  clearTimeout(tipTimer);

  const id = e.currentTarget.id || "";
  if      (id.includes("fox"))    sndFox();
  else if (id.includes("rabbit")) sndRabbit();
  else if (id.includes("owl"))    sndOwl();
  else if (id.includes("mush"))   sndMush();
  else if (id.includes("hedge"))  sndHedge();
  else if (id.includes("snail"))  sndSnail();
  else if (id.includes("fish"))   sndFish();
  else sndLadybug();

  tip.textContent  = msg;
  tip.style.left   = (e.clientX + 12) + "px";
  tip.style.top    = (e.clientY - 30) + "px";
  tip.classList.add("show");
  tipTimer = setTimeout(() => tip.classList.remove("show"), 1800);
}

document.getElementById("ladybug").addEventListener("click", e => {
  critterClick(e, "🐞 wheee!");
  sndLadybug();
});

const ladybug = document.getElementById("ladybug");
const meadow  = document.getElementById("meadow");

function updateLadybug() {
  const nh = (document.documentElement.scrollHeight - window.innerHeight || 1);
  const scrollPct = window.scrollY / nh;
  const meadowTop = meadow.getBoundingClientRect().top;
  const fadeStart = window.innerHeight * 0.90;
  const fadeEnd   = window.innerHeight * 0.50;
  const fade = 1 - (meadowTop - fadeEnd) / (fadeStart - fadeEnd);
  const escapePct = Math.max(0, Math.min(1, fade));
  const y = window.innerHeight - 40 - (scrollPct * (window.innerHeight - 60));

  ladybug.style.top = y + "px";
  ladybug.style.bottom = "auto";
  ladybug.style.transform =
    `rotate(${-10 + scrollPct * 20}deg) translateX(${-escapePct * 60}px)`;
}

window.addEventListener("scroll", updateLadybug);
updateLadybug();

const snailEl = document.getElementById("snail-c");
let snailX    = -60;
let snailLast = performance.now();

function updateSnail(now) {
  const dt = Math.min(32, now - snailLast);
  snailLast = now;
  snailX += dt * 0.018;
  if (snailX > 1660) snailX = -60;
  snailEl.setAttribute("transform", `translate(${snailX}, 650)`);
  requestAnimationFrame(updateSnail);
}
requestAnimationFrame(updateSnail);

const bee = document.getElementById("bee-follow");
let beeX = 100, beeY = 100;

document.addEventListener("mousemove", e => {
  beeX = e.clientX + 30;
  beeY = e.clientY - 20;
  bee.style.left = beeX + "px";
  bee.style.top  = beeY + "px";
});

document.addEventListener("DOMContentLoaded", () => {
  const headingIcons = [
    { text: "Employment History", icon: " ⭐", sound: sndStar },
    { text: "Open Source",        icon: " 🐝", sound: sndBee  },
    { text: "Skills",             icon: " 🍄", sound: sndMush },
    { text: "Education",          icon: " 🦉", sound: sndOwl  },
  ];

  document.querySelectorAll("h2").forEach(heading => {
    headingIcons.forEach(({ text, icon, sound }) => {
      if (!heading.textContent.includes(text)) return;
      const span = document.createElement("span");
      span.textContent = icon;
      span.style.cssText = "cursor:pointer;font-size:1em;";
      span.addEventListener("click", e => {
        e.preventDefault();
        initAudio();
        sound();
        span.style.transform = "scale(1.5)";
        setTimeout(() => (span.style.transform = ""), 300);
      });
      heading.appendChild(span);
    });
  });
});
