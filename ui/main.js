"use strict";

const $ = id => document.getElementById(id);
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

/* ---------- page chrome: title split, top bar, scroll progress, reveal ---------- */
function splitTitle() {
  document.querySelectorAll("[data-split]").forEach(el => {
    el.innerHTML = [...el.textContent].map((c, i) => `<span class="ch" style="--i:${i}">${c}</span>`).join("");
  });
}

function chrome() {
  const bar = $("scroll-bar"), top = $("topbar"), hero = $("top");
  const onScroll = () => {
    const h = document.documentElement.scrollHeight - innerHeight;
    bar.style.transform = `scaleX(${h > 0 ? scrollY / h : 0})`;
    top.classList.toggle("solid", scrollY > hero.offsetHeight - 80);
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const links = [...document.querySelectorAll(".topbar nav a[href^='#']")];
  const spy = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) links.forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
  }), { rootMargin: "-45% 0px -50% 0px" });
  links.forEach(a => { const s = document.querySelector(a.getAttribute("href")); if (s) spy.observe(s); });

  const rev = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.classList.add("in");
    rev.unobserve(e.target);
  }), { threshold: .15 });
  document.querySelectorAll(".reveal").forEach(el => rev.observe(el));

  document.querySelectorAll(".step-card").forEach(card => card.addEventListener("pointermove", e => {
    const r = card.getBoundingClientRect();
    card.style.setProperty("--mx", `${e.clientX - r.left}px`);
    card.style.setProperty("--my", `${e.clientY - r.top}px`);
  }));


}


/* ---------- hero: attempts climb staircases; every rise is a check that passed ---------- */
function heroCanvas() {
  const cv = $("hero-canvas"), ctx = cv.getContext("2d"), hero = $("top");
  let W = 0, H = 0, dpr = 1, paths = [], sparks = [], visible = true, last = 0;
  const mouse = { x: 0, y: 0, tx: 0, ty: 0 };

  function resize() {
    dpr = Math.min(devicePixelRatio || 1, 2);
    W = hero.clientWidth; H = hero.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.round(clamp(W / 55, 10, 30));
    paths = Array.from({ length: n }, () => makePath(true));
  }

  function makePath(scatter) {
    const K = 3, rise = H * (.07 + Math.random() * .06), depth = .3 + Math.random() * .7;
    const passed = Math.random() < .38 ? K : Math.floor(Math.random() * K);
    const len = W * (.35 + Math.random() * .5), seg = len / (K + 1.5);
    let x = -40 + Math.random() * W * .95 - len * .35, y = H * (.45 + Math.random() * .6);
    const pts = [[x, y]], rises = [];
    for (let k = 0; k < K; k++) {
      x += seg * (.6 + Math.random() * .8); pts.push([x, y]);
      if (k < passed) { rises.push(pts.length); y -= rise; pts.push([x, y]); }
    }
    x += seg * .8; pts.push([x, y]);
    const cum = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    const total = cum.at(-1);
    return { pts, cum, total, rises, fired: new Set(), ok: passed === K, depth, speed: 70 + Math.random() * 110,
      d: scatter ? Math.random() * total : -Math.random() * 400, fade: 1, done: false };
  }

  function at(p, d) {
    let i = 1; while (i < p.cum.length - 1 && p.cum[i] < d) i++;
    const a = p.pts[i - 1], b = p.pts[i], s = (d - p.cum[i - 1]) / ((p.cum[i] - p.cum[i - 1]) || 1);
    return [a[0] + (b[0] - a[0]) * clamp(s, 0, 1), a[1] + (b[1] - a[1]) * clamp(s, 0, 1), i];
  }

  function trace(p, d, ox, oy) {
    ctx.beginPath(); ctx.moveTo(p.pts[0][0] + ox, p.pts[0][1] + oy);
    for (let i = 1; i < p.pts.length && p.cum[i - 1] < d; i++) {
      const [x, y] = p.cum[i] <= d ? p.pts[i] : at(p, d);
      ctx.lineTo(x + ox, y + oy);
    }
    ctx.stroke();
  }

  function frame(now) {
    const dt = Math.min((now - (last || now)) / 1000, .05); last = now;
    mouse.x += (mouse.tx - mouse.x) * .05; mouse.y += (mouse.ty - mouse.y) * .05;
    ctx.clearRect(0, 0, W, H);

    // dotted field with slight parallax
    ctx.fillStyle = "rgba(160, 200, 230, .07)";
    const gx = mouse.x * 6, gy = mouse.y * 6;
    for (let x = (gx % 34) - 34; x < W + 34; x += 34) for (let y = (gy % 34) - 34; y < H + 34; y += 34) ctx.fillRect(x, y, 1.3, 1.3);

    ctx.lineCap = "round"; ctx.lineJoin = "round";
    paths.forEach((p, idx) => {
      const ox = mouse.x * 26 * p.depth, oy = mouse.y * 16 * p.depth;
      if (!p.done) {
        p.d += p.speed * dt;
        if (p.d >= p.total) {
          p.d = p.total; p.done = true;
          const [ex, ey] = p.pts.at(-1);
          sparks.push({ x: ex, y: ey, r: 0, life: 1, depth: p.depth, ok: p.ok, big: true });
        }
      } else {
        p.fade -= dt * .45;
        if (p.fade <= 0) { paths[idx] = makePath(false); return; }
      }
      if (p.d <= 0) return;
      p.rises.forEach(ri => {
        if (!p.fired.has(ri) && p.d >= p.cum[ri]) {
          p.fired.add(ri);
          const [sx, sy] = p.pts[ri];
          sparks.push({ x: sx, y: sy, r: 0, life: 1, depth: p.depth, ok: true, label: p.depth > .62 });
        }
      });
      const a = p.fade * p.depth;
      ctx.strokeStyle = `rgba(80, 209, 196, ${.2 * a})`; ctx.lineWidth = 1 + p.depth * .8;
      trace(p, p.d, ox, oy);
      if (!p.done) {
        const tail = Math.max(0, p.d - 140);
        const [hx, hy] = at(p, p.d), [tx, ty] = at(p, tail);
        const g = ctx.createLinearGradient(tx + ox, ty + oy, hx + ox, hy + oy);
        g.addColorStop(0, "rgba(127, 240, 226, 0)"); g.addColorStop(1, `rgba(160, 255, 244, ${.85 * p.depth})`);
        ctx.strokeStyle = g; ctx.lineWidth = 1.6 + p.depth * 1.4;
        ctx.save(); ctx.beginPath();
        // draw only the last 140px
        ctx.moveTo(tx + ox, ty + oy);
        for (let i = 1; i < p.pts.length; i++) if (p.cum[i] > tail && p.cum[i] < p.d) ctx.lineTo(p.pts[i][0] + ox, p.pts[i][1] + oy);
        ctx.lineTo(hx + ox, hy + oy); ctx.stroke(); ctx.restore();
        ctx.fillStyle = `rgba(200, 255, 248, ${p.depth})`;
        ctx.shadowColor = "#50d1c4"; ctx.shadowBlur = 14 * p.depth;
        ctx.beginPath(); ctx.arc(hx + ox, hy + oy, 1.6 + p.depth * 1.6, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0;
      } else {
        const [ex, ey] = p.pts.at(-1);
        ctx.fillStyle = p.ok ? `rgba(127, 240, 226, ${a})` : `rgba(224, 110, 136, ${.75 * a})`;
        ctx.beginPath(); ctx.arc(ex + ox, ey + oy, 2.4, 0, Math.PI * 2); ctx.fill();
      }
    });

    ctx.font = "600 11px 'JetBrains Mono', monospace";
    sparks = sparks.filter(s => (s.life -= dt * (s.big ? .9 : 1.3)) > 0);
    sparks.forEach(s => {
      s.r += dt * (s.big ? 70 : 42);
      const ox = mouse.x * 26 * s.depth, oy = mouse.y * 16 * s.depth, a = s.life * s.depth;
      ctx.strokeStyle = s.ok ? `rgba(127, 240, 226, ${a})` : `rgba(224, 110, 136, ${a})`;
      ctx.lineWidth = s.big ? 1.6 : 1.2;
      ctx.beginPath(); ctx.arc(s.x + ox, s.y + oy, s.r, 0, Math.PI * 2); ctx.stroke();
      if (s.label) {
        ctx.fillStyle = `rgba(160, 255, 244, ${a})`;
        ctx.fillText("+1/3", s.x + ox + 8, s.y + oy - 10 - (1 - s.life) * 26);
      }
    });
    if (visible && !REDUCED) requestAnimationFrame(frame);
  }

  resize();
  addEventListener("resize", () => { resize(); if (REDUCED) frame(performance.now()); });
  hero.addEventListener("pointermove", e => {
    const r = hero.getBoundingClientRect();
    mouse.tx = (e.clientX - r.left) / r.width - .5; mouse.ty = (e.clientY - r.top) / r.height - .5;
  });
  if (REDUCED) { paths.forEach(p => { p.d = p.total * .8; }); frame(performance.now()); return; }
  new IntersectionObserver(([e]) => {
    const was = visible; visible = e.isIntersecting;
    if (visible && !was) { last = 0; requestAnimationFrame(frame); }
  }).observe(hero);
  requestAnimationFrame(frame);
}

/* ---------- credit, following the paper (c = 0.5, γ = 1) ---------- */
const C = 0.5, K = 3, EPS = 1e-8;
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const near0 = x => Math.abs(x) < 1e-7;
const fmt = x => near0(x) ? "0" : (x > 0 ? "+" : "−") + Math.abs(x).toFixed(2);
const kind = x => near0(x) ? "zero" : x > 0 ? "pos" : "neg";

function standardize(xs) {
  const m = mean(xs);
  const sd = Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / (xs.length - 1));
  return xs.map(x => sd < EPS ? 0 : (x - m) / (sd + EPS));
}

// passed[t] = checks passing after turn t. Returns per-turn GRPO and ProCredit credit.
function credit(group) {
  const scored = group.map(passed => {
    const T = passed.length, fin = passed[T - 1], R = fin === K ? 1 : 0;
    const G = passed.map((_, t) => R + C * (fin - (t ? passed[t - 1] : 0)) / K);
    return { R, S: R + C * fin / K, G };
  });
  const traj = standardize(scored.map(a => a.S)), grpo = standardize(scored.map(a => a.R));
  const mG = mean(scored.flatMap(a => a.G));
  return scored.map((a, i) => ({ R: a.R, grpo: a.G.map(() => grpo[i]), pro: a.G.map(g => traj[i] + g - mG) }));
}

/* ---------- replay cases ---------- */
const ALARMS = [{ id: 1, label: "Wake-up", time: "07:00", on: true }, { id: 2, label: "Gym", time: "18:30", on: true }, { id: 3, label: "Work", time: "08:30", on: true }];
const CHECK_NAMES = ["Wake-up alarm at 06:00", "Gym alarm off", "Work alarm off"];
const checksOf = s => [s[0].time === "06:00" && s[0].on, !s[1].on, !s[2].on];
const LIST_OUT = `[{"id": 1, "label": "Wake-up", "time": "07:00", "enabled": true},
 {"id": 2, "label": "Gym", "time": "18:30", "enabled": true},
 {"id": 3, "label": "Work", "time": "08:30", "enabled": true}]`;
const T = {
  read: { short: "Read alarms", think: "Before changing anything, I should see which alarms exist and their ids.", code: `alarms = apis.phone.show_alarms(access_token=token)\nprint(alarms)`, out: LIST_OUT, read: true },
  wake: { short: "Wake-up → 06:00", think: "Wake-up is alarm 1. Move it from 07:00 to 06:00.", code: `apis.phone.update_alarm(access_token=token,\n    alarm_id=1, time="06:00")`, out: `{"message": "Alarm 1 updated."}`, op: s => { s[0].time = "06:00"; }, row: 0 },
  gym: { short: "Gym off", think: "Gym is alarm 2. Turn it off.", code: `apis.phone.update_alarm(access_token=token,\n    alarm_id=2, enabled=False)`, out: `{"message": "Alarm 2 updated."}`, op: s => { s[1].on = false; }, row: 1 },
  work: { short: "Work off", think: "Work is alarm 3. Turn that off as well.", code: `apis.phone.update_alarm(access_token=token,\n    alarm_id=3, enabled=False)`, out: `{"message": "Alarm 3 updated."}`, op: s => { s[2].on = false; }, row: 2 },
  slip: { short: "Disables Wake-up", think: "Next is Gym. I'll turn off alarm 1.", code: `apis.phone.update_alarm(access_token=token,\n    alarm_id=1, enabled=False)`, out: `{"message": "Alarm 1 updated.",\n "alarm": {"label": "Wake-up", "enabled": false}}`, op: s => { s[0].on = false; }, row: 0, slip: true },
  fix: { short: "Wake-up back on", think: "Alarm 1 is Wake-up, not Gym. That was the wrong id. Turn Wake-up back on.", code: `apis.phone.update_alarm(access_token=token,\n    alarm_id=1, enabled=True)`, out: `{"message": "Alarm 1 updated."}`, op: s => { s[0].on = true; }, row: 0 },
  gym2: { short: "Gym off", think: "Now the actual Gym alarm, id 2.", code: `apis.phone.update_alarm(access_token=token,\n    alarm_id=2, enabled=False)`, out: `{"message": "Alarm 2 updated."}`, op: s => { s[1].on = false; }, row: 1 },
  doneEarly: { short: "Declare done", think: "Wake-up is at 06:00 and Gym is off. That covers the request, so I'll finish.", code: `apis.supervisor.complete_task()`, out: `Task marked complete.`, done: true },
  done: { short: "Declare done", think: "All three changes are in place. Finish the task.", code: `apis.supervisor.complete_task()`, out: `Task marked complete.`, done: true }
};
const CASES = [
  {
    name: "Same failure, different progress",
    sub: "Both attempts in the group fail. GRPO sees two zeros; the checks see two changes this attempt got right.",
    request: `Move my wake-up alarm from <mark>07:00 to 06:00</mark>, and turn off <mark>Gym</mark> and <mark>Work</mark>.`,
    turns: [T.read, T.wake, T.gym, T.doneEarly],
    other: { label: "Other attempt · reads the list, then stops", turns: ["Read alarms", "Declare done"], passed: [0, 0] },
    verdict: { grpo: "No signal. Both attempts failed, so every turn gets 0.", pro: "Credit follows the checks. The turns that made progress rank highest." },
    endNote: "Work is still on. The outcome is a failure, but two of three checks pass."
  },
  {
    name: "A success with a slip",
    sub: "This attempt succeeds after undoing its own mistake. The group’s other attempt is the partial failure from case 01.",
    request: `Move my wake-up alarm from <mark>07:00 to 06:00</mark>, and turn off <mark>Gym</mark> and <mark>Work</mark>.`,
    turns: [T.read, T.wake, T.work, T.slip, T.fix, T.gym2, T.done],
    other: { label: "Other attempt · forgets Work", turns: ["Read alarms", "Wake-up → 06:00", "Gym off", "Declare done"], passed: [0, 1, 2, 2] },
    verdict: { grpo: "Same credit for every turn, the slip included.", pro: "The success still ranks first, and the slip earns less than its fix." },
    endNote: "All three checks pass. The slip lowered progress for one turn, and the checks saw it."
  }
];

/* ---------- replay player ---------- */
const CANCEL = Symbol("cancel");
const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const hl = src => esc(src).replace(/("[^"\n]*"?)|\b(True|False|print|apis)\b|\b(\d+)\b/g, (m, s, k, n) => s ? `<span class="s">${s}</span>` : k ? `<span class="k">${k}</span>` : `<span class="n">${n}</span>`);

class Player {
  constructor() {
    this.tok = 0; this.paused = false; this.visible = false; this.speed = 1.6; this.caseIdx = 0;
    this.els = { convo: $("convo"), tracker: $("tracker"), list: $("app-list"), checks: $("checks"), stair: $("stair"), scoring: $("scoring"),
      status: $("agent-status"), phi: $("phi-status"), toast: $("toast"), caption: $("phone-caption"), count: $("turn-count"), tl: $("timeline-bar"), play: $("btn-play") };
    this.els.stair.removeAttribute("preserveAspectRatio");
    $("btn-play").addEventListener("click", () => this.ended ? this.start(this.caseIdx) : this.setPaused(!this.paused));
    $("btn-replay").addEventListener("click", () => this.start(this.caseIdx));
    $("sel-speed").addEventListener("change", e => { this.speed = +e.target.value; });
    document.querySelectorAll(".case-tabs button").forEach(b => b.addEventListener("click", () => this.start(+b.dataset.case)));
    new IntersectionObserver(([e]) => {
      // on phones the stage is taller than the screen, so also count "fills most of the screen"
      const fill = e.intersectionRect.height / innerHeight;
      this.visible = e.isIntersecting && (e.intersectionRatio >= .18 || fill >= .4);
      if (!this.started && (e.intersectionRatio >= .5 || fill >= .6)) { this.started = true; this.start(0); }
    }, { threshold: Array.from({ length: 21 }, (_, i) => i / 20) }).observe($("stage"));
    this.load(0);
  }

  setPaused(p) {
    this.paused = p;
    this.els.play.classList.toggle("paused", p);
    this.els.play.querySelector("span").textContent = p ? "Play" : "Pause";
    this.els.play.setAttribute("aria-label", p ? "Play" : "Pause");
  }

  wait(ms) {
    const tok = this.tok;
    return new Promise((res, rej) => {
      let left = ms, last = performance.now();
      const tick = now => {
        if (tok !== this.tok) return rej(CANCEL);
        if (!this.paused && this.visible && !document.hidden) left -= (now - last) * this.speed;
        last = now;
        left <= 0 ? res() : requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
  }

  async type(el, text, perChar, render = t => esc(t)) {
    el.classList.add("caret");
    for (let i = 1; i <= text.length; i += 2) { el.innerHTML = render(text.slice(0, i)); this.follow(); await this.wait(perChar * 2); }
    el.innerHTML = render(text);
    el.classList.remove("caret");
  }

  follow() { const c = this.els.convo; c.scrollTop = c.scrollHeight; }

  load(idx) {
    const cs = CASES[idx], E = this.els;
    this.caseIdx = idx; this.cs = cs;
    this.state = ALARMS.map(a => ({ ...a }));
    this.passed = []; this.prevLen = 0;
    document.querySelectorAll(".case-tabs button").forEach(b => b.setAttribute("aria-selected", String(+b.dataset.case === idx)));
    $("case-no").textContent = String(idx + 1).padStart(2, "0");
    $("case-name").textContent = cs.name; $("case-sub").textContent = cs.sub; $("case-request").innerHTML = cs.request;
    E.tracker.innerHTML = cs.turns.map((t, i) => `<li><b>${i + 1}</b><span>${t.short}</span></li>`).join("");
    E.convo.innerHTML = `<p class="convo-empty">The replay starts when this panel is on screen.</p>`;
    E.checks.innerHTML = CHECK_NAMES.map((n, i) => `<li class="check" style="--i:${i}"><span class="ic"><svg viewBox="0 0 12 12"><path d="M2 6.5l2.6 2.5L10 3.5"/></svg></span>${n}</li>`).join("");
    E.scoring.hidden = true; E.toast.className = "toast";
    E.caption.textContent = "Device state updates after each tool return.";
    E.status.textContent = "Ready"; E.status.classList.remove("busy");
    this.renderPhone(); this.renderChecks([false, false, false]); this.renderStair();
    this.progress(0);
  }

  progress(f) {
    this.els.tl.style.transform = `scaleX(${clamp(f, 0, 1)})`;
    this.els.count.textContent = `${Math.min(this.passed.length, this.cs.turns.length)} / ${this.cs.turns.length}`;
  }

  renderPhone(row, cls) {
    this.els.list.innerHTML = this.state.map((a, i) => `<li class="app-row ${a.on ? "" : "off"} ${i === row ? cls : ""}"><div><span class="lbl">${a.label}</span><span class="tm">${i === row && cls !== "reading" ? `<span class="flip">${a.time}</span>` : a.time}</span></div><span class="tgl" aria-label="${a.on ? "on" : "off"}"></span></li>`).join("");
  }

  renderChecks(ok, lost = []) {
    [...this.els.checks.children].forEach((li, i) => { li.classList.toggle("pass", ok[i]); li.classList.toggle("lost", lost.includes(i)); });
    const n = ok.filter(Boolean).length;
    this.els.phi.textContent = `Φ = ${n}/${K}`;
  }

  renderStair() {
    const n = this.cs.turns.length, P = [0, ...this.passed];
    const L = 30, R = 312, Tp = 14, B = 96, X = t => L + (R - L) * t / n, Y = p => B - (B - Tp) * p / K;
    let d = `M${X(0)},${Y(0)}`;
    for (let t = 1; t < P.length; t++) d += ` H${X(t)} V${Y(P[t])}`;
    const grid = [0, 1, 2, 3].map(k => `<line class="grid" x1="${L}" x2="${R}" y1="${Y(k)}" y2="${Y(k)}"/><text class="axis-t" x="${L - 8}" y="${Y(k) + 3.5}" text-anchor="end">${["0", "⅓", "⅔", "1"][k]}</text>`).join("");
    const ticks = Array.from({ length: n }, (_, i) => `<text class="axis-t" x="${X(i + 1)}" y="${B + 16}" text-anchor="middle">${i + 1}</text>`).join("");
    const pts = P.slice(1).map((p, i) => {
      const r = C * (p - P[i]) / K, isNew = i === P.length - 2;
      const lbl = `<text class="rt ${kind(r)}" x="${X(i + 1) - 4}" y="${Y(p) - 7}" text-anchor="end">${near0(r) ? "0" : fmt(r)}</text>`;
      return `<circle class="pt ${r < 0 ? "down" : ""} ${isNew ? "new" : ""}" cx="${X(i + 1)}" cy="${Y(p)}" r="3.6"/>${lbl}`;
    }).join("");
    const area = P.length > 1 ? `<path class="area" d="${d} V${B} H${X(0)} Z"/>` : "";
    this.els.stair.innerHTML = `<defs><linearGradient id="stair-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#50d1c4" stop-opacity=".28"/><stop offset="1" stop-color="#50d1c4" stop-opacity="0"/></linearGradient></defs>${grid}${ticks}<text class="axis-t" x="${L - 8}" y="${B + 16}" text-anchor="end">turn</text>${area}<path class="path" id="stair-path" d="${d}"/>${pts}`;
    const path = $("stair-path"), len = path.getTotalLength();
    if (!REDUCED && len > this.prevLen) {
      path.style.strokeDasharray = `${len}`; path.style.strokeDashoffset = `${len - this.prevLen}`;
      path.getBoundingClientRect();
      path.style.transition = "stroke-dashoffset .7s cubic-bezier(.3,.7,.2,1)"; path.style.strokeDashoffset = "0";
    }
    this.prevLen = len;
  }

  toast(text, cls) { const t = this.els.toast; t.textContent = text; t.className = `toast show ${cls || ""}`; }

  async start(idx) {
    ++this.tok;
    this.ended = false;
    this.setPaused(false);
    this.load(idx);
    try {
      await this.run();
      this.ended = true;
      this.setPaused(true); // stopped; Play or Replay starts over
    } catch (e) { if (e !== CANCEL) throw e; }
  }

  async run() {
    const E = this.els, cs = this.cs, n = cs.turns.length, lis = [...E.tracker.children];
    E.convo.innerHTML = "";
    await this.wait(400);
    for (let i = 0; i < n; i++) {
      const t = cs.turns[i];
      lis.forEach((li, j) => li.classList.toggle("active", j === i));
      E.status.textContent = "Agent is reasoning"; E.status.classList.add("busy");

      const m = document.createElement("div");
      m.className = "msg";
      m.innerHTML = `<div class="msg-head"><strong>Agent</strong><span class="msg-turn">turn ${i + 1}</span></div><p class="think-label">Thinking</p><p class="think"></p><pre class="code"></pre>`;
      E.convo.appendChild(m); this.follow();
      await this.type(m.querySelector(".think"), t.think, 24);
      await this.wait(250);
      E.status.textContent = "Writing code";
      const code = m.querySelector(".code"); code.classList.add("streaming");
      await this.type(code, t.code, 11, hl);
      code.classList.remove("streaming");
      E.status.textContent = "Executing";
      await this.wait(550);

      // environment returns; the device changes
      const before = checksOf(this.state);
      if (t.op) t.op(this.state);
      const after = checksOf(this.state);
      const gained = after.some((v, k) => v && !before[k]), lostIdx = after.map((v, k) => !v && before[k] ? k : -1).filter(k => k >= 0);
      const env = document.createElement("div");
      env.className = `msg env-msg ${t.done ? (after.every(Boolean) ? "done-msg" : "fail-msg") : ""}`;
      env.innerHTML = `<div class="msg-head"><strong>Environment</strong><span class="msg-turn">tool return</span></div><pre class="env-out">${esc(t.out)}</pre>`;
      E.convo.appendChild(env); this.follow();

      if (t.read) {
        this.state.forEach((_, k) => setTimeout(() => this.renderPhone(k, "reading"), k * 180));
        E.caption.textContent = "A query reads the state and changes nothing.";
      } else if (t.row !== undefined) {
        this.renderPhone(t.row, lostIdx.length ? "flash-bad" : gained ? "flash" : "");
        E.caption.textContent = lostIdx.length ? `${ALARMS[t.row].label} was turned off by mistake.` : `${ALARMS[t.row].label} alarm updated.`;
      }
      if (t.done) this.toast("Task marked complete", after.every(Boolean) ? "ok" : "bad");

      // rerun the acceptance checks on the new state
      await this.wait(350);
      E.status.textContent = "Rerunning checks";
      E.checks.classList.remove("scan"); void E.checks.offsetWidth; E.checks.classList.add("scan");
      await this.wait(550);
      this.renderChecks(after, lostIdx);
      this.passed.push(after.filter(Boolean).length);
      this.renderStair();
      lis[i].classList.remove("active"); lis[i].classList.add("done"); if (t.slip) lis[i].classList.add("slip");
      this.progress((i + 1) / (n + 1));
      await this.wait(t.done ? 500 : 900);
    }

    // the attempt ends: outcome, then credit for the whole group
    const ok = this.passed.at(-1) === K;
    E.status.classList.remove("busy");
    E.status.textContent = ok ? "Succeeded" : "Failed";
    const ev = document.createElement("div");
    ev.className = `msg env-msg ${ok ? "done-msg" : "fail-msg"}`;
    ev.innerHTML = `<div class="msg-head"><strong>Evaluation</strong><span class="msg-turn">end of attempt</span></div><p class="env-out">${this.passed.at(-1)} of ${K} checks pass → R = ${ok ? 1 : 0} (${ok ? "success" : "failure"}). ${cs.endNote}</p>`;
    E.convo.appendChild(ev); this.follow();
    E.caption.textContent = cs.endNote;
    await this.wait(900);
    await this.score();
    this.progress(1);
  }

  async score() {
    const E = this.els, cs = this.cs;
    const [mine, other] = credit([this.passed, cs.other.passed]);
    const SCALE = 1.7;
    const bar = v => {
      if (near0(v)) return `<span class="cbar zero"><b>0</b></span>`;
      const w = Math.min(Math.abs(v) / SCALE, 1) * 48;
      const pos = v > 0;
      return `<span class="cbar ${kind(v)}"><i style="left:${pos ? 50 : 50 - w}%;width:${w}%"></i><b style="${pos ? "right:calc(50% + 5px)" : "left:calc(50% + 5px)"}">${fmt(v)}</b></span>`;
    };
    const rows = (turns, c, slipIdx) => turns.map((name, i) => `<div class="cr ${slipIdx.includes(i) ? "slip" : ""}"><span class="act">${i + 1}. ${name}</span>${bar(c.grpo[i])}${bar(c.pro[i])}</div>`).join("");
    const okMine = mine.R === 1, okOther = other.R === 1;
    const slips = cs.turns.map((t, i) => t.slip ? i : -1).filter(i => i >= 0);
    E.scoring.hidden = false;
    E.scoring.querySelector("#v-grpo strong").textContent = cs.verdict.grpo;
    E.scoring.querySelector("#v-pro strong").textContent = cs.verdict.pro;
    const box = $("credit-rows");
    box.classList.toggle("grpo-flat", mine.grpo.every(near0));
    box.innerHTML = `<div class="cr cr-head"><span>Turn</span><span>GRPO</span><span>ProCredit</span></div>` +
      `<p class="cr-group">This attempt <em class="${okMine ? "ok" : "fail"}">${okMine ? "success" : "failure"} · Φ = ${this.passed.at(-1)}/3</em></p>` +
      rows(cs.turns.map(t => t.short), mine, slips) +
      `<p class="cr-group">${cs.other.label} <em class="${okOther ? "ok" : "fail"}">${okOther ? "success" : "failure"} · Φ = ${cs.other.passed.at(-1)}/3</em></p>` +
      rows(cs.other.turns, other, []);
    for (const r of box.querySelectorAll(".cr:not(.cr-head)")) { await this.wait(140); r.classList.add("show"); }
  }
}

splitTitle();
chrome();
heroCanvas();
new Player();
