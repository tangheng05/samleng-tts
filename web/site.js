(() => {
  const body = document.body;
  const km = body.dataset.lang === "km";
  const t = (p) => (km ? p.km : p.en);
  const $ = (id) => document.getElementById(id);
  const TZ = "Asia/Phnom_Penh";
  const TEACHER_CER = 3.1;

  // ---------- Theme ----------

  const dark = matchMedia("(prefers-color-scheme: dark)");
  let choice = null;
  try { choice = localStorage.getItem("samleng-theme"); } catch (e) {}
  const applyTheme = () => { body.dataset.theme = choice || (dark.matches ? "night" : "paper"); };
  dark.addEventListener("change", applyTheme);
  $("theme-btn").addEventListener("click", () => {
    choice = body.dataset.theme === "night" ? "paper" : "night";
    try { localStorage.setItem("samleng-theme", choice); } catch (e) {}
    applyTheme();
  });

  // ---------- Reveal on scroll ----------

  const reveal = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("seen"); reveal.unobserve(e.target); } });
  }, { rootMargin: "0px 0px -8% 0px" });
  document.querySelectorAll(".rv, .mask, .results").forEach((el) => reveal.observe(el));

  // ---------- 01 Listen ----------

  const SYSTEMS = {
    base: { name: { km: "ម៉ូដែលដើម", en: "Base model" }, sub: { km: "មុនពេលយើងហ្វឹកហាត់", en: "Before our training" } },
    samleng: { name: { km: "Samleng v1", en: "Samleng v1" }, sub: { km: "36 លានប៉ារ៉ាម៉ែត្រ · ដំណើរការលើ CPU", en: "36M parameters · runs on a CPU" } },
    teacher: { name: { km: "គ្រូ · VoxCPM2", en: "Teacher · VoxCPM2" }, sub: { km: "2 ពាន់លានប៉ារ៉ាម៉ែត្រ · ត្រូវការ GPU", en: "2B parameters · needs a GPU" } },
  };
  const SAMPLES = {
    "pure-34": { title: { km: "សុភាសិតខ្មែរ", en: "A Khmer proverb" }, en: "“When the water rises, fish eat ants; when it falls, ants eat fish.”" },
    "numbers-49": { title: { km: "ភាគរយ និងទសភាគ", en: "Percent and decimals" }, en: "“This loan charges 1.5% interest per month.”" },
    "names-38": { title: { km: "ឈ្មោះទីកន្លែង", en: "Place names" }, en: "“Kratie is where you can see freshwater dolphins in the Mekong.”" },
    "mixed-67": { title: { km: "ខ្មែរលាយអង់គ្លេស", en: "Khmer mixed with English" }, en: "“Please copy this link and paste it into the browser.”" },
    "numbers-19": { title: { km: "ចំនួនធំ", en: "Large numbers" }, en: "“This company has 1,250 employees.”" },
  };
  const CATEGORY = {
    pure: { km: "ខ្មែរសុទ្ធ", en: "Pure Khmer" }, numbers: { km: "លេខ", en: "Numbers" },
    names: { km: "ឈ្មោះ", en: "Names" }, mixed: { km: "ខ្មែរ–អង់គ្លេស", en: "Khmer–English" },
  };
  const clock = (sec) => `${Math.floor(sec / 60)}:${String(Math.round(sec % 60)).padStart(2, "0")}`;
  const ICONS = '<svg class="i-play" viewBox="0 0 16 16"><path d="M4 2.5v11l9.5-5.5z" fill="currentColor"/></svg><svg class="i-pause" viewBox="0 0 16 16"><path d="M4 2.5h3v11H4zm5 0h3v11H9z" fill="currentColor"/></svg>';
  const SCORE = { km: "ពិន្ទុអ្នកស្ដាប់", en: "listener score" };

  let audios = [];
  const stopAll = () => audios.forEach((a) => { a.pause(); a.currentTime = 0; });

  const showSample = (s, i) => {
    stopAll();
    document.querySelectorAll(".pick").forEach((b) => b.setAttribute("aria-pressed", b.dataset.i === String(i)));
    $("say").textContent = s.text;
    $("say-en").textContent = km ? "" : SAMPLES[s.id].en;
    const players = $("players");
    players.innerHTML = "";
    const longest = Math.max(...Object.values(s.clips).map((c) => c.seconds));
    audios = Object.entries(s.clips).map(([key, clip]) => {
      const sys = SYSTEMS[key];
      const row = document.createElement("div");
      row.className = "player" + (key === "samleng" ? " ours" : "");
      const bars = clip.peaks.map((v) => `<i style="height:${Math.max(6, v)}%"></i>`).join("");
      row.innerHTML = `<button type="button" class="play" aria-label="${t(sys.name)}">${ICONS}</button>
        <div class="who"><b>${t(sys.name)}</b></div>
        <div class="wave-box"><div class="wf" style="width:${(100 * clip.seconds) / longest}%">${bars}</div></div>
        <span class="len num">${clock(clip.seconds)}</span>
        <span class="score">${clip.score == null ? "" : `<b class="num" title="${t(SCORE)}">${clip.score}<small> / 5</small></b>`}</span>`;
      const audio = new Audio(`/audio/${key}/${s.id}.m4a`);
      audio.preload = "none";
      solo(audio);
      const wf = row.querySelector(".wf");
      const cells = [...wf.children];
      const paint = (frac) => cells.forEach((c, n) => c.classList.toggle("done", n < frac * cells.length));
      audio.addEventListener("timeupdate", () => paint(audio.currentTime / (audio.duration || clip.seconds)));
      audio.addEventListener("play", () => row.classList.add("on"));
      audio.addEventListener("pause", () => row.classList.remove("on"));
      audio.addEventListener("ended", () => { row.classList.remove("on"); paint(0); });
      const start = () => {
        audios.forEach((a) => { if (a !== audio) { a.pause(); a.currentTime = 0; } });
        document.querySelectorAll(".wf i.done").forEach((c) => { if (c.parentNode !== wf) c.classList.remove("done"); });
        audio.play();
      };
      row.querySelector(".play").addEventListener("click", () => (audio.paused ? start() : audio.pause()));
      wf.addEventListener("click", (e) => {
        const box = wf.getBoundingClientRect();
        const frac = (e.clientX - box.left) / box.width;
        const seek = () => { audio.currentTime = frac * audio.duration; };
        if (audio.readyState > 0) seek(); else audio.addEventListener("loadedmetadata", seek, { once: true });
        start();
      });
      players.appendChild(row);
      return audio;
    });
  };

  fetch("/data/samples.json").then((r) => r.json()).then((samples) => {
    $("picks").innerHTML = samples.map((s, i) => `<li><button type="button" class="pick" data-i="${i}" aria-pressed="false">
      <span class="t-h3">${t(SAMPLES[s.id].title)}</span><span class="bar"></span></button></li>`).join("");
    document.querySelectorAll(".pick").forEach((b) => b.addEventListener("click", () => showSample(samples[b.dataset.i], b.dataset.i)));
    showSample(samples[0], 0);
    heroPlayer(samples[0]);
    startField();
  });

  const everyAudio = new Set();
  const solo = (audio) => {
    everyAudio.add(audio);
    audio.addEventListener("play", () => everyAudio.forEach((a) => { if (a !== audio) a.pause(); }));
  };
  let heroAudio = null;
  const heroPlayer = (s) => {
    const clip = s.clips.samleng, root = $("hp"), wf = $("hp-wf");
    $("hp-say").textContent = s.text;
    wf.innerHTML = clip.peaks.map((v) => `<i style="height:${Math.max(6, v)}%"></i>`).join("");
    const cells = [...wf.children];
    const audio = new Audio(`/audio/samleng/${s.id}.m4a`);
    audio.preload = "none";
    solo(audio);
    heroAudio = audio;
    const paint = (frac) => cells.forEach((c, n) => c.classList.toggle("done", n < frac * cells.length));
    audio.addEventListener("timeupdate", () => paint(audio.currentTime / (audio.duration || clip.seconds)));
    audio.addEventListener("play", () => root.classList.add("on"));
    audio.addEventListener("pause", () => root.classList.remove("on"));
    audio.addEventListener("ended", () => { root.classList.remove("on"); paint(0); });
    root.querySelector(".play").addEventListener("click", () => (audio.paused ? audio.play() : audio.pause()));
    wf.addEventListener("click", () => audio.play());
  };

  // ---------- Voice field: the Khmer word as dots that react to the mouse and the voice ----------

  const startField = async () => {
    const field = $("field");
    if (!field || getComputedStyle(field).display === "none") return;
    try { await document.fonts.load('400 200px "Koulen"', "សំឡេង"); } catch (e) {}
    const ctx = field.getContext("2d");
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const WORD = "សំឡេង";
    let W = 0, H = 0, pts = [], xmin = 0, xmax = 1, colors = {};
    let mouse = { x: -1e4, y: -1e4 }, level = 0, prog = 0, analyser = null, buf = null, running = true;

    const readColors = () => {
      const cs = getComputedStyle(body);
      colors = { fg: cs.getPropertyValue("--fg").trim(), accent: cs.getPropertyValue("--accent").trim() };
    };
    const build = () => {
      const dpr = Math.min(2, devicePixelRatio || 1);
      W = field.clientWidth; H = field.clientHeight;
      field.width = W * dpr; field.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const off = document.createElement("canvas");
      off.width = W; off.height = H;
      const o = off.getContext("2d");
      let size = H * 0.74;
      o.font = `400 ${size}px Koulen`;
      const w0 = o.measureText(WORD).width;
      if (w0 > W * 0.96) size *= (W * 0.96) / w0;
      o.font = `400 ${size}px Koulen`;
      const m = o.measureText(WORD);
      const y = (H - (m.actualBoundingBoxAscent + m.actualBoundingBoxDescent)) / 2 + m.actualBoundingBoxAscent;
      o.fillText(WORD, m.actualBoundingBoxLeft + 2, y);
      const data = o.getImageData(0, 0, W, H).data;
      const step = W > 520 ? 5 : 4;
      pts = [];
      for (let yy = 0; yy < H; yy += step)
        for (let xx = 0; xx < W; xx += step)
          if (data[(yy * W + xx) * 4 + 3] > 140) pts.push({ hx: xx, hy: yy, x: xx, y: yy, vx: 0, vy: 0, j: Math.random() * 6.28 });
      xmin = Math.min(...pts.map((p) => p.hx)); xmax = Math.max(...pts.map((p) => p.hx));
      readColors();
    };

    const listen = () => {
      if (analyser || !heroAudio) return;
      try {
        const ac = new (window.AudioContext || window.webkitAudioContext)();
        const src = ac.createMediaElementSource(heroAudio);
        analyser = ac.createAnalyser();
        analyser.fftSize = 512;
        buf = new Uint8Array(analyser.fftSize);
        src.connect(analyser); analyser.connect(ac.destination);
        heroAudio.addEventListener("play", () => ac.resume());
      } catch (e) {}
    };
    if (heroAudio) heroAudio.addEventListener("play", listen);

    field.addEventListener("pointermove", (e) => { const r = field.getBoundingClientRect(); mouse = { x: e.clientX - r.left, y: e.clientY - r.top }; });
    field.addEventListener("pointerleave", () => { mouse = { x: -1e4, y: -1e4 }; });
    field.addEventListener("click", () => { if (heroAudio) (heroAudio.paused ? heroAudio.play() : heroAudio.pause()); });

    const R = 80, R2 = R * R;
    const frame = (t) => {
      if (!running) return;
      let amp = 0;
      if (analyser && heroAudio && !heroAudio.paused) {
        analyser.getByteTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
        amp = Math.min(1, Math.sqrt(sum / buf.length) * 4);
      }
      level += (amp - level) * 0.25;
      if (heroAudio) prog = heroAudio.ended || !heroAudio.duration ? (heroAudio.ended ? 0 : prog) : heroAudio.currentTime / heroAudio.duration;
      const cut = xmin + prog * (xmax - xmin);
      ctx.clearRect(0, 0, W, H);
      const lit = new Path2D(), dim = new Path2D();
      for (const p of pts) {
        if (!still) {
          const dx = p.x - mouse.x, dy = p.y - mouse.y, d2 = dx * dx + dy * dy;
          if (d2 < R2) { const d = Math.sqrt(d2) || 1, f = (1 - d / R) * 2.6; p.vx += (dx / d) * f; p.vy += (dy / d) * f; }
          const breath = Math.sin(t * 0.0016 + p.hx * 0.018) * 0.9;
          const wob = level * 16 * Math.sin(p.hx * 0.05 + t * 0.014 + p.j);
          p.vx += (p.hx - p.x) * 0.07; p.vy += (p.hy + breath + wob - p.y) * 0.07;
          p.vx *= 0.8; p.vy *= 0.8; p.x += p.vx; p.y += p.vy;
        }
        const on = prog > 0 && p.hx <= cut;
        const r = on ? 1.7 + level * 0.9 : 1.45;
        const path = on ? lit : dim;
        path.moveTo(p.x + r, p.y); path.arc(p.x, p.y, r, 0, 6.2832);
      }
      ctx.globalAlpha = 0.42; ctx.fillStyle = colors.fg; ctx.fill(dim);
      ctx.globalAlpha = 1; ctx.fillStyle = colors.accent; ctx.fill(lit);
      requestAnimationFrame(frame);
    };

    build();
    new IntersectionObserver(([e]) => { const was = running; running = e.isIntersecting; if (running && !was) requestAnimationFrame(frame); }).observe(field);
    new MutationObserver(readColors).observe(body, { attributes: true, attributeFilter: ["data-theme"] });
    let rt;
    addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { if (getComputedStyle(field).display !== "none") build(); }, 150); });
    requestAnimationFrame(frame);
  };

  // ---------- Live training card ----------

  const STATES = {
    training: { km: "កំពុងហ្វឹកហាត់", en: "Training" },
    paused: { km: "ផ្អាកដើម្បីជួសជុល", en: "Paused for a fix" },
    idle: { km: "សម្រាករវាងការហ្វឹកហាត់", en: "Between runs" },
    offline: { km: "គ្មានសញ្ញាថ្មី", en: "No recent signal" },
  };
  const RUNS = {
    distill_v2_clear: {
      km: "កំពុងបង្រៀនម៉ូដែលខ្នាតតូចឱ្យនិយាយសំឡេងថ្មីដែលច្បាស់ជាងមុន ពីសំឡេងគ្រូរយៈពេល ៨.៥ ម៉ោង។",
      en: "Teaching the small model a new, clearer voice from 8.5 hours of teacher audio.",
    },
  };
  const fmt = (n) => n.toLocaleString("en-US");
  const ago = (iso) => {
    const min = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 60000));
    if (min < 1) return km ? "ទើបតែធ្វើបច្ចុប្បន្នភាព" : "Updated just now";
    if (min < 60) return km ? `ធ្វើបច្ចុប្បន្នភាព ${min} នាទីមុន` : `Updated ${min} min ago`;
    const h = Math.round(min / 60);
    return km ? `ធ្វើបច្ចុប្បន្នភាព ${h} ម៉ោងមុន` : `Updated ${h} h ago`;
  };
  const finish = (iso) => {
    const d = new Date(iso);
    const day = (x) => x.toLocaleDateString("en-CA", { timeZone: TZ });
    const time = d.toLocaleTimeString("en-GB", { timeZone: TZ, hour: "2-digit", minute: "2-digit" });
    if (day(d) === day(new Date())) return time;
    const tomorrow = new Date(Date.now() + 864e5);
    if (day(d) === day(tomorrow)) return km ? `ស្អែក ${time}` : `Tomorrow ${time}`;
    return d.toLocaleDateString(km ? "km-KH" : "en-GB", { timeZone: TZ, day: "numeric", month: "short" }) + " " + time;
  };

  const LABELS = { base: { km: "ម៉ូដែលដើម", en: "Base model" }, teacher: { km: "គ្រូ", en: "Teacher" }, step: { km: "ជំហាន", en: "step" } };
  const drawEpochs = (s) => {
    const box = $("epochs");
    if (s.epoch == null) { box.innerHTML = ""; return; }
    const part = Math.round((100 * s.step_in_epoch) / s.steps_per_epoch);
    box.innerHTML = Array.from({ length: s.epochs }, (_, n) =>
      n < s.epoch ? '<i class="done"></i>' : n === s.epoch ? `<i class="cur"><b style="height:${part}%"></b></i>` : "<i></i>").join("");
  };
  const drawGauge = (s) => {
    const box = $("gauge");
    const last = s.scores[s.scores.length - 1];
    const vals = s.scores.map((p) => p.all).concat(s.base.all ?? TEACHER_CER);
    const hi = Math.max(...vals) + 1.5, lo = TEACHER_CER - 1.2;
    const at = (v) => (100 * (hi - v)) / (hi - lo);
    const ref = (v, cls, label) => `<span class="g-ref ${cls}" style="left:${at(v)}%"><i></i><b>${label}</b><small>${v.toFixed(1)}%</small></span>`;
    box.innerHTML = (s.base.all != null ? ref(s.base.all, "base", t(LABELS.base)) : "") + ref(TEACHER_CER, "teacher", t(LABELS.teacher))
      + s.scores.slice(0, -1).map((p) => `<span class="g-old" style="left:${at(p.all)}%"></span>`).join("")
      + (last ? `<span class="g-now" style="left:${at(last.all)}%"><b>${last.all.toFixed(1)}%</b><small>${t(LABELS.step)} ${fmt(last.step)}</small><i></i></span>` : "");
  };

  const SVGNS = "http://www.w3.org/2000/svg";
  const el = (name, attrs, text) => {
    const n = document.createElementNS(SVGNS, name);
    Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
    if (text != null) n.textContent = text;
    return n;
  };
  const drawHistory = (s) => {
    const svg = $("history");
    const W = svg.clientWidth || 900, H = svg.clientHeight || 340, L = 48, R = 16, T = 28, B = 44;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = "";
    const total = (s.step || 0) + (s.steps_left || 0) || 1;
    const vals = s.scores.map((p) => p.all).concat(s.base.all ?? 0);
    const top = Math.max(12, Math.ceil((Math.max(...vals) + 1.5) / 4) * 4);
    const x = (step) => L + ((W - L - R) * step) / total;
    const y = (v) => T + ((H - T - B) * (top - v)) / top;
    for (let v = 0; v <= top; v += 4) {
      svg.append(el("line", { class: "h-grid", x1: L, x2: W - R, y1: y(v), y2: y(v) }));
      svg.append(el("text", { class: "h-tick", x: L - 10, y: y(v) + 5, "text-anchor": "end" }, v + "%"));
    }
    const fracs = W < 560 ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1];
    fracs.forEach((f) => {
      const st = Math.round((total * f) / 1000) * 1000;
      const label = f === 1 ? `${fmt(total)} ${km ? "ជំហាន" : "steps"}` : fmt(st);
      svg.append(el("text", { class: "h-tick", x: x(f === 1 ? total : st), y: H - 14, "text-anchor": f === 0 ? "start" : f === 1 ? "end" : "middle" }, label));
    });
    const ref = (v, cls, label) => {
      svg.append(el("line", { class: `h-ref ${cls}`, x1: L, x2: W - R, y1: y(v), y2: y(v) }));
      svg.append(el("text", { class: `h-ref-label ${cls}`, x: W - R, y: y(v) - 9, "text-anchor": "end" }, `${label} ${v.toFixed(1)}%`));
    };
    if (s.base.all != null) ref(s.base.all, "base", t(LABELS.base));
    ref(TEACHER_CER, "teacher", t(LABELS.teacher));
    if (s.step) svg.append(el("line", { class: "h-now", x1: x(s.step), x2: x(s.step), y1: T, y2: H - B }));
    if (!s.scores.length) return;
    const pts = s.scores.map((p) => [x(p.step), y(p.all)]);
    svg.append(el("polyline", { class: "h-line", points: pts.map((p) => p.join(",")).join(" ") }));
    pts.forEach(([px, py], n) => {
      svg.append(el("circle", { class: "h-pt", cx: px, cy: py, r: 6 }));
      if (W >= 560 || n === pts.length - 1) svg.append(el("text", { class: "h-val", x: px, y: py + 28, "text-anchor": "middle" }, s.scores[n].all.toFixed(1) + "%"));
    });
  };

  const put = (key, text) => document.querySelectorAll(`[data-live="${key}"]`).forEach((n) => { n.textContent = text; });
  const renderLive = (s) => {
    const stale = Date.now() - Date.parse(s.updated_at) > 20 * 60000;
    const state = stale ? "offline" : s.state;
    document.querySelectorAll(".state").forEach((n) => { n.dataset.s = state; });
    put("state", t(STATES[state] || STATES.idle));
    put("updated", ago(s.updated_at));
    put("what", RUNS[s.run] ? t(RUNS[s.run]) : s.run || "");
    if (s.step != null) {
      put("step", fmt(s.step));
      put("total", fmt(s.step + (s.steps_left || 0)));
      document.querySelectorAll(".meter i").forEach((n) => { n.style.width = (100 * (s.progress || 0)).toFixed(1) + "%"; });
      put("epoch", `${s.epoch + 1} / ${s.epochs}`);
      put("speed", s.step_seconds ? `${s.step_seconds.toFixed(1)} s` : "–");
      put("finish", s.finish_at && state === "training" ? finish(s.finish_at) : "–");
    }
    put("best", s.best ? `${s.best.all.toFixed(1)}%` : "–");
    put("pct", s.progress != null ? Math.round(100 * s.progress) + "%" : "–");
    drawEpochs(s);
    drawGauge(s);
    drawHistory(s);
  };

  let last = null;
  const load = () => fetch(body.dataset.status, { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then((s) => { last = s; renderLive(s); })
    .catch(() => { if (last) renderLive(last); });
  addEventListener("resize", () => { if (last) drawHistory(last); });
  load();
  setInterval(load, 120000);

  // ---------- Phone: fold long parts ----------

  const phone = matchMedia("(max-width: 720px)");
  const measure = document.querySelector(".measure");
  if (measure && phone.matches) measure.removeAttribute("open");
  const more = $("log-more");
  if (more) more.addEventListener("click", () => { document.querySelector(".log").classList.add("all"); more.remove(); });

  // ---------- Fonts ready: no layout jump ----------

  document.fonts.ready.then(() => body.classList.add("ready"));
})();
