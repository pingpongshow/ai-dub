"use strict";
// Review screen: watch an episode, see what keeps the original and who speaks each
// line, and change it before the job translates and voices anything.

const RV_COLORS = ["#7d9bff", "#f0a35e", "#5fb3d9", "#e889b5", "#7fc8a9", "#c49bff", "#d9b44a", "#ff8a7a", "#62c3c3", "#b9a17a"];
const RV_KINDS = { song: "Song", opening: "Opening", ending: "Ending", credits: "Credits", ignore: "Don't dub" };
const RV_LANES = [["", 18], ["Voices", 46], ["Keep original", 26], ["Subtitles", 22], ["Speakers", 22], ["Your marks", 24]];
const RV_GAP = 6;
let RV = null;

const rvClone = (o) => JSON.parse(JSON.stringify(o));
function fmtT(t, tenths = true) {
  t = Math.max(0, t || 0);
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${tenths ? s.toFixed(1).padStart(4, "0") : String(Math.floor(s)).padStart(2, "0")}`;
}
function parseT(s) {
  const m = String(s).trim().match(/^(?:(\d+):)?(\d+(?:\.\d+)?)$/);
  return m ? Number(m[1] || 0) * 60 + Number(m[2]) : null;
}
const rvCss = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const overlap = (a0, a1, b0, b1) => Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));

// Model ---------------------------------------------------------------------------------------
function rvSpeaker(cid) { return RV.data.speakers.find((s) => s.id === cid); }
function rvColor(cid) {
  const i = RV.data.speakers.findIndex((s) => s.id === cid);
  return i < 0 ? "#9aa3b2" : RV_COLORS[i % RV_COLORS.length];
}
function lineWho(ln) {
  const d = RV.dec.lines[String(ln.id)];
  return d ? d.character : ln.character;
}
function whoLabel(cid) {
  if (cid === "own") return "Its own voice";
  if (cid && cid.startsWith("new:")) return `${cid.slice(4)} (new)`;
  const s = rvSpeaker(cid);
  return s ? s.name : cid || "nobody";
}
function linesIn(a, b) {
  return RV.data.lines.filter((l) => overlap(l.start, l.end, a, b) > 0.3 * Math.min(l.end - l.start, b - a));
}
function regionsNow() {
  const byId = new Map(RV.dec.regions.map((r) => [r.id, r]));
  const out = [];
  for (const r of RV.data.regions) {
    const o = byId.get(r.id);
    out.push(o ? { ...r, ...o, source: "ai", edited: o.start !== r.start || o.end !== r.end } : { ...r, action: "keep" });
  }
  for (const r of RV.dec.regions) if (r.source === "user") out.push(r);
  return out.sort((a, b) => a.start - b.start);
}
function regionAt(t, user) {
  return regionsNow().find((r) => (user ? r.source === "user" : r.source !== "user") && t >= r.start && t <= r.end);
}
function flaggedLine(id) { return RV.data.flags.some((f) => f.lines.includes(id) && f.kind === "lyrics"); }
function cueState(q) {
  const d = RV.dec.cues[q.key];
  if (d) return d === "keep" ? ["mine", "Keep (you)"] : ["mine", "Dub (you)"];
  if (q.kind === "sign") return ["sign", "Sign"];
  if (q.kind === "hi") return ["sign", "Sound"];
  const kept = regionsNow().find((r) => overlap(r.start, r.end, q.start, q.end) > 0.5 * (q.end - q.start));
  if (kept) return kept.action === "dub" ? ["dub", "Dub (you)"] : ["keep", RV_KINDS[kept.kind] || "Song"];
  if (q.kind === "song") return ["keep", "Song"];
  const ls = linesIn(q.start, q.end);
  if (ls.some((l) => flaggedLine(l.id))) return ["look", "Check"];
  if (ls.some((l) => l.kind === "song")) return ["keep", "Song"];
  if (ls.some((l) => l.kind === "speech")) return ["dub", "Dub"];
  return ["plain", ls.length ? "Kept" : "No line"];
}
function cueLine(q) { return linesIn(q.start, q.end).find((l) => l.kind === "speech" || l.kind === "song"); }
function setDirty() {
  RV.dirty = true;
  const s = $("#rvSave");
  if (s) s.textContent = "Save";
}
function decisionsOut() {
  const regs = RV.dec.regions.slice().sort((a, b) => (a.source === "hidden" ? 0 : 1) - (b.source === "hidden" ? 0 : 1));
  return { regions: regs, cues: RV.dec.cues, lines: RV.dec.lines };
}
function setRegion(r, patch) {
  const list = RV.dec.regions;
  if (r.source === "user") {
    Object.assign(list.find((x) => x.id === r.id), patch);
  } else {
    const ai = RV.data.regions.find((x) => x.id === r.id);
    let o = list.find((x) => x.id === r.id);
    if (!o) { o = { id: r.id, start: ai.start, end: ai.end, kind: ai.kind, action: "keep", source: "ai" }; list.push(o); }
    Object.assign(o, patch);
    const moved = o.start !== ai.start || o.end !== ai.end;
    RV.dec.regions = list.filter((x) => x.id !== r.id + "~was");
    if (moved && o.action === "keep") {
      // The part of Ai Dub's region you trimmed off is dubbed.
      RV.dec.regions.push({ id: r.id + "~was", start: ai.start, end: ai.end, kind: ai.kind, action: "dub", source: "hidden" });
    }
    if (!moved && o.action === "keep" && !o.show_rule && o.kind === ai.kind) RV.dec.regions = RV.dec.regions.filter((x) => x.id !== r.id);
  }
  setDirty();
}
function removeRegion(r) {
  if (r.source === "user") RV.dec.regions = RV.dec.regions.filter((x) => x.id !== r.id);
  else setRegion(r, { action: "dub", start: RV.data.regions.find((x) => x.id === r.id).start, end: RV.data.regions.find((x) => x.id === r.id).end });
  if (RV.sel && RV.sel.id === r.id) RV.sel = null;
  setDirty();
}
function addRegion(a, b, kind = "song") {
  if (b - a < 0.3) return null;
  const r = { id: `u-${Date.now()}`, start: +a.toFixed(2), end: +b.toFixed(2), kind, action: "keep", source: "user" };
  RV.dec.regions.push(r);
  setDirty();
  return r;
}

// Page --------------------------------------------------------------------------------------------
async function pageReview(id) {
  main.classList.add("wide");
  const data = await api(`/jobs/${id}/review`);
  if (!data.ready) {
    main.innerHTML = `<p><a href="#queue">← Queue</a></p><div class="panel"><h2>${esc(data.name)}</h2>
      <p class="muted">${esc(data.why || "Not ready for review yet.")} Come back once the job is past "Align lines".</p></div>`;
    return;
  }
  const keep = RV && RV.id === Number(id) ? RV : null;
  RV = {
    id: Number(id), data, dec: keep && keep.dirty ? keep.dec : rvClone(data.decisions || {}),
    dirty: keep ? keep.dirty : false, tab: keep ? keep.tab : "subs", sel: keep ? keep.sel : null, hear: "orig",
    search: "", spk: null, follow: true, view: keep ? keep.view : [0, Math.min(240, data.duration)], markIn: null, flag: -1,
  };
  for (const k of ["regions", "cues", "lines"]) if (!RV.dec[k]) RV.dec[k] = k === "regions" ? [] : {};
  rvShell();
  rvMedia();
  rvList();
  rvInspector();
  rvDraw();
}

function rvShell() {
  const d = RV.data, base = `/api/jobs/${RV.id}/review/media/`;
  const nLook = d.flags.length;
  const statusChip = d.editable ? `<span class="chip keep">Waiting for review</span>`
    : `<span class="chip plain">${esc(d.job.status)}</span>`;
  main.innerHTML = `
  <div class="rv ${d.editable ? "editable" : ""}">
    <div class="rv-title">
      <div class="crumbs"><a href="#queue">Queue</a> › <b>${esc(d.name)}</b></div>
      ${statusChip}${nLook ? `<span class="chip look">${nLook} to check</span>` : `<span class="chip dub">Nothing flagged</span>`}
      <span class="grow"></span>
      ${d.editable ? `<button id="rvDiscard">Discard my changes</button><button id="rvSave">${RV.dirty ? "Save" : "Saved"}</button>
        <button class="primary" id="rvStart">Start dubbing ›</button>` : ""}
    </div>
    ${d.editable ? "" : `<div class="rv-note">This job isn't waiting for review, so you can look but not change anything. Set "Review before dubbing" to Always when adding a file to stop here.</div>`}
    <div class="rv-body">
      <div class="rv-left">
        <div class="rv-video">
          <video id="rvv" preload="auto" playsinline crossorigin="anonymous">
            <track kind="subtitles" srclang="en" label="English" src="${base}subs.en.vtt" default>
            <track kind="subtitles" srclang="ja" label="Japanese" src="${base}subs.ja.vtt">
          </video>
          <div class="tag" id="rvTag"></div>
          <div class="wait" id="rvWait" hidden></div>
        </div>
        <audio id="rvVoices" preload="auto" src="${base}voices.m4a"></audio>
        <audio id="rvMusic" preload="auto" src="${base}music.m4a"></audio>
        <div class="rv-transport">
          <button class="play" id="rvPlay" title="Play / pause (Space)">▶</button>
          <button id="rvBack" title="Back 5 s (Shift+←)">−5 s</button><button id="rvFwd" title="Forward 5 s (Shift+→)">+5 s</button>
          <span class="rv-time" id="rvTime">0:00.0 / ${fmtT(d.duration, false)}</span>
          <span class="grow" style="flex:1"></span>
          <span class="muted small">Hear</span>
          <div class="seg" id="rvHear"><button data-v="orig" class="on">Original</button><button data-v="voices">Voices only</button><button data-v="music">Music only</button></div>
          <div class="seg" id="rvRate"><button data-v="0.75">0.75×</button><button data-v="1" class="on">1×</button><button data-v="1.5">1.5×</button></div>
          <div class="seg" id="rvSubs"><button data-v="en" class="on">English subs</button><button data-v="both">+ Japanese</button><button data-v="none">Off</button></div>
        </div>
      </div>
      <div class="rv-right">
        <div class="rv-tabs" id="rvTabs"></div>
        <div class="rv-filter" id="rvFilter"><input type="text" id="rvSearch" placeholder="Search subtitles…">
          <button id="rvFollow" class="on">Follow video ✓</button></div>
        <div class="rv-list" id="rvList"></div>
        <div class="rv-ins" id="rvIns"></div>
      </div>
    </div>
    <div class="rv-tl">
      <div class="rv-tl-top">
        <b>Timeline</b><span class="rv-time muted" id="rvRange"></span>
        <div class="seg" id="rvZoom"><button data-v="all">Whole episode</button><button data-v="240">4 min</button><button data-v="30">30 s</button></div>
        <span class="grow" style="flex:1"></span>
        ${nLook ? `<span class="chip look" id="rvFlagPos">Needs a look ${nLook}</span><button id="rvPrev">‹ Prev</button><button id="rvNext">Next ›</button>` : ""}
        ${d.editable ? `<button id="rvMark" title="Mark in (I) and out (O)">+ Mark region</button>` : ""}
      </div>
      <div class="rv-lanes">
        <div class="rv-lane-names">${RV_LANES.map(([n, h], i) => `<span style="height:${h}px;margin-bottom:${i < RV_LANES.length - 1 ? RV_GAP : 0}px">${n}</span>`).join("")}</div>
        <div class="rv-canvas-wrap"><canvas id="rvTl"></canvas></div>
      </div>
      <div class="rv-overview"><canvas id="rvOv" title="The whole episode: drag the box to move"></canvas></div>
      <div class="rv-keys">
        <span><kbd>Space</kbd> play / pause</span><span><kbd>←</kbd><kbd>→</kbd> 1 s</span><span><kbd>Shift</kbd>+<kbd>←</kbd><kbd>→</kbd> 5 s</span>
        ${d.editable ? `<span><kbd>I</kbd> <kbd>O</kbd> mark in / out</span><span><kbd>K</kbd> keep original</span><span><kbd>D</kbd> dub</span><span><kbd>Delete</kbd> remove region</span>` : ""}
        <span><kbd>N</kbd> <kbd>P</kbd> next / previous to check</span><span><kbd>Z</kbd> zoom</span>
      </div>
    </div>
  </div>`;
  rvTabs();
  rvWire();
}

function rvTabs() {
  const d = RV.data;
  const tabs = [["subs", "Subtitles", d.cues.length || d.lines.length], ["voices", "Voices", d.speakers.length],
    ["look", "Needs a look", d.flags.length], ["kept", "Kept", regionsNow().filter((r) => r.action === "keep").length],
    ["rules", "Show rules", (d.rules.regions || []).length]];
  $("#rvTabs").innerHTML = tabs.map(([k, t, n]) => `<button data-tab="${k}" class="${RV.tab === k ? "on" : ""}">${t}<span class="n">${n}</span></button>`).join("");
  $("#rvTabs").querySelectorAll("button").forEach((b) => (b.onclick = () => { RV.tab = b.dataset.tab; rvTabs(); rvList(); }));
  $("#rvFilter").hidden = RV.tab !== "subs";
}

// Media and playback ----------------------------------------------------------------------------------
function rvMedia() {
  const st = RV.data.media || {}, v = $("#rvv"), wait = $("#rvWait");
  if (st.state === "ready") {
    wait.hidden = true;
    v.src = `/api/jobs/${RV.id}/review/media/preview.mp4`;
    rvSubsMode("en");
    return;
  }
  wait.hidden = false;
  if (st.state === "building") {
    wait.innerHTML = `<div>Preparing the preview video… this takes a minute or two.</div>`;
    setTimeout(async () => {
      if (!location.hash.startsWith(`#review/${RV.id}`)) return;
      const d = await api(`/jobs/${RV.id}/review`).catch(() => null);
      if (d && d.media) { RV.data.media = d.media; rvMedia(); }
    }, 4000);
  } else {
    wait.innerHTML = `<div>${st.state === "failed" ? `The preview couldn't be made: ${esc(st.error || "")}` : "No preview video yet."}</div>
      <div><button class="primary" id="rvPrep">Prepare preview</button></div>`;
    $("#rvPrep").onclick = async () => {
      try { RV.data.media = await api(`/jobs/${RV.id}/review/media`, { body: {} }); } catch (e) { toast(e.message); }
      rvMedia();
    };
  }
}
function rvSubsMode(mode) {
  const tt = $("#rvv").textTracks;
  for (let i = 0; i < tt.length; i++) {
    const lang = tt[i].language;
    tt[i].mode = (mode === "en" && lang === "en") || (mode === "both") ? "showing" : "hidden";
  }
}
function rvSeek(t) {
  const v = $("#rvv");
  t = Math.max(0, Math.min(RV.data.duration, t));
  if (v && v.readyState) v.currentTime = t;
  RV.t = t;
  rvSync(true);
  rvKeepInView(t);
  rvDraw();
  rvClock();
}
function rvNow() { const v = $("#rvv"); return v && v.readyState ? v.currentTime : (RV.t || 0); }
function rvSync(force) {
  const v = $("#rvv"), aud = { voices: $("#rvVoices"), music: $("#rvMusic") };
  for (const [k, a] of Object.entries(aud)) {
    if (!a) continue;
    if (RV.hear !== k) { if (!a.paused) a.pause(); continue; }
    a.playbackRate = v.playbackRate;
    if (force || Math.abs(a.currentTime - v.currentTime) > 0.25) a.currentTime = v.currentTime;
    if (v.paused && !a.paused) a.pause();
    if (!v.paused && a.paused) a.play().catch(() => {});
  }
  v.muted = RV.hear !== "orig";
}
function rvClock() {
  const t = rvNow();
  $("#rvTime").textContent = `${fmtT(t)} / ${fmtT(RV.data.duration, false)}`;
  const r = regionsNow().find((x) => t >= x.start && t <= x.end && x.action === "keep");
  const ln = RV.data.lines.find((l) => t >= l.start && t <= l.end && (l.kind === "speech" || l.kind === "song"));
  $("#rvTag").innerHTML = r ? `<span class="chip keep">♪ ${esc(RV_KINDS[r.kind] || "Song")} · keeps original</span>`
    : ln && ln.kind === "speech" ? `<span class="chip dub">Dubbed · <span class="vdot" style="background:${rvColor(lineWho(ln))}"></span>${esc(whoLabel(lineWho(ln)))}</span>` : "";
  if (RV.follow && RV.tab === "subs") {
    const rows = $("#rvList").querySelectorAll(".srow");
    let cur = null;
    rows.forEach((row) => {
      const on = t >= Number(row.dataset.a) && t <= Number(row.dataset.b) + 0.3;
      row.classList.toggle("cur", on);
      if (on && !cur) cur = row;
    });
    if (cur && !RV.listHover) cur.scrollIntoView({ block: "nearest" });
  }
}
function rvKeepInView(t) {
  const [a, b] = RV.view, w = b - a;
  if (t < a || t > b) {
    const s = Math.max(0, Math.min(RV.data.duration - w, t - w * 0.2));
    RV.view = [s, s + w];
  }
}
function rvZoom(span) {
  const d = RV.data.duration, t = rvNow();
  if (span === "all") RV.view = [0, d];
  else {
    const w = Math.min(d, Number(span));
    const s = Math.max(0, Math.min(d - w, t - w / 2));
    RV.view = [s, s + w];
  }
  rvDraw();
}

// Lists ----------------------------------------------------------------------------------------------
function rvList() {
  const box = $("#rvList");
  const tab = RV.tab;
  if (tab === "subs") box.innerHTML = rvSubRows();
  else if (tab === "voices") box.innerHTML = rvVoiceRows();
  else if (tab === "look") box.innerHTML = RV.data.flags.length ? RV.data.flags.map((f, i) => `
      <div class="frow" data-flag="${i}"><span class="rv-time muted">${fmtT(f.start, false)}</span>
        <div><b>${esc(f.title)}</b><span class="d">${esc(f.detail)}</span></div></div>`).join("")
    : `<div class="rv-empty">Nothing flagged. Ai Dub was sure of every decision in this episode.</div>`;
  else if (tab === "kept") {
    const regs = regionsNow();
    box.innerHTML = regs.length ? regs.map((r) => `
      <div class="frow" data-region="${esc(r.id)}"><span class="rv-time muted">${fmtT(r.start, false)}</span>
        <div><b>${esc(RV_KINDS[r.kind] || "Song")} · ${fmtT(r.start)}–${fmtT(r.end)}
          ${r.action === "dub" ? `<span class="chip dub">will be dubbed</span>` : r.source === "user" ? `<span class="chip mine">your mark</span>` : ""}</b>
          <span class="d">${esc((r.reasons || ["marked by you"]).join(" · "))}</span></div></div>`).join("")
      : `<div class="rv-empty">Nothing keeps the original in this episode.</div>`;
  } else if (tab === "rules") box.innerHTML = rvRulesHtml();
  rvListWire();
  rvClock();
}
function rvSubRows() {
  const d = RV.data, q = RV.search.toLowerCase();
  let items = d.cues.length ? d.cues.map((c) => ({ ...c, cue: true }))
    : d.lines.filter((l) => l.source).map((l) => ({ key: `l${l.id}`, start: l.start, end: l.end, text: l.source, line: l }));
  if (q) items = items.filter((c) => c.text.toLowerCase().includes(q));
  if (RV.spk) items = items.filter((c) => { const l = c.line || cueLine(c); return l && lineWho(l) === RV.spk; });
  if (!items.length) return `<div class="rv-empty">No subtitles match.</div>`;
  const filt = RV.spk ? `<div class="rv-empty" style="padding:8px 10px">Lines of ${esc(whoLabel(RV.spk))} · <a href="#" id="rvAll">show all</a></div>` : "";
  return filt + items.map((c) => {
    const [cls, label] = c.cue ? cueState(c) : [c.line.kind === "speech" ? "dub" : "keep", c.line.kind === "speech" ? "Dub" : "Kept"];
    const ln = c.line || cueLine(c);
    const who = ln ? lineWho(ln) : "";
    const sel = RV.sel && RV.sel.type === "cue" && RV.sel.id === c.key;
    const text = c.italic ? `<i>${esc(c.text)}</i>` : esc(c.text);
    const why = ln ? `<span class="why"><button class="who" data-line="${ln.id}" title="${RV.data.editable ? "Change whose voice" : ""}"><span class="vdot" style="background:${rvColor(who)}"></span>${esc(whoLabel(who))}</button>
      ${ln.kept_why && ln.kind === "song" ? `<span>· ${esc(ln.kept_why)}</span>` : ""}${flaggedLine(ln.id) ? `<span>· sounds sung</span>` : ""}
      ${ln.voice_note === "moved by pitch" ? `<span>· moved by pitch</span>` : ""}</span>` : "";
    return `<div class="srow ${sel ? "sel" : ""}" data-key="${esc(c.key)}" data-a="${c.start}" data-b="${c.end}">
      <span class="t">${fmtT(c.start, false)}</span><div class="x">${text}${why}</div>
      ${c.cue && d.editable ? `<button class="chip ${cls}" data-cycle="${esc(c.key)}" title="Click: keep original / dub / Ai Dub decides">${esc(label)}</button>`
        : `<span class="chip ${cls}">${esc(label)}</span>`}</div>`;
  }).join("");
}
function rvVoiceRows() {
  const sp = RV.data.speakers;
  if (!sp.length) return `<div class="rv-empty">No speakers found yet.</div>`;
  return sp.map((s) => {
    const sel = RV.sel && RV.sel.type === "speaker" && RV.sel.id === s.id;
    const sex = s.age === "child" ? "Child" : s.gender === "male" ? "Man" : s.gender === "female" ? "Woman" : "Man or woman unsure";
    const borrowed = /^voice of|^generic voice/.test(s.voice);
    const initials = s.name.replace(/^Character\s+/, "").split(/\s+/).map((w) => w[0]).join("").slice(0, 2);
    return `<div class="vrow ${sel ? "sel" : ""}" data-spk="${esc(s.id)}">
      <span class="vav" style="background:${rvColor(s.id)}">${esc(initials)}</span>
      <div><div class="vname">${esc(s.name)}</div><div class="vmeta"><span>${sex}${s.age && s.age !== "child" ? ` · ${esc(s.age)}` : ""}</span>
        <span>${s.lines} line${s.lines === 1 ? "" : "s"} · ${dur(s.seconds)}</span><span>${esc(s.voice)}</span></div></div>
      <div class="row">${s.sample ? `<button class="vplay" data-sample="${esc(s.sample)}" title="Play a sample of the original voice">▶</button>` : ""}
        ${s.moved_here ? `<span class="chip look">${s.moved_here} moved here</span>` : ""}${borrowed || !s.gender ? `<span class="chip look">Check</span>` : ""}</div></div>`;
  }).join("");
}
function rvRulesHtml() {
  const r = RV.data.rules || {}, habits = { romaji_lyrics: true, italic_quoted_lyrics: true, tilde_signs: true, narrator_dubbed: true, ...(r.habits || {}) };
  const names = {
    romaji_lyrics: ["Romaji lines are lyrics", "A subtitle written in romanised Japanese is a song."],
    italic_quoted_lyrics: ["Italics in quotation marks are lyrics", "Over singing, such a subtitle keeps the original."],
    tilde_signs: ["“~ text ~” is a sign", "Shown on screen, not spoken."],
    narrator_dubbed: ["The narrator's opening is dubbed", "Subtitled lines inside the shared opening are dubbed."],
  };
  const regs = r.regions || [];
  return `<div class="switch"><div style="flex:1"><b>Review this show</b><small>When to stop before dubbing its episodes.</small></div>
      <select id="rvRuleMode"><option value="">Use the default</option><option value="off">Off</option><option value="unsure">When unsure</option><option value="always">Always</option></select></div>
    ${Object.entries(names).map(([k, [t, s]]) => `<label class="switch"><input type="checkbox" data-habit="${k}" ${habits[k] ? "checked" : ""}>
      <span><b>${t}</b><small>${s}</small></span></label>`).join("")}
    <div class="rv-empty" style="padding:10px 10px 4px"><b style="color:var(--text)">Openings and endings for every episode</b><br>
      Mark one as Opening or Ending, keep it, and tick "Use for every episode of this show".</div>
    ${regs.map((x) => `<div class="frow"><span class="rv-time muted">${fmtT(x.start, false)}</span><div><b>${esc(x.label || x.kind)} · ${x.seconds} s</b>
      <span class="d">set on ${esc(x.episode || "")}</span> <button class="small" data-unrule="${esc(x.id)}">Remove</button></div></div>`).join("")}`;
}
function rvListWire() {
  const box = $("#rvList"), d = RV.data;
  box.onmouseenter = () => (RV.listHover = true);
  box.onmouseleave = () => (RV.listHover = false);
  box.querySelectorAll(".srow").forEach((row) => (row.onclick = (ev) => {
    if (ev.target.closest("button")) return;
    RV.sel = { type: "cue", id: row.dataset.key };
    rvSeek(Number(row.dataset.a));
    rvList(); rvInspector();
  }));
  box.querySelectorAll("[data-cycle]").forEach((b) => (b.onclick = () => {
    const k = b.dataset.cycle, now = RV.dec.cues[k];
    if (!now) RV.dec.cues[k] = "keep"; else if (now === "keep") RV.dec.cues[k] = "dub"; else delete RV.dec.cues[k];
    setDirty(); rvList(); rvInspector(); rvDraw();
  }));
  box.querySelectorAll(".who").forEach((b) => (b.onclick = (ev) => { if (d.editable) rvWhoMenu(ev, Number(b.dataset.line)); }));
  const all = $("#rvAll");
  if (all) all.onclick = (e) => { e.preventDefault(); RV.spk = null; rvList(); };
  box.querySelectorAll(".vrow").forEach((row) => (row.onclick = (ev) => {
    if (ev.target.closest("button")) return;
    RV.sel = { type: "speaker", id: row.dataset.spk }; rvList(); rvInspector();
  }));
  box.querySelectorAll("[data-sample]").forEach((b) => (b.onclick = () => {
    const a = new Audio(`/api/profiles/${encodeURIComponent(d.job.profile)}/ref?file=${encodeURIComponent(b.dataset.sample)}`);
    a.play().catch(() => toast("Couldn't play the sample"));
  }));
  box.querySelectorAll("[data-flag]").forEach((row) => (row.onclick = () => rvGoFlag(Number(row.dataset.flag))));
  box.querySelectorAll("[data-region]").forEach((row) => (row.onclick = () => {
    const r = regionsNow().find((x) => x.id === row.dataset.region);
    RV.sel = { type: "region", id: r.id }; rvSeek(r.start); rvInspector();
  }));
  const mode = $("#rvRuleMode");
  if (mode) {
    mode.value = (d.rules || {}).review || "";
    mode.onchange = () => rvSaveRules({ review: mode.value });
    box.querySelectorAll("[data-habit]").forEach((cb) => (cb.onchange = () => {
      const habits = {};
      box.querySelectorAll("[data-habit]").forEach((x) => (habits[x.dataset.habit] = x.checked));
      rvSaveRules({ habits });
    }));
    box.querySelectorAll("[data-unrule]").forEach((b) => (b.onclick = () => {
      if (confirm("Remove this opening/ending rule? Later episodes won't look for it.")) rvSaveRules({ remove_regions: [b.dataset.unrule] });
    }));
  }
}
async function rvSaveRules(body) {
  try {
    await api(`/profiles/${encodeURIComponent(RV.data.job.profile)}/rules`, { method: "PUT", body });
    const r = await api(`/profiles/${encodeURIComponent(RV.data.job.profile)}/rules`);
    RV.data.rules = { review: r.review, habits: r.habits, regions: r.regions };
    toast("Saved for every episode of this show"); rvTabs(); rvList();
  } catch (e) { toast(e.message); }
}
function rvGoFlag(i) {
  const f = RV.data.flags[i];
  if (!f) return;
  RV.flag = i;
  const reg = regionsNow().find((r) => f.lines.some((id) => r.lines && r.lines.includes(id)));
  const cue = RV.data.cues.find((q) => overlap(q.start, q.end, f.start, f.end) > 0);
  RV.sel = f.kind === "voice" && f.lines.length ? { type: "line", id: f.lines[0] }
    : reg ? { type: "region", id: reg.id } : cue ? { type: "cue", id: cue.key } : { type: "flag", id: i };
  RV.selFlag = i;
  rvSeek(Math.max(0, f.start - 1));
  const pos = $("#rvFlagPos");
  if (pos) pos.textContent = `Needs a look ${i + 1} of ${RV.data.flags.length}`;
  if (RV.tab !== "look") { rvList(); }
  rvInspector();
}

// Speaker menu ------------------------------------------------------------------------------------------
function rvWhoMenu(ev, lineId) {
  rvClosePop();
  const pop = document.createElement("div");
  pop.className = "rv-pop";
  const now = lineWho(RV.data.lines.find((l) => l.id === lineId));
  pop.innerHTML = RV.data.speakers.map((s) => `<button data-who="${esc(s.id)}"><span class="vdot" style="background:${rvColor(s.id)}"></span>${esc(s.name)}
      <span class="muted small">${s.gender === "male" ? "man" : s.gender === "female" ? "woman" : ""}</span>${s.id === now ? " ✓" : ""}</button>`).join("")
    + `<hr><button data-who="own">Its own voice (this line's audio)</button><button data-who="new">New speaker…</button>`
    + (RV.dec.lines[String(lineId)] ? `<button data-who="undo">Back to Ai Dub's choice</button>` : "");
  document.body.appendChild(pop);
  const r = ev.target.getBoundingClientRect();
  pop.style.left = `${Math.min(window.innerWidth - 240, r.left)}px`;
  pop.style.top = `${Math.min(window.innerHeight - 330, r.bottom + 4)}px`;
  pop.querySelectorAll("button").forEach((b) => (b.onclick = () => {
    const w = b.dataset.who;
    if (w === "undo") delete RV.dec.lines[String(lineId)];
    else if (w === "new") {
      const name = prompt("Name for the new speaker", "");
      if (!name) return rvClosePop();
      RV.dec.lines[String(lineId)] = { character: `new:${name.trim()}` };
    } else RV.dec.lines[String(lineId)] = { character: w };
    setDirty(); rvClosePop(); rvList(); rvInspector(); rvDraw();
  }));
  setTimeout(() => document.addEventListener("click", rvClosePop, { once: true }), 0);
}
function rvClosePop() { document.querySelectorAll(".rv-pop").forEach((p) => p.remove()); }

// Inspector ------------------------------------------------------------------------------------------------
function rvInspector() {
  const box = $("#rvIns"), d = RV.data, s = RV.sel, ed = d.editable;
  if (!box) return;
  if (!s) {
    box.innerHTML = `<h4>Details</h4><div class="muted small">Click a region, a subtitle or a speaker to see why Ai Dub decided what it did${ed ? " and change it" : ""}.
      ${d.flags.length ? `Start with <a href="#" id="rvFirst">the first thing to check</a>.` : ""}</div>`;
    const f = $("#rvFirst");
    if (f) f.onclick = (e) => { e.preventDefault(); rvGoFlag(0); };
    return;
  }
  if (s.type === "region") {
    const r = regionsNow().find((x) => x.id === s.id);
    if (!r) { RV.sel = null; return rvInspector(); }
    const ai = r.source !== "user";
    box.innerHTML = `<h4>Region ${fmtT(r.start)} – ${fmtT(r.end)} <span class="chip ${r.action === "dub" ? "dub" : "keep"}">${r.action === "dub" ? "Dubbed" : esc(RV_KINDS[r.kind] || "Song")}</span>
        <span class="chip ${ai ? "plain" : "mine"}">${ai ? (r.edited ? "found by Ai Dub, changed by you" : "found by Ai Dub") : "your mark"}</span></h4>
      <div class="grid"><span class="lab">Type</span><select id="rvKind" ${ed ? "" : "disabled"}>${Object.entries(RV_KINDS).map(([k, t]) => `<option value="${k}" ${r.kind === k ? "selected" : ""}>${t}</option>`).join("")}</select>
        <span class="lab">From / to</span><span class="row"><input class="time" id="rvFrom" value="${fmtT(r.start)}" ${ed ? "" : "disabled"}> → <input class="time" id="rvTo" value="${fmtT(r.end)}" ${ed ? "" : "disabled"}>
          <span class="muted small">or drag the edges on the timeline</span></span></div>
      ${r.reasons ? `<ul>${r.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : `<ul><li>Marked by you: everything inside keeps the original.</li></ul>`}
      ${ed ? `<div class="row"><button class="btn-keep ${r.action === "keep" ? "on" : ""}" id="rvKeep">Keep original</button>
        <button class="btn-dub ${r.action === "dub" ? "on" : ""}" id="rvDub">Dub it</button>
        ${!ai ? `<button id="rvRemove">Remove region</button>` : ""}</div>
        ${["opening", "ending"].includes(r.kind) && r.action === "keep" ? `<label class="row small"><input type="checkbox" id="rvRule" ${r.show_rule ? "checked" : ""}> Use for every episode of this show</label>` : ""}` : ""}`;
    if (!ed) return;
    $("#rvKind").onchange = (e) => { setRegion(r, { kind: e.target.value }); rvInspector(); rvDraw(); };
    const setTimes = () => {
      const a = parseT($("#rvFrom").value), b = parseT($("#rvTo").value);
      if (a == null || b == null || b <= a) return toast("Times look like 12:44.8");
      setRegion(r, { start: a, end: b }); rvInspector(); rvDraw();
    };
    $("#rvFrom").onchange = setTimes; $("#rvTo").onchange = setTimes;
    $("#rvKeep").onclick = () => { setRegion(r, { action: "keep" }); rvInspector(); rvDraw(); rvList(); };
    $("#rvDub").onclick = () => { setRegion(r, { action: "dub" }); rvInspector(); rvDraw(); rvList(); };
    if ($("#rvRemove")) $("#rvRemove").onclick = () => { removeRegion(r); rvInspector(); rvDraw(); rvList(); };
    if ($("#rvRule")) $("#rvRule").onchange = (e) => { setRegion(r, { show_rule: e.target.checked, label: RV_KINDS[r.kind] }); };
    return;
  }
  if (s.type === "cue" || s.type === "line") {
    const q = s.type === "cue" ? d.cues.find((x) => x.key === s.id) : null;
    const ln = s.type === "line" ? d.lines.find((l) => l.id === s.id) : q && cueLine(q);
    const [cls, label] = q ? cueState(q) : [ln && ln.kind === "speech" ? "dub" : "keep", ln && ln.kind === "speech" ? "Dub" : "Kept"];
    const who = ln ? lineWho(ln) : "";
    box.innerHTML = `<h4>${q ? "Subtitle" : "Line"} ${fmtT((q || ln).start)} <span class="chip ${cls}">${esc(label)}</span></h4>
      <div class="grid">
        ${q ? `<span class="lab">Subtitle</span><span>${q.italic ? `<i>${esc(q.text)}</i>` : esc(q.text)}</span>` : ""}
        ${ln ? `<span class="lab">Heard</span><span>${esc(ln.source || "—")}</span>
        <span class="lab">Voice</span><span><button class="who" id="rvWho"><span class="vdot" style="background:${rvColor(who)}"></span>${esc(whoLabel(who))}</button>
          ${ln.voice_note ? `<span class="muted small">${esc(ln.voice_note)}</span>` : ""}</span>
        <span class="lab">Why</span><span class="muted small">${esc(ln.kind === "song" ? (ln.kept_why || "kept as a song") : flaggedLine(ln.id)
          ? `held notes over music (${ln.sung.toFixed(2)}): the lyrics check decides unless you choose` : "spoken line, dubbed")}</span>` : ""}
      </div>
      ${ed && q ? `<div class="row"><button class="btn-keep ${RV.dec.cues[q.key] === "keep" ? "on" : ""}" id="rvCKeep">Keep original</button>
        <button class="btn-dub ${RV.dec.cues[q.key] === "dub" ? "on" : ""}" id="rvCDub">Dub it</button>
        ${RV.dec.cues[q.key] ? `<button id="rvCAuto">Let Ai Dub decide</button>` : ""}</div>` : ""}`;
    if (!ed) return;
    if ($("#rvWho") && ln) $("#rvWho").onclick = (ev) => rvWhoMenu(ev, ln.id);
    if (q) {
      $("#rvCKeep").onclick = () => { RV.dec.cues[q.key] = "keep"; setDirty(); rvList(); rvInspector(); rvDraw(); };
      $("#rvCDub").onclick = () => { RV.dec.cues[q.key] = "dub"; setDirty(); rvList(); rvInspector(); rvDraw(); };
      if ($("#rvCAuto")) $("#rvCAuto").onclick = () => { delete RV.dec.cues[q.key]; setDirty(); rvList(); rvInspector(); rvDraw(); };
    }
    return;
  }
  if (s.type === "speaker") {
    const sp = rvSpeaker(s.id);
    if (!sp) { RV.sel = null; return rvInspector(); }
    const sex = sp.age === "child" ? "child" : sp.gender;
    box.innerHTML = `<h4><span class="vdot" style="background:${rvColor(sp.id)}"></span>${esc(sp.name)}</h4>
      <div class="grid">
        <span class="lab">Name</span><input type="text" id="rvSpName" value="${esc(sp.name)}" ${ed ? "" : "disabled"}>
        <span class="lab">Voice</span><div class="seg" id="rvSpSex">${[["male", "Man"], ["female", "Woman"], ["child", "Child"]].map(([v, t]) =>
          `<button data-v="${v}" class="${sex === v ? "on" : ""}" ${ed ? "" : "disabled"}>${t}</button>`).join("")}</div>
        <span class="lab">Age</span><div class="seg" id="rvSpAge">${[["young", "Young"], ["adult", "Adult"], ["senior", "Older"]].map(([v, t]) =>
          `<button data-v="${v}" class="${sp.age === v ? "on" : ""}" ${ed ? "" : "disabled"}>${t}</button>`).join("")}</div>
        <span class="lab">Dubbed with</span><span class="muted small">${esc(sp.voice)}</span>
      </div>
      <div class="row" style="margin-top:8px"><button id="rvSpLines">Show lines</button>
        ${ed && d.speakers.length > 1 ? `<button id="rvSpSame">Same person as…</button>` : ""}</div>
      <p class="muted small" style="margin:8px 0 0">Names, man / woman and merges are kept for every episode of this show.</p>`;
    $("#rvSpLines").onclick = () => { RV.spk = sp.id; RV.tab = "subs"; rvTabs(); rvList(); };
    if (!ed) return;
    const act = async (body, msg) => {
      try {
        await api(`/profiles/${encodeURIComponent(d.job.profile)}/characters/${encodeURIComponent(sp.id)}`, { body });
        const fresh = await api(`/jobs/${RV.id}/review`);
        RV.data.speakers = fresh.speakers; RV.data.lines = fresh.lines;
        if (msg) toast(msg);
        rvList(); rvInspector(); rvDraw();
      } catch (e) { toast(e.message); }
    };
    const name = $("#rvSpName");
    name.onchange = () => act({ action: "rename", name: name.value }, "Renamed");
    name.onkeydown = (e) => { if (e.key === "Enter") name.blur(); };
    $("#rvSpSex").querySelectorAll("button").forEach((b) => (b.onclick = () => act(b.dataset.v === "child"
      ? { action: "set", age: "child" } : { action: "set", gender: b.dataset.v, ...(sp.age === "child" ? { age: "young" } : {}) })));
    $("#rvSpAge").querySelectorAll("button").forEach((b) => (b.onclick = () => act({ action: "set", age: b.dataset.v })));
    const same = $("#rvSpSame");
    if (same) same.onclick = (ev) => {
      rvClosePop();
      const pop = document.createElement("div");
      pop.className = "rv-pop";
      pop.innerHTML = `<div class="muted small" style="padding:4px 8px">${esc(sp.name)} is the same person as:</div>` + d.speakers.filter((x) => x.id !== sp.id)
        .map((x) => `<button data-other="${esc(x.id)}"><span class="vdot" style="background:${rvColor(x.id)}"></span>${esc(x.name)}</button>`).join("");
      document.body.appendChild(pop);
      const r = ev.target.getBoundingClientRect();
      pop.style.left = `${Math.min(window.innerWidth - 240, r.left)}px`;
      pop.style.top = `${Math.max(8, r.top - 330)}px`;
      pop.querySelectorAll("[data-other]").forEach((b) => (b.onclick = () => {
        const other = rvSpeaker(b.dataset.other);
        rvClosePop();
        if (!confirm(`Merge ${other.name} into ${sp.name}? Their voice clips join ${sp.name}'s, for every episode of this show.`)) return;
        act({ action: "merge", other: other.id }, `${other.name} is now ${sp.name}`);
      }));
      setTimeout(() => document.addEventListener("click", rvClosePop, { once: true }), 0);
    };
    return;
  }
  if (s.type === "flag") {
    const f = d.flags[s.id];
    box.innerHTML = `<h4>${esc(f.title)}</h4><div class="muted small">${esc(f.detail)}</div>`;
  }
}

// Timeline ---------------------------------------------------------------------------------------------------
function rvLaneY() {
  const ys = [];
  let y = 0;
  for (const [, h] of RV_LANES) { ys.push([y, h]); y += h + RV_GAP; }
  return { ys, height: y - RV_GAP };
}
function rvDraw() {
  const cv = $("#rvTl");
  if (!cv || !RV) return;
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, { ys, height } = rvLaneY();
  cv.style.height = `${height}px`;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(height * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(height * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, height);
  const [v0, v1] = RV.view, X = (t) => ((t - v0) / (v1 - v0)) * W;
  const C = { line: rvCss("--line"), bar: rvCss("--bg"), muted: rvCss("--muted"), text: rvCss("--text"), accent: rvCss("--accent"),
    keep: rvCss("--keep") || rvCss("--warn"), dub: rvCss("--ok"), look: rvCss("--bad"), show: rvCss("--show"), keepBg: rvCss("--keep-bg"),
    showBg: rvCss("--show-bg"), mineBg: rvCss("--mine-bg") };
  for (let i = 1; i < ys.length; i++) { g.fillStyle = C.bar; rr(g, 0, ys[i][0], W, ys[i][1], 4); g.fill(); }
  // Ruler
  const span = v1 - v0, steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  const step = steps.find((s) => (s / span) * W >= 70) || 600;
  g.fillStyle = C.muted; g.font = "10.5px system-ui, sans-serif"; g.textAlign = "center";
  for (let t = Math.ceil(v0 / step) * step; t <= v1; t += step) {
    const x = X(t);
    g.fillText(fmtT(t, step < 1), x, 11);
    g.fillRect(x, 13, 1, 5);
  }
  const regs = regionsNow();
  const keptAt = (t) => regs.some((r) => r.action === "keep" && t >= r.start && t <= r.end);
  // Waveform
  const pk = RV.peaks, [wy, wh] = ys[1];
  if (pk) {
    const n = pk.peaks.length, mid = wy + wh / 2;
    for (let x = 0; x < W; x += 2) {
      const t0 = v0 + (x / W) * span, t1 = v0 + ((x + 2) / W) * span;
      let m = 0;
      for (let i = Math.floor(t0 / pk.step); i <= Math.min(n - 1, Math.floor(t1 / pk.step)); i++) m = Math.max(m, pk.peaks[i] || 0);
      const h = Math.max(1, (m / 100) * (wh / 2 - 2));
      g.fillStyle = keptAt(t0) ? C.keep : C.accent; g.globalAlpha = keptAt(t0) ? 0.75 : 0.6;
      g.fillRect(x, mid - h, 1.5, h * 2);
    }
    g.globalAlpha = 1;
  }
  // Keep original
  const [ky, kh] = ys[2];
  for (const r of regs.filter((x) => x.source !== "user")) {
    const x0 = X(r.start), x1 = X(r.end);
    if (x1 < 0 || x0 > W) continue;
    const shared = /shared|show rule/.test((r.reasons || []).join(" "));
    const col = r.action === "dub" ? C.muted : shared ? C.show : C.keep;
    g.fillStyle = r.action === "dub" ? "transparent" : shared ? C.showBg : C.keepBg;
    rr(g, x0, ky + 3, Math.max(2, x1 - x0), kh - 6, 4); g.fill();
    g.strokeStyle = col; g.lineWidth = 1; g.setLineDash(r.action === "dub" ? [4, 3] : []); g.stroke(); g.setLineDash([]);
    label(g, x0, x1, ky + kh / 2, `${r.action === "dub" ? "dubbed" : "♪ " + (RV_KINDS[r.kind] || "Song")}`, col);
    if (RV.sel && RV.sel.type === "region" && RV.sel.id === r.id) handles(g, x0, x1, ky, kh, col);
  }
  // Subtitles
  const [sy, sh] = ys[3];
  const colOf = { dub: C.dub, keep: C.keep, look: C.look, sign: C.muted, plain: C.muted, mine: C.accent };
  for (const q of RV.data.cues) {
    const x0 = X(q.start), x1 = X(q.end);
    if (x1 < 0 || x0 > W) continue;
    g.fillStyle = colOf[cueState(q)[0]]; g.globalAlpha = 0.85;
    rr(g, x0, sy + 6, Math.max(2, x1 - x0 - 1), sh - 12, 2); g.fill(); g.globalAlpha = 1;
    if (RV.sel && RV.sel.type === "cue" && RV.sel.id === q.key) { g.strokeStyle = C.text; g.lineWidth = 1.5; g.stroke(); }
  }
  g.fillStyle = C.look;
  for (const f of RV.data.flags) {
    const x = X(f.start);
    if (x < -6 || x > W + 6) continue;
    g.beginPath(); g.moveTo(x - 5, sy); g.lineTo(x + 5, sy); g.lineTo(x, sy + 7); g.closePath(); g.fill();
  }
  // Speakers
  const [py, ph] = ys[4];
  for (const ln of RV.data.lines) {
    if (ln.kind !== "speech" && ln.kind !== "song") continue;
    const x0 = X(ln.start), x1 = X(ln.end);
    if (x1 < 0 || x0 > W) continue;
    const who = lineWho(ln);
    g.fillStyle = rvColor(who); g.globalAlpha = ln.kind === "song" ? 0.35 : 0.8;
    rr(g, x0, py + 5, Math.max(2, x1 - x0 - 1), ph - 10, 2); g.fill(); g.globalAlpha = 1;
    if (RV.dec.lines[String(ln.id)] || ln.voice_note === "moved by pitch") { g.strokeStyle = C.text; g.lineWidth = 1; g.stroke(); }
    if (x1 - x0 > 60) label(g, x0, x1, py + ph / 2, whoLabel(who), C.text, true);
  }
  // Your marks
  const [my, mh] = ys[5];
  for (const r of regs.filter((x) => x.source === "user")) {
    const x0 = X(r.start), x1 = X(r.end);
    if (x1 < 0 || x0 > W) continue;
    g.fillStyle = C.mineBg; rr(g, x0, my + 3, Math.max(2, x1 - x0), mh - 6, 4); g.fill();
    g.strokeStyle = C.accent; g.setLineDash([4, 3]); g.stroke(); g.setLineDash([]);
    label(g, x0, x1, my + mh / 2, RV_KINDS[r.kind] || "keep", C.accent);
    if (RV.sel && RV.sel.type === "region" && RV.sel.id === r.id) handles(g, x0, x1, my, mh, C.accent);
  }
  if (RV.markIn != null) { g.fillStyle = C.accent; g.fillRect(X(RV.markIn), my, 2, mh); }
  if (RV.drag && RV.drag.kind === "new") {
    const a = Math.min(RV.drag.t0, RV.drag.t1), b = Math.max(RV.drag.t0, RV.drag.t1);
    g.fillStyle = C.mineBg; g.fillRect(X(a), my + 3, X(b) - X(a), mh - 6);
  }
  // Playhead
  const xp = X(rvNow());
  if (xp >= 0 && xp <= W) { g.fillStyle = C.text; g.fillRect(xp - 1, 0, 2, height); }
  $("#rvRange").textContent = `${fmtT(v0, false)} – ${fmtT(v1, false)}`;
  rvOverview(C);
}
function rr(g, x, y, w, h, r) {
  g.beginPath(); r = Math.min(r, w / 2, h / 2);
  g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}
function label(g, x0, x1, y, text, col, small) {
  const w = x1 - x0;
  if (w < 34) return;
  g.save(); g.beginPath(); g.rect(Math.max(0, x0), y - 10, w, 20); g.clip();
  g.fillStyle = col; g.font = `${small ? 10.5 : 11}px system-ui, sans-serif`; g.textAlign = "left"; g.textBaseline = "middle";
  g.fillText(text, Math.max(0, x0) + 6, y + 0.5); g.restore();
}
function handles(g, x0, x1, y, h, col) {
  g.fillStyle = col;
  for (const x of [x0, x1]) { rr(g, x - 3, y + 4, 6, h - 8, 2); g.fill(); }
  g.lineWidth = 2; g.strokeStyle = col; rr(g, x0, y + 2, x1 - x0, h - 4, 4); g.stroke();
}
function rvOverview(C) {
  const cv = $("#rvOv");
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = 18, d = RV.data.duration;
  if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  g.fillStyle = C.bar; rr(g, 0, 0, W, H, 4); g.fill();
  for (const r of regionsNow()) {
    if (r.action !== "keep") continue;
    const shared = /shared|show rule/.test((r.reasons || []).join(" "));
    g.fillStyle = r.source === "user" ? C.accent : shared ? C.show : C.keep;
    g.fillRect((r.start / d) * W, 4, Math.max(2, ((r.end - r.start) / d) * W), H - 8);
  }
  g.fillStyle = C.look;
  for (const f of RV.data.flags) g.fillRect((f.start / d) * W, 0, 2, 4);
  const [v0, v1] = RV.view;
  g.strokeStyle = C.accent; g.lineWidth = 1.5; rr(g, (v0 / d) * W, 0.75, Math.max(4, ((v1 - v0) / d) * W), H - 1.5, 3); g.stroke();
  g.fillStyle = C.text; g.fillRect((rvNow() / d) * W, 0, 1.5, H);
}

// Wiring -------------------------------------------------------------------------------------------------------
function rvWire() {
  const v = $("#rvv"), d = RV.data;
  v.onplay = () => { $("#rvPlay").textContent = "❚❚"; rvSync(true); rvLoop(); };
  v.onpause = () => { $("#rvPlay").textContent = "▶"; rvSync(true); rvDraw(); };
  v.onseeked = () => rvSync(true);
  v.ontimeupdate = () => { rvSync(false); rvClock(); };
  v.onloadedmetadata = () => { if (RV.t) v.currentTime = RV.t; };
  $("#rvPlay").onclick = () => (v.paused ? v.play().catch(() => {}) : v.pause());
  $("#rvBack").onclick = () => rvSeek(rvNow() - 5);
  $("#rvFwd").onclick = () => rvSeek(rvNow() + 5);
  segWire("#rvHear", (val) => { RV.hear = val; rvSync(true); });
  segWire("#rvRate", (val) => { v.playbackRate = Number(val); rvSync(true); });
  segWire("#rvSubs", (val) => rvSubsMode(val));
  segWire("#rvZoom", (val) => rvZoom(val));
  const search = $("#rvSearch");
  search.oninput = () => { RV.search = search.value; rvList(); };
  $("#rvFollow").onclick = (e) => { RV.follow = !RV.follow; e.target.textContent = RV.follow ? "Follow video ✓" : "Follow video"; };
  if ($("#rvNext")) { $("#rvNext").onclick = () => rvGoFlag((RV.flag + 1) % d.flags.length); $("#rvPrev").onclick = () => rvGoFlag((RV.flag - 1 + d.flags.length) % d.flags.length); }
  if ($("#rvMark")) $("#rvMark").onclick = () => rvMarkKey();
  if (d.editable) {
    $("#rvSave").onclick = () => rvSave(false);
    $("#rvStart").onclick = () => rvSave(true);
    $("#rvDiscard").onclick = async () => {
      if (!confirm("Discard your changes to this episode? (Speaker names and merges are already saved.)")) return;
      const fresh = await api(`/jobs/${RV.id}/review`);
      RV.dec = rvClone(fresh.decisions); RV.dirty = false; RV.data = fresh;
      rvShell(); rvMedia(); rvList(); rvInspector(); rvDraw();
    };
  }
  fetch(`/api/jobs/${RV.id}/review/media/peaks.json`).then((r) => (r.ok ? r.json() : null)).then((p) => { RV.peaks = p; rvDraw(); }).catch(() => {});
  rvTimelineWire();
  window.onresize = () => rvDraw();
}
function segWire(sel, fn) {
  const box = $(sel);
  box.querySelectorAll("button").forEach((b) => (b.onclick = () => {
    box.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    fn(b.dataset.v);
  }));
}
function rvLoop() {
  const v = $("#rvv");
  if (!v || v.paused || !location.hash.startsWith("#review/")) return;
  const t = v.currentTime;
  if (RV.follow && (t > RV.view[1] || t < RV.view[0])) rvKeepInView(t);
  rvDraw();
  requestAnimationFrame(rvLoop);
}
async function rvSave(start) {
  try {
    if (start) {
      const n = Object.keys(RV.dec.cues).length + RV.dec.regions.filter((r) => r.source !== "hidden").length + Object.keys(RV.dec.lines).length;
      if (!confirm(n ? `Start dubbing with your ${n} change${n === 1 ? "" : "s"}?` : "Start dubbing with Ai Dub's decisions?")) return;
      await api(`/jobs/${RV.id}/review/start`, { body: decisionsOut() });
      RV.dirty = false;
      toast("Dubbing continues: the job is back at the front of the queue");
      location.hash = "#queue";
      return;
    }
    await api(`/jobs/${RV.id}/review`, { method: "PUT", body: decisionsOut() });
    RV.dirty = false;
    $("#rvSave").textContent = "Saved";
    toast("Saved");
  } catch (e) { toast(e.message); }
}
function rvMarkKey() {
  const t = rvNow();
  if (RV.markIn == null) { RV.markIn = t; toast("Marked in. Press O (or the button again) at the end."); }
  else {
    const r = addRegion(Math.min(RV.markIn, t), Math.max(RV.markIn, t));
    RV.markIn = null;
    if (r) { RV.sel = { type: "region", id: r.id }; rvInspector(); rvList(); }
  }
  rvDraw();
}
function rvTimelineWire() {
  const cv = $("#rvTl"), ov = $("#rvOv");
  const at = (ev) => {
    const r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    const t = RV.view[0] + (x / r.width) * (RV.view[1] - RV.view[0]);
    const { ys } = rvLaneY();
    const lane = ys.findIndex(([y0, h]) => y >= y0 && y <= y0 + h);
    return { x, t, lane, pxPerS: r.width / (RV.view[1] - RV.view[0]) };
  };
  cv.onpointerdown = (ev) => {
    const p = at(ev), ed = RV.data.editable;
    cv.setPointerCapture(ev.pointerId);
    if (p.lane === 2 || p.lane === 5) {
      const user = p.lane === 5;
      const sel = RV.sel && RV.sel.type === "region" ? regionsNow().find((r) => r.id === RV.sel.id) : null;
      if (ed && sel && (sel.source === "user") === user) {
        for (const edge of ["start", "end"]) {
          if (Math.abs((sel[edge] - p.t) * p.pxPerS) < 7) { RV.drag = { kind: "edge", edge, r: sel }; return; }
        }
      }
      const hit = regionAt(p.t, user);
      if (hit) { RV.sel = { type: "region", id: hit.id }; rvInspector(); rvDraw(); RV.drag = { kind: "seek" }; rvSeek(p.t); return; }
      if (user && ed) { RV.drag = { kind: "new", t0: p.t, t1: p.t }; return; }
    }
    if (p.lane === 3) {
      const q = RV.data.cues.find((c) => p.t >= c.start - 0.05 && p.t <= c.end + 0.05);
      if (q) { RV.sel = { type: "cue", id: q.key }; RV.tab = "subs"; rvTabs(); rvList(); rvInspector(); }
    }
    if (p.lane === 4) {
      const ln = RV.data.lines.find((l) => p.t >= l.start && p.t <= l.end && (l.kind === "speech" || l.kind === "song"));
      if (ln) { RV.sel = { type: "line", id: ln.id }; rvInspector(); }
    }
    RV.drag = { kind: "seek" };
    rvSeek(p.t);
  };
  cv.onpointermove = (ev) => {
    const p = at(ev);
    if (!RV.drag) {
      const sel = RV.sel && RV.sel.type === "region" ? regionsNow().find((r) => r.id === RV.sel.id) : null;
      const near = sel && RV.data.editable && (p.lane === 2 || p.lane === 5) && ["start", "end"].some((e) => Math.abs((sel[e] - p.t) * p.pxPerS) < 7);
      cv.style.cursor = near ? "ew-resize" : p.lane === 5 && RV.data.editable ? "copy" : "crosshair";
      return;
    }
    if (RV.drag.kind === "seek") rvSeek(p.t);
    else if (RV.drag.kind === "new") { RV.drag.t1 = p.t; rvDraw(); }
    else if (RV.drag.kind === "edge") {
      const r = RV.drag.r;
      const t = Math.max(0, Math.min(RV.data.duration, p.t));
      if (RV.drag.edge === "start") r.start = Math.min(t, r.end - 0.3); else r.end = Math.max(t, r.start + 0.3);
      RV.drag.moved = true;
      rvDrawDragged(r);
    }
  };
  cv.onpointerup = () => {
    const dr = RV.drag;
    RV.drag = null;
    if (!dr) return;
    if (dr.kind === "new") {
      const r = addRegion(Math.min(dr.t0, dr.t1), Math.max(dr.t0, dr.t1));
      if (r) { RV.sel = { type: "region", id: r.id }; rvInspector(); rvTabs(); }
    } else if (dr.kind === "edge" && dr.moved) {
      setRegion(dr.r, { start: +dr.r.start.toFixed(2), end: +dr.r.end.toFixed(2) });
      rvInspector(); rvList();
    }
    rvDraw();
  };
  cv.onwheel = (ev) => {
    const span = RV.view[1] - RV.view[0], d = RV.data.duration;
    if (ev.ctrlKey || ev.metaKey) {
      ev.preventDefault();
      const p = at(ev), k = ev.deltaY > 0 ? 1.25 : 0.8;
      const w = Math.max(5, Math.min(d, span * k));
      const s = Math.max(0, Math.min(d - w, p.t - (p.t - RV.view[0]) * (w / span)));
      RV.view = [s, s + w];
    } else if (Math.abs(ev.deltaX) > Math.abs(ev.deltaY) || ev.shiftKey) {
      ev.preventDefault();
      const dx = (ev.shiftKey ? ev.deltaY : ev.deltaX) / cv.clientWidth * span;
      const s = Math.max(0, Math.min(d - span, RV.view[0] + dx));
      RV.view = [s, s + span];
    } else return;
    rvDraw();
  };
  const ovMove = (ev) => {
    const r = ov.getBoundingClientRect(), d = RV.data.duration, span = RV.view[1] - RV.view[0];
    const t = ((ev.clientX - r.left) / r.width) * d;
    const s = Math.max(0, Math.min(d - span, t - span / 2));
    RV.view = [s, s + span];
    rvDraw();
  };
  ov.onpointerdown = (ev) => { ov.setPointerCapture(ev.pointerId); RV.ovDrag = true; ovMove(ev); };
  ov.onpointermove = (ev) => { if (RV.ovDrag) ovMove(ev); };
  ov.onpointerup = () => (RV.ovDrag = false);
}
function rvDrawDragged(r) {
  // The region being dragged is drawn from the live copy until the pointer lets go.
  const saved = RV.dec.regions;
  const live = saved.filter((x) => x.id !== r.id);
  live.push({ ...r, source: r.source === "user" ? "user" : "ai" });
  RV.dec.regions = live;
  rvDraw();
  RV.dec.regions = saved;
}
document.addEventListener("keydown", (ev) => {
  if (!RV || !location.hash.startsWith("#review/")) return;
  if (ev.target.closest("input, select, textarea") || ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const v = $("#rvv"), ed = RV.data.editable, k = ev.key;
  const sel = RV.sel;
  const act = (action) => {
    if (!ed || !sel) return;
    if (sel.type === "region") { const r = regionsNow().find((x) => x.id === sel.id); if (r) setRegion(r, { action }); }
    else if (sel.type === "cue") RV.dec.cues[sel.id] = action;
    else return;
    setDirty(); rvList(); rvInspector(); rvDraw();
  };
  if (k === " ") { ev.preventDefault(); v.paused ? v.play().catch(() => {}) : v.pause(); }
  else if (k === "ArrowLeft" || k === "ArrowRight") { ev.preventDefault(); rvSeek(rvNow() + (k === "ArrowLeft" ? -1 : 1) * (ev.shiftKey ? 5 : 1)); }
  else if ((k === "i" || k === "I") && ed) { RV.markIn = rvNow(); rvDraw(); toast("Marked in. Press O at the end."); }
  else if ((k === "o" || k === "O") && ed && RV.markIn != null) rvMarkKey();
  else if (k === "k" || k === "K") act("keep");
  else if (k === "d" || k === "D") act("dub");
  else if ((k === "n" || k === "N") && RV.data.flags.length) rvGoFlag((RV.flag + 1) % RV.data.flags.length);
  else if ((k === "p" || k === "P") && RV.data.flags.length) rvGoFlag((RV.flag - 1 + RV.data.flags.length) % RV.data.flags.length);
  else if (k === "z" || k === "Z") {
    const span = RV.view[1] - RV.view[0], d = RV.data.duration;
    const next = span >= d - 1 ? 240 : span > 60 ? 30 : "all";
    rvZoom(next);
    $("#rvZoom").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === String(next)));
  } else if ((k === "Delete" || k === "Backspace") && ed && sel && sel.type === "region") {
    const r = regionsNow().find((x) => x.id === sel.id);
    if (r) { removeRegion(r); rvInspector(); rvDraw(); rvList(); }
  } else if (k === "Escape") { rvClosePop(); RV.sel = null; RV.markIn = null; rvInspector(); rvDraw(); }
});
window.addEventListener("beforeunload", (ev) => {
  if (RV && RV.dirty && location.hash.startsWith("#review/")) { ev.preventDefault(); ev.returnValue = ""; }
});
