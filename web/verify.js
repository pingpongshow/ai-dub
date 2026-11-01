"use strict";
// Check a finished dub: listen to it against the original, then keep it (the original
// and the check files are deleted), put the original back, or dub again with a review.

let VF = null;

async function pageVerify(id) {
  main.classList.add("wide");
  const d = await api(`/jobs/${id}/verify`);
  if (!d.ready) {
    const done = { good: "You kept this dub; its original and working files are deleted.", restored: "The original is back; the dub was deleted.",
      redo: "The original is back and the episode is queued again with a review." }[d.verify];
    main.innerHTML = `<p><a href="#queue">← Queue</a></p><div class="panel"><h2>${esc(d.name)}</h2>
      <p class="muted">${esc(done || "There's nothing to check for this job: its check files aren't there.")}</p></div>`;
    return;
  }
  VF = { id: Number(id), d, hear: "dub", m: -1, ab: null, ticks: new Set(store.get(`ticks.${id}`, [])),
    redo: store.get(`redo.${id}`, {}), now: null };
  VF.en = Object.fromEntries((d.lines || []).map(([lid, , , en]) => [lid, en]));
  d.moments.forEach((m) => { if (m.line != null && m.en != null) VF.en[m.line] = m.en; });
  const frees = (d.original.exists ? d.original.size : 0) + (d.work_size || 0);
  const base = `/api/jobs/${id}/verify/media/`;
  const gbs = (n) => (n / 1e9).toFixed(2) + " GB";
  main.innerHTML = `
  <div class="rv editable vf">
    <div class="rv-title">
      <div class="crumbs"><a href="#queue">Queue</a><span class="faint">›</span><b title="${esc(d.name)}">${epLabel(d.name)}</b>${episode(d.name).show ? `<span class="muted">${esc(episode(d.name).show)}</span>` : ""}</div>
      <span class="chip keep">Dubbed: not checked yet</span>
      <span class="grow"></span>
      ${d.can_redub ? `<a class="button" href="#review/${id}" title="Open the full review page: change songs, subtitles or voices, then redub from there (the analysis is kept)">Review and change…</a>`
        : `<button id="vfRedo" title="Put the original back and queue the episode again, stopping for review">Redo with review</button>`}
      <button class="ghost" id="vfRestore" title="Delete the dub and put the original file back">Put the original back</button>
      <span class="vsep"></span>
      <button class="primary" id="vfGood" title="Keep the dub; the original and the working files are deleted for good">✓ Keep the dub${frees ? ` · frees ${gbs(frees)}` : ""}</button>
    </div>
    <div class="rv-body">
      <div class="rv-left">
        <div class="rv-video">
          <video id="vfv" preload="auto" playsinline src="${base}preview.mp4">
            <track kind="subtitles" srclang="en" label="English" src="${base}subs.en.vtt" default>
          </video>
          <div class="tag" id="vfTag"><span class="chip dub">Hearing the dub</span></div>
        </div>
        <audio id="vfDub" preload="auto" src="${base}dub.m4a"></audio>
        <div class="rv-transport">
          <button class="play" id="vfPlay" title="Play / pause (Space)">▶</button>
          <button id="vfBack">−5 s</button><button id="vfFwd">+5 s</button>
          <span class="rv-time" id="vfTime">0:00.0 / ${fmtT(d.duration, false)}</span>
          <span class="grow" style="flex:1"></span>
          <span class="muted small">Hear</span>
          <div class="seg" id="vfHear"><button data-v="dub" class="on">Dub</button><button data-v="orig">Original</button></div>
          <div class="seg" id="vfRate"><button data-v="1" class="on">1×</button><button data-v="1.5">1.5×</button></div>
          <div class="seg" id="vfSubs"><button data-v="en" class="on">English subs</button><button data-v="none">Off</button></div>
        </div>
        <div class="rv-overview" style="margin-left:0;margin-top:12px"><canvas id="vfBar" style="height:26px" title="Click to jump; marks are moments to check"></canvas></div>
        <div class="rv-keys"><span><kbd>Space</kbd> play / pause</span><span><kbd>←</kbd><kbd>→</kbd> 1 s</span>
          <span><kbd>Shift</kbd>+<kbd>←</kbd><kbd>→</kbd> 5 s</span><span><kbd>N</kbd> <kbd>P</kbd> next / previous moment</span>
          <span><kbd>D</kbd> dub · <kbd>O</kbd> original</span></div>
        ${d.can_redo_lines ? `<div class="vf-now" id="vfNow"><span class="muted small">Hear a line that's wrong? Its controls show here while it plays.</span></div>` : ""}
      </div>
      <div class="rv-right">
        <div class="rv-tabs"><button class="on">Moments to check<span class="n" id="vfTicked"></span></button></div>
        <div class="vf-redo" id="vfRedoBar" hidden><span class="grow" id="vfRedoN"></span>
          <button class="sm ghost" id="vfRedoClear">Clear</button><button class="sm primary" id="vfRedoGo"></button></div>
        <div class="rv-list" id="vfList">${d.moments.length ? d.moments.map((m, i) => `
          <div class="frow vfm" data-m="${i}"><button class="tick" data-tick="${i}" title="Tick it off once you've listened" aria-label="Listened">✓</button>
            <div><span class="rv-time muted">${fmtT(m.t, false)}</span> <b>${esc(m.title)}</b>${m.detail ? `<span class="d">${esc(m.detail)}</span>` : ""}
              <div class="row" style="margin-top:6px;gap:6px"><button class="sm" data-ab="${i}" title="Play these few seconds as the original, then as the dub">A/B</button>
              ${m.line != null && !d.can_redo_lines && d.can_redub ? `<a class="button sm" href="#review/${id}/${m.line}" title="Open this line on the review page: new English, another speaker, or just voice it again">Fix this line…</a>` : ""}</div>
              ${m.line != null && d.can_redo_lines ? `<div class="vf-act" data-line="${m.line}"></div>` : ""}</div></div>`).join("")
          : `<div class="rv-empty">Nothing stood out. Play a few scenes to be sure.</div>`}</div>
        <div class="rv-ins">
          <h4>When you're happy with it</h4>
          <div class="grid">
            <span class="lab">Original</span><span>${d.original.exists ? `${gbs(d.original.size)}, set aside next to the dub` : "already gone"}</span>
            <span class="lab">Dubbed lines</span><span>${d.qc.dubbed_lines ?? "?"}${d.qc.fallback_rate != null ? ` · ${(d.qc.fallback_rate * 100).toFixed(1)} % left in Japanese` : ""}</span>
            ${d.qc.free_timing ? `<span class="lab">Timing</span><span>latest start ${d.qc.free_timing.max_wait_s} s after its original</span>` : ""}
          </div>
          ${d.can_redub ? `<div class="grid" style="margin-top:6px"><span class="lab">Working files</span><span>${gbs(d.work_size)}, kept so you can change and redub</span></div>` : ""}
          <p class="muted small" style="margin:8px 0 0"><b>Keep the dub</b> deletes the original and the working files, for good.
            <b>Put the original back</b> deletes the dub instead. ${d.can_redub
              ? "<b>Review and change</b> opens the full review page, even if nothing was flagged; redub from there with your changes."
              : "<b>Redo with review</b> puts the original back and dubs again, stopping for review first."}</p>
        </div>
      </div>
    </div>
  </div>`;
  vfWire();
  vfDraw();
}

// A/B: the same few seconds as the original, then as the dub.
function vfAB(i) {
  const m = VF.d.moments[i], v = $("#vfv");
  if (!m || !v) return;
  const a = Math.max(0, m.t - 1.5), b = Math.min(VF.d.duration, m.t + 4.5);
  clearInterval(VF.abTimer);
  VF.ab = { phase: "A", a, b };
  vfGo(i, true);
  vfHear("orig");
  pairedSeek(v, null, a);
  v.play().catch(() => {});
  $("#vfTag").innerHTML = `<span class="chip plain">A · original</span>`;
  VF.abTimer = setInterval(() => {
    if (!VF.ab) return clearInterval(VF.abTimer);
    if (v.paused && !v._pairBusy) return;
    if (v.currentTime < VF.ab.b) return;
    if (VF.ab.phase === "A") {
      VF.ab.phase = "B";
      vfHear("dub");
      pairedSeek(v, $("#vfDub"), VF.ab.a);
      $("#vfTag").innerHTML = `<span class="chip dub">B · the dub</span>`;
    } else {
      v.pause(); VF.ab = null; clearInterval(VF.abTimer);
      $("#vfTag").innerHTML = `<span class="chip dub">Hearing the dub</span>`;
    }
  }, 100);
}
function vfStopAB() { if (VF && VF.ab) { VF.ab = null; clearInterval(VF.abTimer); } }

function vfNow() { const v = $("#vfv"); return v ? v.currentTime : 0; }
function vfSync(force) {
  const v = $("#vfv"), a = $("#vfDub");
  if (!v || !a) return;
  v.muted = VF.hear === "dub";
  pairedSync(v, a, VF.hear === "dub", force);
}
function vfSeek(t) {
  vfStopAB();
  const v = $("#vfv");
  pairedSeek(v, VF.hear === "dub" ? $("#vfDub") : null, Math.max(0, Math.min(VF.d.duration, t)));
  vfDraw();
}
function vfHear(h) {
  VF.hear = h;
  $("#vfHear").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === h));
  $("#vfTag").innerHTML = h === "dub" ? `<span class="chip dub">Hearing the dub</span>` : `<span class="chip plain">Hearing the original</span>`;
  vfSync(true);
}
function vfGo(i, noSeek = false) {
  const m = VF.d.moments[i];
  if (!m) return;
  VF.m = i;
  document.querySelectorAll("#vfList .frow").forEach((r) => r.classList.toggle("sel", Number(r.dataset.m) === i));
  const row = document.querySelector(`#vfList .frow[data-m="${i}"]`);
  if (row) row.scrollIntoView({ block: "nearest" });
  if (!noSeek) vfSeek(Math.max(0, m.t - 2));
}
function vfDraw() {
  const cv = $("#vfBar");
  if (!cv || !VF) return;
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = 26, d = VF.d.duration;
  if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  g.fillStyle = rvCss("--bg"); rr(g, 0, 0, W, H, 5); g.fill();
  const col = { song: rvCss("--keep") || rvCss("--warn"), kept: rvCss("--bad"), voice: rvCss("--accent"), unclear: rvCss("--warn"), late: rvCss("--muted") };
  const ms = VF.d.moments;
  for (let i = 0; i + 1 < ms.length; i++) {  // song stretches as bands between their start and end marks
    if (ms[i].kind === "song" && /starts/.test(ms[i].title)) {
      const end = ms.slice(i + 1).find((x) => x.kind === "song" && /ends/.test(x.title));
      if (end) { g.fillStyle = rvCss("--keep-bg"); g.fillRect((ms[i].t / d) * W, 3, ((end.t - ms[i].t) / d) * W, H - 6); }
    }
  }
  for (const m of ms) { g.fillStyle = col[m.kind] || col.late; g.fillRect((m.t / d) * W - 1, 4, 2, H - 8); }
  g.fillStyle = rvCss("--text"); g.fillRect((vfNow() / d) * W - 1, 0, 2, H);
}
function vfLoop() {
  const v = $("#vfv");
  if (!v || v.paused || !location.hash.startsWith("#verify/")) return;
  vfDraw();
  requestAnimationFrame(vfLoop);
}
function vfWire() {
  const v = $("#vfv"), d = VF.d;
  v.onplay = () => { $("#vfPlay").textContent = "❚❚"; vfSync(false); vfLoop(); };
  v.onpause = () => { if (v._pairBusy) return; $("#vfPlay").textContent = "▶"; vfSync(false); vfDraw(); };
  v.onseeked = () => vfSync(false);
  v.ontimeupdate = () => { vfSync(false); $("#vfTime").textContent = `${fmtT(v.currentTime)} / ${fmtT(d.duration, false)}`; vfNowLine(); };
  $("#vfPlay").onclick = () => { vfStopAB(); if (v._pairBusy) $("#vfPlay").textContent = pairedToggle(v) ? "❚❚" : "▶"; else pairedToggle(v); };
  $("#vfBack").onclick = () => vfSeek(vfNow() - 5);
  $("#vfFwd").onclick = () => vfSeek(vfNow() + 5);
  segWire("#vfHear", (val) => { vfStopAB(); vfHear(val); });
  segWire("#vfRate", (val) => { v.playbackRate = Number(val); vfSync(false); });
  segWire("#vfSubs", (val) => { for (const t of v.textTracks) t.mode = val === "en" ? "showing" : "hidden"; });
  document.querySelectorAll("#vfList .frow").forEach((r) => (r.onclick = () => vfGo(Number(r.dataset.m))));
  const bar = $("#vfBar");
  bar.onclick = (ev) => { const r = bar.getBoundingClientRect(); vfSeek(((ev.clientX - r.left) / r.width) * d.duration); };
  const decide = async (verdict, question, done) => {
    if (!confirm(question)) return;
    try {
      const r = await api(`/jobs/${VF.id}/verify`, { body: { verdict } });
      toast(r.message || done);
      location.hash = "#queue";
    } catch (e) { toast(e.message); }
  };
  $("#vfGood").onclick = () => {
    const left = d.moments.length - VF.ticks.size;
    decide("good", (left > 0 ? `You haven't ticked ${left} of the ${d.moments.length} moments yet.\n\n` : "")
      + `Keep this dub and permanently delete the original${d.original.exists ? ` (${(d.original.size / 1e9).toFixed(2)} GB)` : ""} and the working files? This can't be undone.`, "Kept");
  };
  // Ticks: which moments you've listened to (kept in this browser, per dub).
  const ticks = () => {
    document.querySelectorAll("#vfList .vfm").forEach((r) => r.classList.toggle("ticked", VF.ticks.has(Number(r.dataset.m))));
    $("#vfTicked").textContent = d.moments.length ? `${VF.ticks.size} / ${d.moments.length}` : "0";
  };
  document.querySelectorAll("#vfList [data-tick]").forEach((b) => (b.onclick = (e) => {
    e.stopPropagation();
    const i = Number(b.dataset.tick);
    VF.ticks.has(i) ? VF.ticks.delete(i) : VF.ticks.add(i);
    store.set(`ticks.${VF.id}`, [...VF.ticks]);
    ticks();
  }));
  document.querySelectorAll("#vfList [data-ab]").forEach((b) => (b.onclick = (e) => { e.stopPropagation(); vfAB(Number(b.dataset.ab)); }));
  document.querySelectorAll("#vfList a").forEach((a) => (a.onclick = (e) => e.stopPropagation()));
  ticks();
  $("#vfRestore").onclick = () => decide("restore", "Put the original back and delete this dub?", "Original put back");
  if ($("#vfRedo")) $("#vfRedo").onclick = () => decide("redo", "Put the original back, delete this dub, and dub the episode again (stopping for review first)?", "Queued again");
  window.onresize = () => vfDraw();
  if (d.can_redo_lines) {
    $("#vfRedoClear").onclick = () => { VF.redo = {}; vfRedoSave(); vfActions(); };
    $("#vfRedoGo").onclick = () => vfRedoGo();
    vfActions();
  }
}

// Voicing lines again: mark them here (same words, or new ones), then send them all at
// once. Each gets fresh takes; every other line keeps its voice.
function vfRedoSave() { store.set(`redo.${VF.id}`, VF.redo); }
function vfActionHtml(lid) {
  const r = VF.redo[lid];
  if (!r) {
    return `<div class="row" style="gap:6px"><button class="sm" data-vo="${lid}" title="Voice this line again with the same words: fresh takes, the best one is kept">↻ Voice again</button>
      <button class="sm ghost" data-vw="${lid}" title="Change the English, then voice it again">Change words…</button>
      <a class="button sm ghost" href="#review/${VF.id}/${lid}" title="Give the line to another speaker on the review page">Other speaker…</a></div>`;
  }
  return `<div class="vf-marked"><span class="chip dub">↻ Will be voiced again${r.en != null ? " with your words" : ""}</span>
      ${r.en != null ? `<textarea rows="2" data-ven="${lid}" aria-label="English to voice">${esc(r.en)}</textarea>
        <span class="small muted">Saved as you type. Sent with the others when you press “Voice them again”.</span>` : ""}
      <div class="row" style="gap:6px">${r.en == null ? `<button class="sm ghost" data-vw="${lid}">Change words…</button>` : ""}
        <button class="sm ghost" data-vx="${lid}">Don't voice it again</button></div></div>`;
}
function vfActions() {
  document.querySelectorAll(".vf-act[data-line]").forEach((box) => (box.innerHTML = vfActionHtml(Number(box.dataset.line))));
  const root = document.querySelector(".rv");
  root.querySelectorAll("[data-vo]").forEach((b) => (b.onclick = (e) => {
    e.stopPropagation();
    const box = b.closest(".vf-act");
    VF.redo[b.dataset.vo] = {}; vfRedoSave(); vfActions();
    box.scrollIntoView({ block: "nearest" });
  }));
  root.querySelectorAll("[data-vw]").forEach((b) => (b.onclick = (e) => {
    e.stopPropagation();
    VF.redo[b.dataset.vw] = { en: VF.en[b.dataset.vw] || "" };
    vfRedoSave(); vfActions();
    const t = root.querySelector(`[data-ven="${b.dataset.vw}"]`);
    if (t) {
      t.focus({ preventScroll: true });
      t.setSelectionRange(t.value.length, t.value.length);
      t.closest(".vf-act").scrollIntoView({ block: "nearest" });  // the box and its buttons in view
    }
  }));
  root.querySelectorAll("[data-vx]").forEach((b) => (b.onclick = (e) => { e.stopPropagation(); delete VF.redo[b.dataset.vx]; vfRedoSave(); vfActions(); }));
  root.querySelectorAll("[data-ven]").forEach((t) => {
    t.onclick = (e) => e.stopPropagation();
    t.oninput = () => { if (VF.redo[t.dataset.ven]) { VF.redo[t.dataset.ven].en = t.value; vfRedoSave(); } };
  });
  root.querySelectorAll(".vf-act a").forEach((a) => (a.onclick = (e) => e.stopPropagation()));
  const n = Object.keys(VF.redo).length;
  $("#vfRedoBar").hidden = !n;
  $("#vfRedoN").textContent = `${n} line${n === 1 ? "" : "s"} to voice again`;
  $("#vfRedoGo").textContent = `↻ Voice ${n === 1 ? "it" : "them"} again ›`;
}
// The line playing now, for one the check didn't flag.
function vfNowLine() {
  const box = $("#vfNow");
  if (!box || !VF.d.lines.length) return;
  const t = vfNow();
  let cur = null;
  for (const l of VF.d.lines) { if (l[1] > t) break; if (t <= l[2] + 0.5) cur = l; }
  const lid = cur ? cur[0] : null;
  if (lid === VF.now && box.dataset.ready) return;
  VF.now = lid; box.dataset.ready = "1";
  box.innerHTML = cur ? `<span class="rv-time muted">${fmtT(cur[1], false)}</span> <span class="said">“${esc(cur[3])}”</span>
      <div class="vf-act" data-line="${lid}"></div>`
    : `<span class="muted small">Hear a line that's wrong? Its controls show here while it plays.</span>`;
  vfActions();
}
async function vfRedoGo() {
  const ids = Object.keys(VF.redo);
  if (!ids.length) return;
  const lines = {};
  for (const lid of ids) {
    const en = (VF.redo[lid].en || "").trim();
    lines[lid] = en && en !== (VF.en[lid] || "") ? { en } : {};
  }
  const words = Object.values(lines).filter((x) => x.en).length;
  if (!confirm(`Voice ${ids.length} line${ids.length === 1 ? "" : "s"} again${words ? ` (${words} with your words)` : ""}?\n\n`
    + "Each gets fresh takes and the best one is kept; every other line keeps its voice. The original goes back while this runs "
    + "(a few minutes), then the dub comes back here to check.")) return;
  try {
    const r = await api(`/jobs/${VF.id}/lines/redo`, { body: { lines } });
    VF.redo = {}; vfRedoSave();
    toast(r.message || "Voicing again");
    location.hash = "#queue";
  } catch (e) { toast(e.message); }
}
document.addEventListener("keydown", (ev) => {
  if (!VF || !location.hash.startsWith("#verify/")) return;
  if ((ev.target.closest && ev.target.closest("input, select, textarea")) || ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const v = $("#vfv"), k = ev.key, n = VF.d.moments.length;
  if (!v) return;
  if (k === " ") { ev.preventDefault(); $("#vfPlay").click(); }
  else if (k === "ArrowLeft" || k === "ArrowRight") { ev.preventDefault(); vfSeek(vfNow() + (k === "ArrowLeft" ? -1 : 1) * (ev.shiftKey ? 5 : 1)); }
  else if ((k === "n" || k === "N") && n) vfGo((VF.m + 1) % n);
  else if ((k === "p" || k === "P") && n) vfGo((VF.m - 1 + n) % n);
  else if (k === "d" || k === "D") vfHear("dub");
  else if (k === "o" || k === "O") vfHear("orig");
});
