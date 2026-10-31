"use strict";
// Check a finished dub: listen to it against the original, then keep it (the original
// and the check files are deleted), put the original back, or dub again with a review.

let VF = null;

async function pageVerify(id) {
  main.classList.add("wide");
  const d = await api(`/jobs/${id}/verify`);
  if (!d.ready) {
    const done = { good: "You kept this dub; its original and check files are deleted.", restored: "The original is back; the dub was deleted.",
      redo: "The original is back and the episode is queued again with a review." }[d.verify];
    main.innerHTML = `<p><a href="#queue">← Queue</a></p><div class="panel"><h2>${esc(d.name)}</h2>
      <p class="muted">${esc(done || "There's nothing to check for this job: its check files aren't there.")}</p></div>`;
    return;
  }
  VF = { id: Number(id), d, hear: "dub", m: -1 };
  const base = `/api/jobs/${id}/verify/media/`;
  const gbs = (n) => (n / 1e9).toFixed(2) + " GB";
  main.innerHTML = `
  <div class="rv editable">
    <div class="rv-title">
      <div class="crumbs"><a href="#queue">Queue</a> › <b>${esc(d.name)}</b></div>
      <span class="chip keep">Dubbed: not checked yet</span>
      <span class="grow"></span>
      <button id="vfRedo" title="Put the original back and queue the episode again, stopping for review">Redo with review</button>
      <button id="vfRestore">Put the original back</button>
      <button class="primary" id="vfGood">Looks good: delete the original</button>
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
      </div>
      <div class="rv-right">
        <div class="rv-tabs"><button class="on">Moments to check<span class="n">${d.moments.length}</span></button></div>
        <div class="rv-list" id="vfList">${d.moments.length ? d.moments.map((m, i) => `
          <div class="frow" data-m="${i}"><span class="rv-time muted">${fmtT(m.t, false)}</span>
            <div><b>${esc(m.title)}</b>${m.detail ? `<span class="d">${esc(m.detail)}</span>` : ""}</div></div>`).join("")
          : `<div class="rv-empty">Nothing stood out. Play a few scenes to be sure.</div>`}</div>
        <div class="rv-ins">
          <h4>When you're happy with it</h4>
          <div class="grid">
            <span class="lab">Original</span><span>${d.original.exists ? `${gbs(d.original.size)}, set aside next to the dub` : "already gone"}</span>
            <span class="lab">Dubbed lines</span><span>${d.qc.dubbed_lines ?? "?"}${d.qc.fallback_rate != null ? ` · ${(d.qc.fallback_rate * 100).toFixed(1)} % left in Japanese` : ""}</span>
            ${d.qc.free_timing ? `<span class="lab">Timing</span><span>latest start ${d.qc.free_timing.max_wait_s} s after its original</span>` : ""}
          </div>
          <p class="muted small" style="margin:8px 0 0"><b>Looks good</b> deletes the original and these check files, for good.
            <b>Put the original back</b> deletes the dub instead. <b>Redo with review</b> puts the original back and dubs again, stopping for review first.</p>
        </div>
      </div>
    </div>
  </div>`;
  vfWire();
  vfDraw();
}

function vfNow() { const v = $("#vfv"); return v ? v.currentTime : 0; }
function vfSync(force) {
  const v = $("#vfv"), a = $("#vfDub");
  if (!v || !a) return;
  v.muted = VF.hear === "dub";
  if (VF.hear !== "dub") { if (!a.paused) a.pause(); return; }
  a.playbackRate = v.playbackRate;
  if (force || Math.abs(a.currentTime - v.currentTime) > 0.12) a.currentTime = v.currentTime;
  if (v.paused && !a.paused) a.pause();
  if (!v.paused && a.paused) a.play().catch(() => {});
}
function vfSeek(t) {
  const v = $("#vfv");
  v.currentTime = Math.max(0, Math.min(VF.d.duration, t));
  vfSync(true);
  vfDraw();
}
function vfHear(h) {
  VF.hear = h;
  $("#vfHear").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === h));
  $("#vfTag").innerHTML = h === "dub" ? `<span class="chip dub">Hearing the dub</span>` : `<span class="chip plain">Hearing the original</span>`;
  vfSync(true);
}
function vfGo(i) {
  const m = VF.d.moments[i];
  if (!m) return;
  VF.m = i;
  document.querySelectorAll("#vfList .frow").forEach((r) => r.classList.toggle("sel", Number(r.dataset.m) === i));
  const row = document.querySelector(`#vfList .frow[data-m="${i}"]`);
  if (row) row.scrollIntoView({ block: "nearest" });
  vfSeek(Math.max(0, m.t - 2));
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
  const col = { song: rvCss("--keep") || rvCss("--warn"), kept: rvCss("--bad"), voice: rvCss("--accent"), late: rvCss("--muted") };
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
  v.onplay = () => { $("#vfPlay").textContent = "❚❚"; vfSync(true); vfLoop(); };
  v.onpause = () => { $("#vfPlay").textContent = "▶"; vfSync(true); vfDraw(); };
  v.onseeked = () => vfSync(true);
  v.ontimeupdate = () => { vfSync(false); $("#vfTime").textContent = `${fmtT(v.currentTime)} / ${fmtT(d.duration, false)}`; };
  $("#vfPlay").onclick = () => (v.paused ? v.play().catch(() => {}) : v.pause());
  $("#vfBack").onclick = () => vfSeek(vfNow() - 5);
  $("#vfFwd").onclick = () => vfSeek(vfNow() + 5);
  segWire("#vfHear", (val) => vfHear(val));
  segWire("#vfRate", (val) => { v.playbackRate = Number(val); vfSync(true); });
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
  $("#vfGood").onclick = () => decide("good", `Keep this dub and permanently delete the original${d.original.exists ? ` (${(d.original.size / 1e9).toFixed(2)} GB)` : ""} and the check files? This can't be undone.`, "Kept");
  $("#vfRestore").onclick = () => decide("restore", "Put the original back and delete this dub?", "Original put back");
  $("#vfRedo").onclick = () => decide("redo", "Put the original back, delete this dub, and dub the episode again (stopping for review first)?", "Queued again");
  window.onresize = () => vfDraw();
}
document.addEventListener("keydown", (ev) => {
  if (!VF || !location.hash.startsWith("#verify/")) return;
  if ((ev.target.closest && ev.target.closest("input, select, textarea")) || ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const v = $("#vfv"), k = ev.key, n = VF.d.moments.length;
  if (!v) return;
  if (k === " ") { ev.preventDefault(); v.paused ? v.play().catch(() => {}) : v.pause(); }
  else if (k === "ArrowLeft" || k === "ArrowRight") { ev.preventDefault(); vfSeek(vfNow() + (k === "ArrowLeft" ? -1 : 1) * (ev.shiftKey ? 5 : 1)); }
  else if ((k === "n" || k === "N") && n) vfGo((VF.m + 1) % n);
  else if ((k === "p" || k === "P") && n) vfGo((VF.m - 1 + n) % n);
  else if (k === "d" || k === "D") vfHear("dub");
  else if (k === "o" || k === "O") vfHear("orig");
});
