"use strict";
// Ai Dub web UI: plain JavaScript, no build step.

const $ = (sel, el = document) => el.querySelector(sel);
const main = $("#main");
let timer = null;
let status = null;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const gb = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + " GB" : n >= 1e6 ? (n / 1e6).toFixed(1) + " MB" : Math.round((n || 0) / 1e3) + " kB");
const when = (t) => (t ? new Date(t * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "");
const counts = (o) => Object.entries(o || {}).filter(([, v]) => v).map(([k, v]) => `${v} ${k.replace(/_/g, " ")}`).join(", ") || "none";
const dur = (s) => { s = Math.round(s || 0); const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60; return h ? `${h}h ${m}m` : `${m}m ${s % 60}s`; };
const mins = (s) => { const m = Math.max(1, Math.round((s || 0) / 60)); return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min`; };
const plural = (n, one, many = one + "s") => `${n} ${n === 1 ? one : many}`;
const ago = (t) => { const s = Date.now() / 1000 - t; return s < 90 ? "just now" : s < 3600 ? `${Math.round(s / 60)} min ago` : s < 86400 ? `${Math.round(s / 3600)} h ago` : when(t); };

const ICON = {
  folder: `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M3 6.5A2.5 2.5 0 0 1 5.5 4h3.6c.7 0 1.3.3 1.8.8L12.3 6H18.5A2.5 2.5 0 0 1 21 8.5v9a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5z"/></svg>`,
  film: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="M7.5 4.5v15M16.5 4.5v15M3.5 9.5h4M3.5 14.5h4M16.5 9.5h4M16.5 14.5h4"/></svg>`,
};

// "Midnight Diner - S01E07 - Egg Sandwich Bluray-1080p.mkv" as its show, episode code and title.
function episode(path) {
  const file = String(path).split("/").pop();
  let base = file.replace(/\.[A-Za-z0-9]+$/, "").replace(/\s+(Bluray|BluRay|WEBDL|WEB-DL|WEBRip|HDTV|DVD|Remux)[-\w. ]*$/i, "");
  const m = base.match(/^(.*?)\s+-\s+(S\d+E\d+(?:-E?\d+)?)\s*(?:-\s+(.*))?$/i);
  return m ? { show: m[1], code: m[2].toUpperCase(), title: m[3] || "", file } : { show: "", code: "", title: base, file };
}
const epLabel = (path) => { const e = episode(path); return e.code ? `${esc(e.code)}${e.title ? ` · ${esc(e.title)}` : ""}` : esc(e.title); };

async function api(path, opts = {}) {
  const init = { method: opts.method || (opts.body !== undefined ? "POST" : "GET"), headers: {} };
  if (opts.body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(opts.body); }
  const r = await fetch("/api" + path, init);
  const text = await r.text();
  let data = text;
  try { data = JSON.parse(text); } catch { /* plain text */ }
  if (!r.ok) throw new Error((data && data.detail) || r.statusText);
  return data;
}

function toast(msg, bad = false) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false; t.classList.toggle("bad", bad);
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), bad ? 6000 : 4000);
}
const fail = (e) => toast(e.message || String(e), true);

async function refreshStatus() {
  try {
    status = await api("/status");
    // Ai Dub was updated while this page stayed open: reload, so the page and the server
    // match (unless the review page has unsaved changes).
    if (refreshStatus.ui && status.ui && status.ui !== refreshStatus.ui) {
      if (typeof RV !== "undefined" && RV && RV.dirty && location.hash.startsWith("#review/")) toast("Ai Dub was updated: save your changes, then reload the page.");
      else { location.reload(); return; }
    }
    refreshStatus.ui = refreshStatus.ui || status.ui;
    const g = status.gpu, pct = g.total_mb ? g.used_mb / g.total_mb : 0;
    $("#gpu").innerHTML = g.present ? `<span title="${esc(g.name)}">${esc(g.name.replace(/^NVIDIA\s+(GeForce\s+)?/, ""))}</span>
      <span class="meter ${pct > 0.85 ? "hot" : ""}" title="GPU memory in use"><div style="width:${Math.round(pct * 100)}%"></div></span>
      <span class="num">${Math.round(g.used_mb / 1024)} / ${Math.round(g.total_mb / 1024)} GB</span>` : `<span>No GPU visible</span>`;
    const c = status.counts || {}, mine = (c.review || 0) + (c.check || 0);
    const nq = $("#nQueue");
    nq.textContent = mine || c.queue || "";
    nq.className = "count" + (mine ? " warn" : "");
    nq.title = mine ? `${plural(mine, "thing")} waiting for you` : `${plural(c.queue || 0, "job")} in the queue`;
    $("#nOrig").textContent = c.originals || "";
  } catch { $("#gpu").innerHTML = `<span class="tag bad">server unreachable</span>`; }
}

// Queue ------------------------------------------------------------------------------------
let browsePath = "";
let selected = new Set();
const openLogs = new Map();  // job id -> log text, kept while the list refreshes
let polls = 0;

async function pageQueue() {
  main.innerHTML = `
    <div class="page-head"><div class="grow"><h1>Queue</h1>
      <p>Pick episodes to dub. Ai Dub works through them one at a time and asks you when it needs a look.</p></div></div>
    <div id="needs"></div>
    <div class="section-head"><h2>In the queue</h2><span class="grow"></span><span class="small muted" id="qtotal"
      title="From how long earlier episodes took for each step; a stop for review isn't counted"></span></div>
    <div id="jobs"></div>
    <div class="section-head"><h2>Add episodes</h2></div>
    <div class="panel flush">
      <div class="browser-bar"><div class="crumbs grow" id="crumbs"></div></div>
      <div class="browser" id="browser"></div>
      <div class="browser-bar">
        <span class="grow small muted" id="selcount">Nothing selected</span>
        <button class="ghost sm" id="selall">Select all here</button>
        <button class="primary" id="add" disabled>Add to queue…</button>
      </div>
      <div class="addform" id="addform" hidden></div>
    </div>`;
  $("#add").onclick = () => showAddForm();
  $("#selall").onclick = () => {
    const free = (loadBrowser.entries || []).filter((e) => !e.dir && !e.queued);
    const all = free.length && free.every((e) => selected.has(e.path));
    free.forEach((e) => (all ? selected.delete(e.path) : selected.add(e.path)));
    renderBrowser(); toggle();
  };
  polls = 0;
  await Promise.all([loadBrowser(browsePath), loadQueue(), loadNeeds()]);
  timer = setInterval(() => { loadQueue(); if (++polls % 4 === 0) loadNeeds(); }, 2500);
}

async function loadBrowser(path) {
  const data = await api("/browse?path=" + encodeURIComponent(path));
  browsePath = data.path;
  loadBrowser.entries = data.entries;
  loadBrowser.data = data;
  renderBrowser();
}

function renderBrowser() {
  const data = loadBrowser.data;
  if (!data || !$("#crumbs")) return;
  // Breadcrumbs: every folder above is a step back.
  const parts = data.path ? data.path.split("/").filter(Boolean) : [];
  let html = `<button data-go="">Media</button>`;
  let acc = "";
  parts.forEach((p, i) => {
    acc += "/" + p;
    const last = i === parts.length - 1;
    if (i === 0) return;  // "/media" itself isn't a folder to browse: its folders are the roots
    const name = i === 1 ? p[0].toUpperCase() + p.slice(1) : p;
    html += `<span class="sep">›</span>` + (last ? `<b>${esc(name)}</b>` : `<button data-go="${esc(acc)}">${esc(name)}</button>`);
  });
  $("#crumbs").innerHTML = html;
  $("#crumbs").querySelectorAll("[data-go]").forEach((b) => (b.onclick = () => loadBrowser(b.dataset.go).catch(fail)));
  const box = $("#browser");
  const files = data.entries.filter((e) => !e.dir);
  $("#selall").hidden = !files.some((e) => !e.queued);
  box.innerHTML = data.entries.length ? data.entries.map((e, i) => e.dir
    ? `<div class="entry dir" data-i="${i}"><span class="ico">${ICON.folder}</span><span class="name">${esc(e.name)}</span>
        <span>${data.path ? `<button class="sm ghost" data-addall="${i}" title="Every episode in this folder and its season folders that isn't queued or dubbed yet">Add all…</button>` : ""}</span><span class="size"></span></div>`
    : `<div class="entry ${selected.has(e.path) ? "on" : ""} ${e.queued ? "disabled" : ""}" data-i="${i}">
        <input type="checkbox" ${selected.has(e.path) ? "checked" : ""} ${e.queued ? "disabled" : ""} aria-label="Select ${esc(e.name)}">
        <span class="name">${esc(e.name)}</span>
        <span>${e.queued ? `<span class="tag accent">in the queue</span>` : ""}${e.set_aside || e.dubbed ? `<span class="tag ok" title="${e.set_aside ? "Dubbed; its original is set aside next to it" : "Dubbed by Ai Dub"}">dubbed</span>` : ""}</span>
        <span class="size">${gb(e.size)}</span></div>`).join("")
    : `<div class="empty"><b>This folder is empty</b>No video files here.</div>`;
  box.querySelectorAll(".entry").forEach((div) => {
    const e = data.entries[Number(div.dataset.i)];
    if (e.dir) {
      div.onclick = (ev) => { if (!ev.target.closest("[data-addall]")) loadBrowser(e.path).catch(fail); };
      const all = div.querySelector("[data-addall]");
      if (all) all.onclick = () => addFolder(e).catch(fail);
      return;
    }
    if (e.queued) return;
    div.onclick = () => { selected.has(e.path) ? selected.delete(e.path) : selected.add(e.path); renderBrowser(); toggle(); };
  });
}

async function addFolder(e) {
  // A whole show or season: every episode under the folder that isn't queued or dubbed yet.
  const r = await api("/expand?path=" + encodeURIComponent(e.path));
  if (!r.files.length) return toast(r.skipped ? `Every episode in ${e.name} is already queued or dubbed` : `No video files in ${e.name}`);
  selected = new Set(r.files.map((f) => f.path));
  addFolder.from = { name: e.name, n: r.files.length, skipped: r.skipped };
  toggle();
  renderBrowser();
  await showAddForm();
  $("#addform").scrollIntoView({ block: "center", behavior: "smooth" });
}

function toggle() {
  $("#selcount").textContent = selected.size ? `${plural(selected.size, "episode")} selected` : "Nothing selected";
  $("#add").disabled = !selected.size;
  if (!selected.size) $("#addform").hidden = true;
}

async function showAddForm() {
  const form = $("#addform");
  const paths = [...selected];
  form.hidden = false;
  form.innerHTML = `<span class="muted small">Checking ${plural(paths.length, "file")}…</span>`;
  const first = await api("/inspect", { body: { path: paths[0] } }).catch((e) => ({ error: e.message }));
  const profiles = await api("/profiles");
  if (first.error) { form.innerHTML = `<div class="callout bad">${esc(first.error)}</div>`; return; }
  const tracks = first.foreign_tracks || [];
  const note = first.english_audio ? `<div class="callout warn" style="margin-top:10px">This file already has English audio: it is skipped unless you tick Redo.</div>`
    : first.aidub_present ? `<div class="callout warn" style="margin-top:10px">This file already has an Ai Dub track. Tick Redo to rebuild it.</div>` : "";
  form.innerHTML = `
    <div class="opts">
      <label class="opt">Show<input type="text" id="prof" list="proflist" value="${esc(first.profile_guess)}" style="min-width:200px"></label>
      <datalist id="proflist">${profiles.map((p) => `<option value="${esc(p.slug)}">${esc(p.name)}</option>`).join("")}</datalist>
      ${tracks.length > 1 ? `<label class="opt">Audio track<select id="track"><option value="">Default</option>${tracks.map((t) =>
        `<option value="${t.index}">#${t.index} ${esc(t.language || "?")} ${esc(t.title)} ${t.channels}ch</option>`).join("")}</select></label>` : ""}
      <label class="opt">Language<select id="lang"><option value="">Detect</option><option value="ja">Japanese</option><option value="ko">Korean</option>
        <option value="zh">Mandarin</option><option value="yue">Cantonese</option></select></label>
      <label class="opt">Review before dubbing<select id="review"><option value="">Show's setting</option><option value="off">Off</option>
        <option value="unsure">When unsure</option><option value="always">Always</option></select></label>
      <label style="padding-bottom:7px"><input type="checkbox" id="redo"> Redo</label>
      <label style="padding-bottom:7px" title="Saves the review choice as this show's setting, so later episodes use it too"><input type="checkbox" id="showrev" disabled> Use this review setting for every episode of the show</label>
      <span class="grow"></span>
      <button class="ghost" id="canceladd">Cancel</button>
      <button class="primary" id="confirmadd">Add ${plural(paths.length, "episode")}</button>
    </div>
    ${addFolder.from && addFolder.from.n === paths.length ? `<div class="callout" style="margin-top:10px">All of <b>${esc(addFolder.from.name)}</b>: ${plural(paths.length, "episode")}${addFolder.from.skipped ? `, leaving out ${addFolder.from.skipped} already queued or dubbed` : ""}. They are dubbed one at a time, in order.</div>` : ""}
    <p class="small muted" style="margin:10px 0 0">The show is guessed from the folder name${paths.length > 1 ? " of the first file; leave it to let each file use its own show" : ""}.
      ${esc(dur(first.duration))} long, ${esc(first.container)}.</p>
    ${note}`;
  $("#canceladd").onclick = () => (form.hidden = true);
  $("#review").onchange = () => { $("#showrev").disabled = !$("#review").value; if (!$("#review").value) $("#showrev").checked = false; };
  $("#confirmadd").onclick = async () => {
    const prof = $("#prof").value.trim();
    const body = { paths, profile: paths.length > 1 && prof === first.profile_guess ? null : prof || null,
      audio_stream: $("#track") && $("#track").value !== "" ? Number($("#track").value) : null,
      language: $("#lang").value || null, redo: $("#redo").checked, review: $("#review").value || null,
      show_review: $("#showrev").checked };
    try {
      const res = await api("/jobs", { body });
      toast(`Added ${plural(res.added.length, "episode")}` + (res.errors.length ? `; ${res.errors.length} not added: ${res.errors[0].error}` : ""), !!res.errors.length);
    } catch (e) { return fail(e); }
    selected.clear(); addFolder.from = null; toggle(); form.hidden = true;
    await loadBrowser(browsePath); await loadQueue(); refreshStatus();
  };
}

async function loadNeeds() {
  // What waits for the user: episodes stopped for review, and finished dubs to check.
  const [items, jobs] = await Promise.all([api("/verify").catch(() => []), api("/jobs?view=queue").catch(() => [])]);
  const box = $("#needs");
  if (!box) return;
  const reviews = jobs.filter((j) => j.status === "review");
  if (!items.length && !reviews.length) { box.innerHTML = ""; return; }
  box.innerHTML = `<div class="section-head"><h2>Waiting for you</h2><span class="tag warn">${items.length + reviews.length}</span></div>
    <div class="needs-list">
    ${reviews.map((j) => `<div class="panel job needs" data-id="${j.id}">
      <div><div class="title">${epLabel(j.path)}</div>
        <div class="sub"><span class="ep-show">${esc(episode(j.path).show || j.profile)}</span><span class="tag warn">Review before dubbing</span>
          <span>${esc(j.message.replace(/^waiting for review:?\s*/, "") || "nothing flagged")}</span></div></div>
      <div class="btn-group"><button class="ghost sm" data-act="cancel" data-id="${j.id}">Remove</button>
        <a class="button primary" href="#review/${j.id}">Review</a></div></div>`).join("")}
    ${items.map((x) => `<div class="panel job needs">
      <div><div class="title">${epLabel(x.name)}</div>
        <div class="sub"><span class="ep-show">${esc(episode(x.name).show)}</span><span class="tag ok">Dubbed</span>
          <span>finished ${esc(ago(x.finished))} · ${plural(x.moments, "moment")} to listen to · original (${gb(x.original_size)}) set aside</span></div></div>
      <div class="btn-group"><a class="button" href="#review/${x.id}" title="The full review page: change anything and redub">Review &amp; change</a>
        <a class="button primary" href="#verify/${x.id}">Check the dub</a></div></div>`).join("")}
    </div>`;
  box.querySelectorAll("[data-act=cancel]").forEach((b) => (b.onclick = async () => {
    if (!confirm("Remove this episode from the queue? Its working files are deleted.")) return;
    await api(`/jobs/${b.dataset.id}`, { method: "DELETE" }).catch(fail);
    loadNeeds(); loadQueue(); refreshStatus();
  }));
}

const AFTER_REVIEW = ["review", "translate", "speak", "check", "finalize", "condense", "bwe", "mix", "deliver"];

async function loadQueue() {
  const all = await api("/jobs?view=queue");
  const box = $("#jobs");
  if (!box) return;
  const jobs = all.filter((j) => j.status !== "review");
  const total = jobs.reduce((a, j) => a + (j.eta || 0), 0);
  const head = $("#qtotal");
  if (head) head.textContent = total ? `about ${mins(total)} for everything in the queue` : "";
  if (!jobs.length) {
    box.innerHTML = `<div class="panel empty"><b>Nothing is being dubbed</b>${all.length ? "Episodes waiting for your review are above." : "Pick episodes below and add them to the queue."}</div>`;
    return;
  }
  if (++loadQueue.n % 4 === 0) for (const id of openLogs.keys()) api(`/jobs/${id}/log`).then((t) => openLogs.set(id, t)).catch(() => {});
  let qpos = 0;
  box.innerHTML = jobs.map((j, i) => {
    const live = j.live || {}, pct = Math.round((j.progress || 0) * 100);
    const st = j.status === "waiting" ? `<span class="tag warn"><span class="dot"></span>Waiting for the GPU</span>`
      : j.status === "running" ? `<span class="tag ok live"><span class="dot"></span>Dubbing</span>`
      : `<span class="tag">${(qpos = jobs.slice(0, i).filter((x) => x.status === "queued").length) ? `${qpos + 1}${["st", "nd", "rd"][qpos] || "th"} in line` : "Next up"}</span>`;
    const early = !AFTER_REVIEW.includes(j.step || "");
    const holding = (j.options || {}).review === "always";
    const step = live.steps ? `Step ${live.step_index + 1} of ${live.steps}: <b>${esc(j.step_title)}</b>` : j.step_title ? `<b>${esc(j.step_title)}</b>` : "";
    return `<div class="panel job" data-id="${j.id}">
      <div><div class="title">${epLabel(j.path)}</div>
        <div class="sub"><span class="ep-show">${esc(episode(j.path).show || j.profile)}</span>${st}
          ${j.status === "queued" && j.eta != null ? `<span>about ${mins(j.eta)}</span>` : ""}
          ${holding && early ? `<span class="tag accent" title="It pauses for your review before anything is translated">Will stop for review</span>` : ""}</div>
        ${j.status !== "queued" ? `<div class="meta"><span>${step}${j.message ? ` · ${esc(j.message)}` : ""}</span>
          <span class="num nowrap">${j.eta != null ? `about ${mins(j.eta)} left · ` : ""}${pct} %</span></div>
          <div class="progress"><div style="width:${pct}%"></div></div>` : ""}</div>
      <div class="btn-group">
        ${j.status === "running" ? `<a class="button sm" href="#review/${j.id}" title="See what Ai Dub has decided so far">Look</a>` : ""}
        ${early && !holding ? `<button class="sm" data-act="hold" title="Pause this episode at its review step, even if nothing is flagged">Stop for review</button>` : ""}
        ${j.status === "queued" ? `<button class="sm" data-act="up" title="Move up" ${i === 0 || jobs[i - 1].status !== "queued" ? "disabled" : ""}>↑</button><button class="sm" data-act="down" title="Move down" ${i === jobs.length - 1 ? "disabled" : ""}>↓</button>` : ""}
        <button class="sm ${openLogs.has(j.id) ? "on" : ""}" data-act="log">${openLogs.has(j.id) ? "Hide log" : "Log"}</button>
        <button class="sm ghost" data-act="cancel" style="color:var(--bad)">${j.status === "queued" ? "Remove" : "Cancel"}</button>
      </div>
      ${openLogs.has(j.id) ? `<pre class="log">${esc(openLogs.get(j.id) || "Loading…")}</pre>` : ""}</div>`;
  }).join("");
  box.querySelectorAll("pre.log").forEach((p) => (p.scrollTop = p.scrollHeight));
  box.querySelectorAll("button[data-act]").forEach((b) => {
    const id = Number(b.closest(".job").dataset.id);
    b.onclick = async () => {
      const act = b.dataset.act;
      try {
        if (act === "up" || act === "down") await api(`/jobs/${id}/move`, { body: { direction: act === "up" ? -1 : 1 } });
        if (act === "cancel") {
          if (!confirm("Stop this episode and remove it from the queue? Its working files are deleted.")) return;
          await api(`/jobs/${id}`, { method: "DELETE" });
        }
        if (act === "hold") { await api(`/jobs/${id}/review/hold`, { body: {} }); toast("It will stop for review before anything is translated"); }
        if (act === "log") {
          if (openLogs.has(id)) openLogs.delete(id);
          else { openLogs.set(id, ""); openLogs.set(id, await api(`/jobs/${id}/log`)); }
        }
      } catch (e) { fail(e); }
      loadQueue(); refreshStatus();
    };
  });
}
loadQueue.n = 0;

// History ------------------------------------------------------------------------------------
const HIST = { filter: "all", q: "", open: new Set() };

async function pageHistory() {
  const jobs = await api("/jobs?view=history");
  pageHistory.jobs = jobs;
  const n = (f) => jobs.filter(f).length;
  main.innerHTML = `
    <div class="page-head"><div class="grow"><h1>History</h1><p>Every episode Ai Dub has finished, skipped or stopped.</p></div>
      <div class="filters">
        <div class="seg" id="hfilter">${[["all", "All", jobs.length], ["check", "To check", n((j) => (j.record || {}).verify === "pending")],
          ["done", "Done", n((j) => j.status === "done")], ["failed", "Failed", n((j) => j.status === "failed")]]
          .map(([k, t, c]) => `<button data-v="${k}" class="${HIST.filter === k ? "on" : ""}">${t} <span class="faint">${c}</span></button>`).join("")}</div>
        <input type="search" id="hsearch" placeholder="Search episodes…" value="${esc(HIST.q)}"></div></div>
    <div class="panel flush tablewrap"><table class="hist" id="htable"></table></div>`;
  $("#hfilter").querySelectorAll("button").forEach((b) => (b.onclick = () => { HIST.filter = b.dataset.v; pageHistory(); }));
  $("#hsearch").oninput = (e) => { HIST.q = e.target.value; renderHistory(); };
  renderHistory();
}

function renderHistory() {
  const jobs = (pageHistory.jobs || []).filter((j) => {
    const r = j.record || {};
    if (HIST.filter === "check" && r.verify !== "pending") return false;
    if (HIST.filter === "done" && j.status !== "done") return false;
    if (HIST.filter === "failed" && j.status !== "failed") return false;
    return !HIST.q || j.path.toLowerCase().includes(HIST.q.toLowerCase());
  });
  const table = $("#htable");
  if (!jobs.length) { table.innerHTML = `<tr><td><div class="empty"><b>Nothing here</b>${HIST.q || HIST.filter !== "all" ? "No episode matches." : "Finished episodes appear here."}</div></td></tr>`; return; }
  const result = (j) => {
    const r = j.record || {};
    const v = { pending: `<span class="tag warn">Not checked yet</span>`, good: `<span class="tag ok">Kept · original deleted</span>`,
      restored: `<span class="tag">Original put back</span>`, redo: `<span class="tag">Redone</span>` }[r.verify];
    const s = { done: `<span class="tag ok">Dubbed</span>`, failed: `<span class="tag bad">Failed</span>`, skipped: `<span class="tag">Skipped</span>`,
      cancelled: `<span class="tag">Cancelled</span>` }[j.status] || `<span class="tag">${esc(j.status)}</span>`;
    return v && j.status === "done" ? v : s;
  };
  table.innerHTML = `<thead><tr><th>Episode</th><th>Result</th><th>Dub</th><th>Finished</th><th></th></tr></thead><tbody>` + jobs.map((j) => {
    const r = j.record || {}, qc = r.qc || {};
    const took = j.started && j.finished ? mins(j.finished - j.started) : "";
    const stats = j.status === "done" && qc.dubbed_lines != null
      ? `<div class="stats"><span><b>${qc.dubbed_lines}</b> lines</span><span><b>${((qc.fallback_rate || 0) * 100).toFixed(1)} %</b> left in original</span>${took ? `<span>took <b>${took}</b></span>` : ""}</div>` : "";
    const open = HIST.open.has(j.id);
    return `<tr data-id="${j.id}"><td><div class="ep">${epLabel(j.path)}</div><div class="msg">${esc(episode(j.path).show || j.profile)}${
        j.status !== "done" || r.verify === "redo" ? ` · ${esc(j.message)}` : ""}</div>
        ${qc.subtitle_track && qc.subtitle_track.trusted === false && qc.subtitle_track.why !== "no English subtitle"
          ? `<div class="msg" style="color:var(--warn)">Dubbed from the speech: the English subtitles weren't good enough (${esc(qc.subtitle_track.why)})</div>` : ""}</td>
      <td>${result(j)}</td><td>${stats}</td><td class="small muted nowrap">${esc(when(j.finished))}</td>
      <td class="actions"><div class="btn-group">
        ${r.verify === "pending" ? `<a class="button sm primary" href="#verify/${j.id}">Check</a>` : ""}
        ${j.status === "failed" ? `<button class="sm" data-act="retry" title="Continue from the step that failed">Retry</button>` : ""}
        <button class="sm ghost" data-act="details">${open ? "Hide" : "Details"}</button>
        <button class="sm ghost" data-act="remove" title="Remove from history (deletes any working files)">✕</button></div></td></tr>
      ${open ? `<tr class="detail"><td colspan="5">${historyDetail(j)}</td></tr>` : ""}`;
  }).join("") + "</tbody>";
  table.querySelectorAll("button[data-act]").forEach((b) => {
    const id = Number(b.closest("tr").dataset.id);
    b.onclick = async () => {
      const act = b.dataset.act;
      if (act === "details") { HIST.open.has(id) ? HIST.open.delete(id) : HIST.open.add(id); renderHistory(); return; }
      try {
        if (act === "retry") { await api(`/jobs/${id}/retry`, { body: {} }); toast("Queued again from the failed step"); }
        if (act === "remove") {
          const j = pageHistory.jobs.find((x) => x.id === id);
          const pending = (j.record || {}).verify === "pending";
          if (!confirm(pending ? "Remove this episode from history? Its check and review files are deleted (the dub and the set-aside original stay; manage the original under Originals)."
            : "Remove this episode from history? Any working files left are deleted.")) return;
          await api(`/jobs/${id}`, { method: "DELETE" });
        }
      } catch (e) { return fail(e); }
      pageHistory(); refreshStatus();
    };
  });
}

function historyDetail(j) {
  const r = j.record || {}, qc = r.qc || {};
  if (!r.version) return `<div class="kv"><span>Message</span><span>${esc(j.message)}</span>
    ${j.status === "failed" ? `<span>Working files kept</span><span>${gb(j.work_bytes || 0)}</span>` : ""}</div>${r.error ? `<pre>${esc(r.error)}</pre>` : ""}`;
  return `<div class="kv">
      <span>File</span><span style="overflow-wrap:anywhere">${esc(j.path)}</span>
      <span>Message</span><span>${esc(j.message)}</span>
      <span>Show profile</span><span><a href="#profiles/${encodeURIComponent(j.profile)}">${esc(j.profile)}</a></span>
      <span>Text from</span><span>${esc(r.text_source || "")}</span>
      <span>Lines</span><span>${esc(counts(qc.lines))}</span>
      <span>Timing</span><span>${esc(counts(qc.fit))}</span>
      <span>Left in the original</span><span>${((qc.fallback_rate || 0) * 100).toFixed(1)} %</span>
      <span>Minor-speaker voices</span><span>${esc(counts(qc.minor_speaker_substitutions))}</span>
      <span>Subtitle tracks added</span><span>${esc(((r.delivered || {}).subtitle_tracks || []).map((s) => s.title).join(", ") || "none")}</span>
      <span>Step times</span><span>${esc(Object.entries(r.timings || {}).map(([k, v]) => `${k} ${dur(v)}`).join(", "))}</span>
      <span>Ai Dub</span><span>${esc(r.version)}</span>
      <span>Models</span><span class="small">${esc(Object.entries(r.models || {}).map(([k, v]) => `${k}: ${v}`).join("; "))}</span>
    </div>
    ${(qc.fallback_lines || []).length ? `<details style="margin-top:10px"><summary>${plural(qc.fallback_lines.length, "line")} kept in the original language</summary>
      <pre>${esc(qc.fallback_lines.map((l) => `${dur(l.start).padStart(7)}  ${l.reason}`).join("\n"))}</pre></details>` : ""}
    ${r.error ? `<pre>${esc(r.error)}</pre>` : ""}`;
}

// Shows (profiles) ------------------------------------------------------------------------------
async function pageProfiles(slug) {
  if (slug) return pageProfile(slug);
  const list = await api("/profiles");
  main.innerHTML = `<div class="page-head"><div class="grow"><h1>Shows</h1>
      <p>What Ai Dub remembers per show: voices, names and terms, and its rules. Each dubbed episode adds to it.</p></div></div>` +
    (list.length ? `<div class="panel flush tablewrap"><table>
    <thead><tr><th>Show</th><th>Episodes</th><th>Voices</th><th>Glossary</th><th>Size</th></tr></thead><tbody>
    ${list.map((p) => `<tr style="cursor:pointer" data-slug="${esc(p.slug)}"><td><a href="#profiles/${encodeURIComponent(p.slug)}"><b>${esc(p.name)}</b></a></td>
      <td class="num">${p.episodes}</td><td class="num">${p.characters}</td><td class="num">${p.glossary} terms</td><td class="num muted">${gb(p.size)}</td></tr>`).join("")}</tbody></table></div>`
      : `<div class="panel empty"><b>No shows yet</b>One is made for each show you dub.</div>`);
  main.querySelectorAll("tr[data-slug]").forEach((tr) => (tr.onclick = (e) => { if (e.target.tagName !== "A") location.hash = `#profiles/${encodeURIComponent(tr.dataset.slug)}`; }));
}

const VOICES_SHOWN = 12;

async function pageProfile(slug) {
  const p = await api("/profiles/" + encodeURIComponent(slug));
  const others = (cid) => p.characters.filter((c) => c.id !== cid).map((c) => `<option value="${c.id}">${esc(c.name)}</option>`).join("");
  const chars = [...p.characters].sort((a, b) => (b.clean_seconds || 0) - (a.clean_seconds || 0));
  const gloss = Object.entries(p.glossary).sort((a, b) => (b[1].count || 0) - (a[1].count || 0));
  main.innerHTML = `
    <div class="page-head"><div class="grow"><div class="small"><a href="#profiles">Shows</a> <span class="faint">›</span></div><h1>${esc(p.name)}</h1>
      <p>${plural(p.episodes.length, "episode")} · ${plural(p.characters.length, "voice")} · ${plural(gloss.length, "glossary term")} · ${gb(p.size)}</p></div>
      <button class="danger" id="delprof">Delete show</button></div>
    <div class="panel"><h3>Translation</h3>
      <div class="field-grid" style="margin-top:10px">
        <span class="lab">Honorifics</span><span><select id="hon"><option value="keep">Keep them (-san, -kun…)</option><option value="drop">Drop them</option></select></span>
        <span class="lab">Style</span><span><select id="style"><option value="natural">Natural English</option><option value="close">Close to the original</option></select></span>
        <span class="help">Both apply to lines translated from the speech. Lines dubbed from English subtitles keep the subtitles' own wording.</span>
      </div></div>
    <div class="panel"><h3>Show rules</h3><div id="rules"><span class="muted small">Loading…</span></div></div>
    <div class="section-head"><h2>Voices</h2><span class="faint small grow">Clips each character's dub voice is cloned from; most audio first</span>
      <input type="search" id="vfind" placeholder="Find a voice…"></div>
    <div class="chars">${chars.map((c, i) => `
      <div class="panel char" data-cid="${c.id}" data-name="${esc(c.name.toLowerCase())}" ${i >= VOICES_SHOWN && !pageProfile.all ? "hidden" : ""}>
        <div class="head"><input type="text" value="${esc(c.name)}" class="cname" aria-label="Name"><button class="sm" data-act="rename">Rename</button></div>
        <div class="small muted" style="margin-top:6px">${esc([c.gender, c.age].filter(Boolean).join(", ") || "Gender and age unknown")} · ${c.clean_seconds} s of clean audio
          ${c.voice_of ? ` · speaks with ${esc((p.characters.find((x) => x.id === c.voice_of) || {}).name || c.voice_of)}'s voice` : ""}</div>
        <div class="refs">${(c.refs || []).map((r) => `<div class="ref" data-ref="${esc(r.file)}">
          <audio controls preload="none" src="/api/profiles/${encodeURIComponent(p.slug)}/ref?file=${encodeURIComponent(r.file)}"></audio>
          <span class="small muted num nowrap">${Number(r.duration).toFixed(1)} s</span>
          <label class="small" title="Keep: never replace this clip with a newer one"><input type="checkbox" data-act="lock" ${r.locked ? "checked" : ""}> Keep</label>
          <input type="checkbox" class="pick" title="Select, to split selected clips off as another character" aria-label="Select clip">
          <button data-act="rmref" class="sm ghost" title="Remove this clip">✕</button></div>`).join("") || `<p class="small muted">No clean clips yet.</p>`}</div>
        <div class="tools">
          ${p.characters.length > 1 ? `<select class="other sm" style="max-width:150px">${others(c.id)}</select><button class="sm" data-act="merge">Merge into this</button>` : ""}
          <button class="sm" data-act="split" title="Make the selected clips a new character">Split off selected</button>
          <span class="grow"></span><button data-act="delchar" class="sm ghost" style="color:var(--bad)">Delete</button>
        </div>
      </div>`).join("") || `<div class="panel empty"><b>No voices yet</b>They are made from the first dubbed episode.</div>`}</div>
    ${chars.length > VOICES_SHOWN ? `<div class="row" style="justify-content:center"><button id="vall">${pageProfile.all ? "Show fewer" : `Show all ${chars.length} voices`}</button></div>` : ""}
    <div class="section-head"><h2>Glossary</h2><span class="faint small">How names and terms are always written in English</span></div>
    <div class="panel flush">
      <div class="browser-bar"><input type="text" id="gsrc" placeholder="Term as said (e.g. 吾郎)"><input type="text" id="gen" placeholder="English (e.g. Goro)">
        <button id="gadd">Add or change</button><span class="grow"></span><input type="search" id="gfind" placeholder="Filter…"></div>
      <div class="tablewrap" style="max-height:480px;overflow:auto"><table id="gtable"><thead><tr><th>Term</th><th>English</th><th>Seen</th><th></th></tr></thead><tbody>
      ${gloss.map(([k, v]) => `<tr data-g="${esc((k + " " + v.en).toLowerCase())}"><td>${esc(k)}</td><td>${esc(v.en)}${v.edited ? ' <span class="tag accent">yours</span>' : ""}</td>
        <td class="num muted">${v.count || 0}×</td><td class="actions"><button class="sm ghost" data-gdel="${esc(k)}" title="Remove">✕</button></td></tr>`).join("")
        || `<tr><td colspan="4" class="muted">No terms yet.</td></tr>`}</tbody></table></div>
    </div>`;
  showRules(slug);
  $("#hon").value = p.settings.honorifics || "keep";
  $("#style").value = p.settings.style || "natural";
  const save = () => api("/profiles/" + encodeURIComponent(slug), { method: "PATCH", body: { settings: { honorifics: $("#hon").value, style: $("#style").value } } }).then(() => toast("Saved")).catch(fail);
  $("#hon").onchange = save; $("#style").onchange = save;
  $("#vfind").oninput = (e) => {
    const q = e.target.value.toLowerCase().trim();
    main.querySelectorAll(".char[data-cid]").forEach((c, i) => (c.hidden = q ? !c.dataset.name.includes(q) : i >= VOICES_SHOWN && !pageProfile.all));
    if ($("#vall")) $("#vall").hidden = !!q;
  };
  if ($("#vall")) $("#vall").onclick = () => { pageProfile.all = !pageProfile.all; pageProfile(slug); };
  $("#gfind").oninput = (e) => { const q = e.target.value.toLowerCase(); $("#gtable").querySelectorAll("tr[data-g]").forEach((tr) => (tr.hidden = !!q && !tr.dataset.g.includes(q))); };
  $("#delprof").onclick = async () => {
    if (!confirm(`Delete everything Ai Dub remembers about "${p.name}"? Its voices and glossary are removed and the next episode starts fresh.`)) return;
    await api(`/profiles/${encodeURIComponent(slug)}?confirm=true`, { method: "DELETE" }).catch(fail);
    location.hash = "#profiles";
  };
  $("#gadd").onclick = async () => {
    if (!$("#gsrc").value.trim()) return toast("Enter the term as it's said first");
    await api(`/profiles/${encodeURIComponent(slug)}/glossary`, { method: "PUT", body: { source: $("#gsrc").value, en: $("#gen").value } }).catch(fail);
    pageProfile(slug);
  };
  main.querySelectorAll("[data-gdel]").forEach((b) => (b.onclick = async () => {
    await api(`/profiles/${encodeURIComponent(slug)}/glossary?source=${encodeURIComponent(b.dataset.gdel)}`, { method: "DELETE" }).catch(fail);
    pageProfile(slug);
  }));
  main.querySelectorAll("[data-cid]").forEach((card) => {
    const cid = card.dataset.cid;
    const act = (body) => api(`/profiles/${encodeURIComponent(slug)}/characters/${cid}`, { body }).then(() => pageProfile(slug)).catch(fail);
    card.querySelectorAll("[data-act]").forEach((el) => {
      const a = el.dataset.act;
      const ref = el.closest(".ref") ? el.closest(".ref").dataset.ref : null;
      if (a === "lock") el.onchange = () => act({ action: "lock_ref", ref, locked: el.checked });
      else el.onclick = () => {
        if (a === "rename") act({ action: "rename", name: $(".cname", card).value });
        if (a === "rmref" && confirm("Remove this clip?")) act({ action: "remove_ref", ref });
        if (a === "merge" && confirm("Merge the chosen character into this one? Their clips join this voice.")) act({ action: "merge", other: $(".other", card).value });
        if (a === "split") {
          const refs = [...card.querySelectorAll(".ref")].filter((r) => $(".pick", r).checked).map((r) => r.dataset.ref);
          if (!refs.length) return toast("Select the clips that belong to another character first");
          act({ action: "split", refs, name: prompt("Name for the new character", "") || "" });
        }
        if (a === "delchar" && confirm("Delete this character and its clips?")) act({ action: "delete", confirm: true });
      };
    });
    $(".cname", card).onkeydown = (e) => { if (e.key === "Enter") act({ action: "rename", name: e.target.value }); };
  });
}

async function showRules(slug) {
  const box = $("#rules");
  const r = await api(`/profiles/${encodeURIComponent(slug)}/rules`);
  const names = { romaji_lyrics: "Romaji lines are lyrics", italic_quoted_lyrics: "Italics in quotation marks are lyrics",
    tilde_signs: "“~ text ~” is a sign, not spoken", narrator_dubbed: "The narrator's opening is dubbed" };
  box.innerHTML = `<div class="field-grid" style="margin-top:10px">
      <span class="lab">Review before dubbing</span><span><select id="ruleMode"><option value="">Default (${esc(r.default_review)})</option>
        <option value="off">Off: never stop</option><option value="unsure">When unsure</option><option value="always">Always</option></select></span>
      <span class="lab">How its subtitles are written</span><span class="row" style="gap:6px 18px">${Object.entries(names).map(([k, t]) =>
        `<label class="small"><input type="checkbox" data-habit="${k}" ${r.habits[k] ? "checked" : ""}> ${t}</label>`).join("")}</span>
      <span class="lab">Opening and ending</span><span>${r.regions.length ? `<table class="small"><tbody>
        ${r.regions.map((x) => `<tr><td>${esc(x.label || x.kind)}</td><td class="num">${x.seconds} s</td><td class="muted">from ${esc(x.episode)}</td>
          <td class="actions"><button class="sm ghost" data-unrule="${esc(x.id)}" title="Remove">✕</button></td></tr>`).join("")}</tbody></table>`
      : `<span class="small muted">None marked yet. Mark one on an episode's review page and tick “Use for every episode”.</span>`}</span>
    </div>`;
  $("#ruleMode").value = r.review || "";
  const save = (body) => api(`/profiles/${encodeURIComponent(slug)}/rules`, { method: "PUT", body }).then(() => toast("Saved")).catch(fail);
  $("#ruleMode").onchange = () => save({ review: $("#ruleMode").value });
  box.querySelectorAll("[data-habit]").forEach((cb) => (cb.onchange = () => {
    const habits = {};
    box.querySelectorAll("[data-habit]").forEach((x) => (habits[x.dataset.habit] = x.checked));
    save({ habits });
  }));
  box.querySelectorAll("[data-unrule]").forEach((b) => (b.onclick = async () => {
    if (!confirm("Remove this rule? Later episodes won't look for it.")) return;
    await save({ remove_regions: [b.dataset.unrule] });
    showRules(slug);
  }));
}

// Originals ------------------------------------------------------------------------------------
async function pageOriginals() {
  const data = await api("/originals");
  main.innerHTML = `<div class="page-head"><div class="grow"><h1>Originals</h1>
      <p>Each dubbed file's original is kept next to it as <code>.aidub-original</code> until you delete it: from the check page once you've listened
        to the dub, or here. Restore puts an original back under its own name and sets the dubbed copy aside.</p></div></div>
    <div class="panel flush">
      <div class="browser-bar"><span class="grow"><b>${plural(data.items.length, "file")}</b> <span class="muted">· ${gb(data.total)} in total</span></span>
        <button class="sm" id="scan" title="Find set-aside files in the media folders">Rescan folders</button>
        <button class="sm danger" id="delsel" disabled>Delete selected</button>
        <button class="sm danger" id="delall" ${data.items.length ? "" : "disabled"}>Delete all</button></div>
      ${data.items.length ? `<div class="tablewrap"><table><thead><tr><th style="width:36px"><input type="checkbox" id="pickall" aria-label="Select all"></th>
        <th>File</th><th>Kind</th><th>Size</th><th>Set aside</th><th></th></tr></thead><tbody>
      ${data.items.map((o) => `<tr data-id="${o.id}"><td><input type="checkbox" class="pick"></td>
        <td style="overflow-wrap:anywhere"><b>${epLabel(o.path)}</b><div class="small faint">${esc(o.path)}</div></td>
        <td><span class="tag ${o.kind === "original" ? "" : "accent"}">${esc(o.kind)}</span></td><td class="num nowrap">${gb(o.size)}</td>
        <td class="small muted nowrap">${esc(when(o.created))}</td>
        <td class="actions"><button class="sm" data-act="restore">Restore</button></td></tr>`).join("")}</tbody></table></div>`
        : `<div class="empty"><b>Nothing set aside</b>Originals show up here after a dub, until you delete them.</div>`}
    </div>`;
  const picks = () => [...main.querySelectorAll("tr[data-id]")].filter((r) => $(".pick", r).checked).map((r) => Number(r.dataset.id));
  const sync = () => { const n = picks().length; $("#delsel").disabled = !n; $("#delsel").textContent = n ? `Delete ${n} selected` : "Delete selected"; };
  main.querySelectorAll(".pick").forEach((cb) => (cb.onchange = sync));
  if ($("#pickall")) $("#pickall").onchange = (e) => { main.querySelectorAll(".pick").forEach((cb) => (cb.checked = e.target.checked)); sync(); };
  const del = async (ids) => {
    const size = data.items.filter((o) => ids.includes(o.id)).reduce((a, o) => a + o.size, 0);
    if (!confirm(`Permanently delete ${plural(ids.length, "file")}, ${gb(size)}? This cannot be undone.`)) return;
    try { const r = await api("/originals/delete", { body: { ids, confirm: true } }); toast(`Deleted ${plural(r.deleted, "file")}, freed ${gb(r.freed)}`); } catch (e) { fail(e); }
    pageOriginals(); refreshStatus();
  };
  $("#delsel").onclick = () => del(picks());
  $("#delall").onclick = () => del(data.items.map((o) => o.id));
  $("#scan").onclick = async () => { try { const r = await api("/originals/scan", { body: {} }); toast(r.added ? `Found ${r.added} more` : "Nothing new found"); } catch (e) { fail(e); } pageOriginals(); refreshStatus(); };
  main.querySelectorAll("[data-act=restore]").forEach((b) => (b.onclick = async () => {
    if (!confirm("Restore this file under its own name? The current file is set aside in its place.")) return;
    try { toast((await api(`/originals/${b.closest("tr").dataset.id}/restore`, { body: {} })).message); } catch (e) { fail(e); }
    pageOriginals();
  }));
}

// Models ------------------------------------------------------------------------------------
async function pageModels() {
  const list = await api("/models");
  const busy = list.some((m) => m.job === "queued" || m.job === "installing");
  const missing = list.filter((m) => !(m.installed && m.weights)).length;
  main.innerHTML = `<div class="page-head"><div class="grow"><h1>Models</h1>
      <p>Nothing is bundled with Ai Dub. Each model is downloaded from its official source into the models folder and runs in its own
        environment; downloads total tens of GB, and each model keeps its own licence.</p></div>
      <button id="voices" title="Design the small library of generic voices used when a speaker has too little audio">Make generic voices</button>
      <button class="primary" id="installall" ${missing && !busy ? "" : "disabled"}>Install all missing</button></div>
    <div class="callout ${missing ? "warn" : ""}" style="margin-bottom:14px">${missing ? `${missing} of ${list.length} models aren't installed yet: episodes can't be dubbed until they are.`
      : `All ${list.length} models are installed and ready.`}</div>
    <div class="panel flush tablewrap"><table>
      <thead><tr><th>Step</th><th>Model</th><th>Licence</th><th>Size</th><th>Status</th><th></th></tr></thead><tbody>
      ${list.map((m) => {
        const st = m.job === "installing" ? `<span class="tag warn live"><span class="dot"></span>Installing</span>` : m.job === "queued" ? `<span class="tag">Queued</span>`
          : m.job === "failed" ? `<span class="tag bad">Failed</span>` : m.installed && m.weights ? `<span class="tag ok">Ready</span>`
          : m.installed ? `<span class="tag warn">Weights missing</span>` : `<span class="tag">Not installed</span>`;
        return `<tr data-name="${esc(m.name)}"><td><b>${esc(m.title)}</b></td><td class="small">${esc(m.models)}<div class="faint">${esc(m.model)}</div></td>
          <td class="small muted">${esc(m.licence)}</td><td class="small muted nowrap">${esc(m.download)}</td><td>${st}</td>
          <td class="actions"><div class="btn-group"><button class="sm ghost" data-act="log">Log</button>
          <button class="sm" data-act="install" ${m.job === "installing" || m.job === "queued" ? "disabled" : ""}>${m.installed && m.weights ? "Reinstall" : "Install"}</button></div></td></tr>
          <tr hidden class="logrow"><td colspan="6"><pre></pre></td></tr>`;
      }).join("")}</tbody></table></div>`;
  $("#installall").onclick = async () => { await api("/models/install", { body: { names: [] } }).catch(fail); pageModels(); };
  $("#voices").onclick = async () => {
    if (!confirm("Design the small library of generic voices (needs the voice-design model installed)?")) return;
    try { await api("/voices/design", { body: { force: false } }); toast("Started; see the voice-design log"); } catch (e) { fail(e); }
  };
  main.querySelectorAll("tr[data-name] button").forEach((b) => (b.onclick = async () => {
    const tr = b.closest("tr"), name = tr.dataset.name;
    if (b.dataset.act === "install") {
      if (b.textContent === "Reinstall" && !confirm("Download and install this model again?")) return;
      await api("/models/install", { body: { names: [name] } }).catch(fail); pageModels(); return;
    }
    const row = tr.nextElementSibling; row.hidden = !row.hidden;
    b.textContent = row.hidden ? "Log" : "Hide log";
    if (!row.hidden) $("pre", row).textContent = (await api(`/models/${name}/log`).catch(() => "")) || "No log yet.";
  }));
  if (busy) timer = setTimeout(() => location.hash === "#models" && pageModels(), 4000);
}

// Settings ------------------------------------------------------------------------------------
async function pageSettings() {
  const s = await api("/settings");
  const bz = ((s.effective.subtitles || {}).providers || {}).bazarr || {};
  main.innerHTML = `<div class="page-head"><div class="grow"><h1>Settings</h1><p>Changes are saved to config.yaml in the app data folder.</p></div></div>
    <div class="panel"><h3>Review before dubbing</h3>
      <p class="small muted">An episode can stop once Ai Dub has worked out its songs and speakers, so you can check them on the review page before
        anything is translated or voiced. Each show (on its page) and each episode (when you add it) can choose differently.</p>
      <div class="row" style="margin-top:10px"><select id="revMode"><option value="off">Off: never stop</option><option value="unsure">When unsure: stop only if something needs a look</option>
        <option value="always">Always: stop for every episode</option></select></div></div>
    <div class="panel"><h3>Bazarr</h3>
      <p class="small muted">Ai Dub can ask Bazarr for English subtitles it doesn't find next to the episode.</p>
      <div class="field-grid" style="margin-top:12px">
        <span class="lab">Use Bazarr</span><span><label><input type="checkbox" id="bzon" ${bz.enabled ? "checked" : ""}> Ask Bazarr for missing subtitles</label></span>
        <span class="lab">Address</span><span><input type="text" id="bzurl" value="${esc(bz.url || "")}" placeholder="http://bazarr:6767" style="width:min(360px,100%)"></span>
        <span class="lab">Folder map</span><span><textarea id="bzmap" rows="3">${esc((bz.path_map || []).map((m) => `${m.aidub} = ${m.bazarr}`).join("\n"))}</textarea></span>
        <span class="help">How Bazarr sees the same folders, one per line: <code>ai-dub path = bazarr path</code>.</span>
        <span></span><span><button class="primary" id="bzsave">Save Bazarr settings</button></span>
      </div></div>
    <div class="panel"><h3>Keys</h3><p class="small muted">Keys can be entered or replaced, but are never shown back. A restart applies them.</p>
      <div class="field-grid" style="margin-top:12px">
      ${Object.entries(s.secrets).map(([k, set]) => `<span class="lab"><code>${esc(k)}</code></span>
        <span class="row"><span class="tag ${set ? "ok" : ""}">${set ? "Set" : "Not set"}</span>
          <input type="password" data-secret="${esc(k)}" placeholder="${set ? "Replace with…" : "Enter value…"}" autocomplete="off" style="width:min(280px,100%)">
          <button class="sm" data-save="${esc(k)}">Save</button>${set ? `<button class="sm ghost" data-clear="${esc(k)}">Clear</button>` : ""}</span>`).join("")}
      </div></div>
    <div class="panel flush"><div class="browser-bar"><h3 class="grow" style="margin:0">Models per step</h3>
      <a class="button sm" href="#models">Models page</a></div>
      <div class="tablewrap"><table><thead><tr><th>Step</th><th>Worker</th><th>Model</th><th></th></tr></thead><tbody>
      ${Object.entries(s.effective.stages || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="muted">${esc(v.worker || "")}</td><td class="small">${esc(v.model || "")}</td>
        <td class="actions">${status && status.workers[v.worker] ? '<span class="tag ok">Installed</span>' : '<span class="tag warn">Not installed</span>'}</td></tr>`).join("")}</tbody></table></div>
      <div class="panel-foot small muted">Choose different models in config.yaml in the app data folder.</div></div>
    <div class="panel"><details><summary>Effective configuration</summary><pre>${esc(JSON.stringify(s.effective, null, 1))}</pre></details></div>`;
  $("#revMode").value = (s.effective.review || {}).mode || "unsure";
  $("#revMode").onchange = async () => {
    const o = s.overrides || {};
    o.review = { ...(o.review || {}), mode: $("#revMode").value };
    try { toast((await api("/settings", { body: { overrides: o } })).note || "Saved"); } catch (e) { fail(e); }
  };
  $("#bzsave").onclick = async () => {
    const o = s.overrides || {};
    o.subtitles = o.subtitles || {}; o.subtitles.providers = o.subtitles.providers || {};
    const map = $("#bzmap").value.split("\n").map((l) => l.split("=").map((x) => x.trim())).filter((p) => p.length === 2 && p[0] && p[1]).map(([a, b]) => ({ aidub: a, bazarr: b }));
    o.subtitles.providers.bazarr = { ...(o.subtitles.providers.bazarr || {}), enabled: $("#bzon").checked, url: $("#bzurl").value.trim(), path_map: map, api_key: "${BAZARR_API_KEY}" };
    try { toast((await api("/settings", { body: { overrides: o } })).note || "Saved"); } catch (e) { fail(e); }
  };
  main.querySelectorAll("[data-save]").forEach((b) => (b.onclick = async () => {
    const v = $(`[data-secret="${b.dataset.save}"]`).value;
    if (!v) return toast("Enter a value first");
    try { await api("/secrets", { body: { name: b.dataset.save, value: v } }); toast("Saved; restart Ai Dub to apply"); } catch (e) { fail(e); }
    pageSettings();
  }));
  main.querySelectorAll("[data-clear]").forEach((b) => (b.onclick = async () => {
    if (!confirm("Clear this key?")) return;
    await api("/secrets", { body: { name: b.dataset.clear, value: "" } }).catch(fail); pageSettings();
  }));
}

// Routing ------------------------------------------------------------------------------------
async function route() {
  clearInterval(timer); clearTimeout(timer);
  const [page, arg, extra] = (location.hash.slice(1) || "queue").split("/");
  main.classList.remove("wide");
  document.querySelectorAll(".rv-pop").forEach((p) => p.remove());
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === (["review", "verify"].includes(page) ? "queue" : page)));
  main.innerHTML = `<div class="loading">Loading…</div>`;
  window.scrollTo(0, 0);
  try {
    await ({ queue: pageQueue, history: pageHistory, profiles: () => pageProfiles(arg && decodeURIComponent(arg)),
      originals: pageOriginals, models: pageModels, settings: pageSettings, review: () => pageReview(arg, extra),
      verify: () => pageVerify(arg) }[page] || pageQueue)();
  } catch (e) { main.innerHTML = `<div class="callout bad">Couldn't load this page: ${esc(e.message)}</div>`; }
}

window.addEventListener("hashchange", route);
refreshStatus().then(route);
setInterval(refreshStatus, 15000);
