"use strict";
// Ai Dub web UI: plain JavaScript, no build step.

const $ = (sel, el = document) => el.querySelector(sel);
const main = $("#main");
let timer = null;
let status = null;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const gb = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + " GB" : n >= 1e6 ? (n / 1e6).toFixed(1) + " MB" : Math.round((n || 0) / 1e3) + " kB");
const when = (t) => (t ? new Date(t * 1000).toLocaleString() : "");
const counts = (o) => Object.entries(o || {}).filter(([, v]) => v).map(([k, v]) => `${v} ${k}`).join(", ") || "none";
const dur = (s) => { s = Math.round(s || 0); const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60; return h ? `${h}h ${m}m` : `${m}m ${s % 60}s`; };

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

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 4000);
}

async function refreshStatus() {
  try {
    status = await api("/status");
    const g = status.gpu;
    $("#gpu").textContent = g.present ? `${g.name}: ${Math.round(g.used_mb / 1024)} of ${Math.round(g.total_mb / 1024)} GB in use` : "No GPU visible";
  } catch { $("#gpu").textContent = "server unreachable"; }
}

// Queue ------------------------------------------------------------------------------------
let browsePath = "";
let selected = new Set();

async function pageQueue() {
  main.innerHTML = `
    <h2>Add files</h2>
    <div class="panel">
      <div class="row small muted" id="crumbs"></div>
      <div class="browser" id="browser"></div>
      <div class="row" style="margin-top:10px">
        <span class="grow small muted" id="selcount">Nothing selected</span>
        <button class="primary" id="add" disabled>Add to queue</button>
      </div>
      <div id="addform" hidden></div>
    </div>
    <h2>Queue</h2>
    <div id="jobs"></div>`;
  $("#add").onclick = () => showAddForm();
  await loadBrowser(browsePath);
  await loadQueue();
  timer = setInterval(loadQueue, 2500);
}

async function loadBrowser(path) {
  const data = await api("/browse?path=" + encodeURIComponent(path));
  browsePath = data.path;
  const crumbs = $("#crumbs");
  crumbs.innerHTML = data.path ? `<button id="up">Up</button> <span>${esc(data.path)}</span>` : "Media folders";
  if (data.path) $("#up").onclick = () => loadBrowser(data.parent || "");
  const box = $("#browser");
  box.innerHTML = data.entries.length ? "" : `<div class="entry muted">Empty</div>`;
  for (const e of data.entries) {
    const div = document.createElement("div");
    div.className = "entry";
    if (e.dir) {
      div.innerHTML = `<span>📁</span><span class="name">${esc(e.name)}</span>`;
      div.onclick = () => loadBrowser(e.path);
    } else {
      const tags = [e.queued ? `<span class="tag">queued</span>` : "", e.set_aside ? `<span class="tag ok">dubbed</span>` : ""].join(" ");
      div.innerHTML = `<input type="checkbox" ${selected.has(e.path) ? "checked" : ""} ${e.queued ? "disabled" : ""}>
        <span class="name">${esc(e.name)}</span>${tags}<span class="small muted">${gb(e.size)}</span>`;
      const cb = $("input", div);
      div.onclick = (ev) => { if (ev.target !== cb && !cb.disabled) cb.checked = !cb.checked; toggle(e.path, cb.checked); };
    }
    box.appendChild(div);
  }
}

function toggle(path, on) {
  on ? selected.add(path) : selected.delete(path);
  $("#selcount").textContent = selected.size ? `${selected.size} file(s) selected` : "Nothing selected";
  $("#add").disabled = !selected.size;
}

async function showAddForm() {
  const form = $("#addform");
  const paths = [...selected];
  form.hidden = false;
  form.innerHTML = `<p class="muted small">Checking ${paths.length} file(s)…</p>`;
  const first = await api("/inspect", { body: { path: paths[0] } }).catch((e) => ({ error: e.message }));
  const profiles = await api("/profiles");
  if (first.error) { form.innerHTML = `<p class="tag bad">${esc(first.error)}</p>`; return; }
  const tracks = first.foreign_tracks || [];
  const note = first.english_audio ? `<p class="tag warn">Has English audio: it will be skipped unless you tick Redo.</p>`
    : first.aidub_present ? `<p class="tag warn">Already has an Ai Dub track. Tick Redo to rebuild it.</p>` : "";
  form.innerHTML = `
    <div style="margin-top:12px" class="row">
      <label>Profile <input type="text" id="prof" list="proflist" value="${esc(first.profile_guess)}"></label>
      <datalist id="proflist">${profiles.map((p) => `<option value="${esc(p.slug)}">${esc(p.name)}</option>`).join("")}</datalist>
      ${tracks.length > 1 ? `<label>Audio track <select id="track"><option value="">default</option>${tracks.map((t) =>
        `<option value="${t.index}">#${t.index} ${esc(t.language || "?")} ${esc(t.title)} ${t.channels}ch</option>`).join("")}</select></label>` : ""}
      <label>Language <select id="lang"><option value="">detect</option><option value="ja">Japanese</option><option value="ko">Korean</option>
        <option value="zh">Mandarin</option><option value="yue">Cantonese</option></select></label>
      <label><input type="checkbox" id="redo"> Redo</label>
      <label>Review before dubbing <select id="review"><option value="">Default</option><option value="off">Off</option>
        <option value="unsure">When unsure</option><option value="always">Always</option></select></label>
      <button class="primary" id="confirmadd">Add ${paths.length} file(s)</button>
    </div>
    <p class="small muted">Profile is guessed from the folder name${paths.length > 1 ? " of the first file; leave it as is to let each file use its own show" : ""}. ${dur(first.duration)}, ${esc(first.container)}.</p>
    ${note}`;
  $("#confirmadd").onclick = async () => {
    const prof = $("#prof").value.trim();
    const body = { paths, profile: paths.length > 1 && prof === first.profile_guess ? null : prof || null,
      audio_stream: $("#track") && $("#track").value !== "" ? Number($("#track").value) : null,
      language: $("#lang").value || null, redo: $("#redo").checked, review: $("#review").value || null };
    const res = await api("/jobs", { body });
    toast(`Added ${res.added.length}` + (res.errors.length ? `; ${res.errors.length} not added: ${res.errors[0].error}` : ""));
    selected.clear(); toggle("", false); form.hidden = true;
    await loadBrowser(browsePath); await loadQueue();
  };
}

async function loadQueue() {
  const jobs = await api("/jobs?view=queue");
  const box = $("#jobs");
  if (!box) return;
  if (!jobs.length) { box.innerHTML = `<div class="panel muted">The queue is empty.</div>`; return; }
  box.innerHTML = jobs.map((j, i) => {
    const pct = Math.round((j.progress || 0) * 100);
    const st = j.status === "waiting" ? `<span class="tag warn">waiting for GPU</span>` : j.status === "running" ? `<span class="tag ok">running</span>`
      : j.status === "review" ? `<span class="tag warn">waiting for review</span>` : `<span class="tag">queued</span>`;
    return `<div class="panel job" data-id="${j.id}">
      <div><div class="path">${esc(j.path.split("/").pop())}</div>
        <div class="small muted">${esc(j.profile)} · ${st} ${esc(j.step_title || "")} — ${esc(j.message)}</div>
        ${j.status !== "queued" && j.status !== "review" ? `<div class="progress"><div style="width:${pct}%"></div></div>` : ""}</div>
      <div class="row">
        ${j.status === "review" ? `<a class="button primary" href="#review/${j.id}">Review</a>` : j.status === "running" ? `<a class="button" href="#review/${j.id}" title="See what Ai Dub has decided so far">Look</a>` : ""}
        ${j.status === "queued" ? `<button data-act="up" ${i === 0 ? "disabled" : ""}>↑</button><button data-act="down">↓</button>` : ""}
        <button data-act="log">Log</button>
        <button class="danger" data-act="cancel">${j.status === "queued" || j.status === "review" ? "Remove" : "Cancel"}</button>
      </div>
      <pre class="log" hidden></pre></div>`;
  }).join("");
  box.querySelectorAll("button[data-act]").forEach((b) => {
    const id = b.closest(".job").dataset.id;
    b.onclick = async () => {
      const act = b.dataset.act;
      if (act === "up" || act === "down") await api(`/jobs/${id}/move`, { body: { direction: act === "up" ? -1 : 1 } });
      if (act === "cancel") { if (!confirm("Cancel this job? Its working files are deleted.")) return; await api(`/jobs/${id}`, { method: "DELETE" }); }
      if (act === "log") { const pre = $("pre.log", b.closest(".job")); pre.hidden = !pre.hidden; pre.textContent = await api(`/jobs/${id}/log`); return; }
      loadQueue();
    };
  });
}

// History ------------------------------------------------------------------------------------
async function pageHistory() {
  const jobs = await api("/jobs?view=history");
  main.innerHTML = `<h2>History</h2>` + (jobs.length ? jobs.map((j) => {
    const r = j.record || {}, qc = r.qc || {};
    const tag = { done: "ok", skipped: "", failed: "bad", cancelled: "" }[j.status] ?? "";
    const fit = counts(qc.fit);
    return `<div class="panel" data-id="${j.id}">
      <div class="row"><span class="grow path">${esc(j.path.split("/").pop())}</span><span class="tag ${tag}">${esc(j.status)}</span></div>
      <div class="small muted">${esc(j.message)} · ${when(j.finished)} · profile ${esc(j.profile)}</div>
      ${qc.subtitle_track && qc.subtitle_track.trusted === false && qc.subtitle_track.why !== "no English subtitle"
        ? `<div class="tag warn" style="margin-top:6px">English subtitles not used as the script: ${esc(qc.subtitle_track.why)}. Dubbed from the speech instead.</div>` : ""}
      ${j.status === "failed" ? `<div class="row small" style="margin-top:6px"><span class="muted">Working files kept: ${gb(j.work_bytes || 0)}</span>
        <button data-act="retry">Retry from the failed step</button><button data-act="remove" class="danger">Remove</button></div>` :
        `<div class="row" style="margin-top:6px"><button data-act="remove">Remove from history</button></div>`}
      ${r.version ? `<details><summary class="small">QC summary and version stamp</summary>
        <table class="small"><tr><th>Text from</th><td>${esc(r.text_source || "")}</td></tr>
        <tr><th>Lines</th><td>${esc(counts(qc.lines))}</td></tr>
        <tr><th>Timing fit</th><td>${esc(fit)}</td></tr>
        <tr><th>Fell back to original</th><td>${((qc.fallback_rate || 0) * 100).toFixed(1)} %</td></tr>
        <tr><th>Minor-speaker voices</th><td>${esc(counts(qc.minor_speaker_substitutions))}</td></tr>
        <tr><th>Subtitle tracks added</th><td>${esc(((r.delivered || {}).subtitle_tracks || []).map((s) => s.title).join(", ") || "none")}</td></tr>
        <tr><th>Ai Dub</th><td>${esc(r.version)}</td></tr>
        <tr><th>Models</th><td>${esc(Object.entries(r.models || {}).map(([k, v]) => `${k}: ${v}`).join("; "))}</td></tr>
        <tr><th>Step times</th><td>${esc(Object.entries(r.timings || {}).map(([k, v]) => `${k} ${dur(v)}`).join(", "))}</td></tr></table>
        ${(qc.fallback_lines || []).length ? `<details><summary class="small">Lines kept in the original language</summary><pre>${esc(qc.fallback_lines.map((l) => `${dur(l.start)}  ${l.reason}`).join("\n"))}</pre></details>` : ""}
        ${r.error ? `<pre>${esc(r.error)}</pre>` : ""}</details>` : ""}
    </div>`;
  }).join("") : `<div class="panel muted">Nothing yet.</div>`);
  main.querySelectorAll("button[data-act]").forEach((b) => {
    const id = b.closest(".panel").dataset.id;
    b.onclick = async () => {
      if (b.dataset.act === "retry") await api(`/jobs/${id}/retry`, { body: {} });
      if (b.dataset.act === "remove") { if (!confirm("Remove this job and its working files?")) return; await api(`/jobs/${id}`, { method: "DELETE" }); }
      pageHistory();
    };
  });
}

// Profiles ------------------------------------------------------------------------------------
async function pageProfiles(slug) {
  if (slug) return pageProfile(slug);
  const list = await api("/profiles");
  main.innerHTML = `<h2>Profiles</h2>` + (list.length ? `<div class="panel tablewrap"><table>
    <tr><th>Show</th><th>Characters</th><th>Glossary</th><th>Episodes</th><th>Size</th></tr>
    ${list.map((p) => `<tr><td><a href="#profiles/${encodeURIComponent(p.slug)}">${esc(p.name)}</a></td><td>${p.characters}</td>
      <td>${p.glossary}</td><td>${p.episodes}</td><td>${gb(p.size)}</td></tr>`).join("")}</table></div>` : `<div class="panel muted">No profiles yet. One is made for each show you dub.</div>`);
}

async function pageProfile(slug) {
  const p = await api("/profiles/" + encodeURIComponent(slug));
  const others = (cid) => p.characters.filter((c) => c.id !== cid).map((c) => `<option value="${c.id}">${esc(c.name)}</option>`).join("");
  main.innerHTML = `
    <p><a href="#profiles">← Profiles</a></p>
    <h2>${esc(p.name)} <span class="small muted">${gb(p.size)} · ${p.episodes.length} episode(s)</span></h2>
    <div class="panel row">
      <label>Honorifics <select id="hon"><option value="keep">keep (-san, -kun…)</option><option value="drop">drop</option></select></label>
      <label>Style <select id="style"><option value="natural">natural</option><option value="close">close</option></select></label>
      <span class="grow"></span><button class="danger" id="delprof">Delete profile</button>
      <p class="small muted" style="width:100%;margin:0">Honorifics and style apply to lines translated from the speech. Lines dubbed from
        English subtitles keep the subtitles' own wording.</p>
    </div>
    <h2>Show rules</h2>
    <div class="panel" id="rules"><span class="muted small">Loading…</span></div>
    <h2>Voice bank</h2>
    <div class="chars">${p.characters.map((c) => `
      <div class="panel" data-cid="${c.id}">
        <div class="row"><input type="text" value="${esc(c.name)}" class="cname grow"><button data-act="rename">Rename</button></div>
        <div class="small muted">${esc([c.gender, c.age].filter(Boolean).join(", ") || "gender/age unknown")} · ${c.clean_seconds}s of clean audio
          ${c.voice_of ? ` · speaks with ${esc((p.characters.find((x) => x.id === c.voice_of) || {}).name || c.voice_of)}'s voice` : ""}</div>
        ${(c.refs || []).map((r) => `<div class="ref" data-ref="${esc(r.file)}">
          <audio controls preload="none" src="/api/profiles/${encodeURIComponent(p.slug)}/ref?file=${encodeURIComponent(r.file)}"></audio>
          <span class="small muted">${r.duration}s</span>
          <label class="small"><input type="checkbox" data-act="lock" ${r.locked ? "checked" : ""}> keep</label>
          <label class="small"><input type="checkbox" class="pick"> select</label>
          <button data-act="rmref" class="small">Remove</button></div>`).join("") || `<p class="small muted">No clean clips yet.</p>`}
        <div class="row" style="margin-top:8px">
          ${p.characters.length > 1 ? `<select class="other">${others(c.id)}</select><button data-act="merge">Merge into this</button>` : ""}
          <button data-act="split">Split selected clips off</button>
          <button data-act="delchar" class="danger">Delete</button>
        </div>
      </div>`).join("") || `<div class="panel muted">No characters yet.</div>`}</div>
    <h2>Glossary</h2>
    <div class="panel">
      <div class="row"><input type="text" id="gsrc" placeholder="source term"><input type="text" id="gen" placeholder="English"><button id="gadd">Add / change</button></div>
      <table class="small" style="margin-top:8px"><tr><th>Source</th><th>English</th><th>Seen</th><th></th></tr>
      ${Object.entries(p.glossary).sort((a, b) => (b[1].count || 0) - (a[1].count || 0)).map(([k, v]) =>
        `<tr><td>${esc(k)}</td><td>${esc(v.en)}${v.edited ? ' <span class="tag">edited</span>' : ""}</td><td>${v.count || 0}</td>
         <td><button data-gdel="${esc(k)}">Remove</button></td></tr>`).join("")}</table>
    </div>`;
  showRules(slug);
  $("#hon").value = p.settings.honorifics || "keep";
  $("#style").value = p.settings.style || "natural";
  const save = () => api("/profiles/" + encodeURIComponent(slug), { method: "PATCH", body: { settings: { honorifics: $("#hon").value, style: $("#style").value } } }).then(() => toast("Saved"));
  $("#hon").onchange = save; $("#style").onchange = save;
  $("#delprof").onclick = async () => {
    if (!confirm(`Delete the profile "${p.name}"? Its voices and glossary are removed and the next episode starts fresh.`)) return;
    await api(`/profiles/${encodeURIComponent(slug)}?confirm=true`, { method: "DELETE" });
    location.hash = "#profiles";
  };
  $("#gadd").onclick = async () => {
    if (!$("#gsrc").value.trim()) return;
    await api(`/profiles/${encodeURIComponent(slug)}/glossary`, { method: "PUT", body: { source: $("#gsrc").value, en: $("#gen").value } });
    pageProfile(slug);
  };
  main.querySelectorAll("[data-gdel]").forEach((b) => (b.onclick = async () => {
    await api(`/profiles/${encodeURIComponent(slug)}/glossary?source=${encodeURIComponent(b.dataset.gdel)}`, { method: "DELETE" });
    pageProfile(slug);
  }));
  main.querySelectorAll("[data-cid]").forEach((card) => {
    const cid = card.dataset.cid;
    const act = (body) => api(`/profiles/${encodeURIComponent(slug)}/characters/${cid}`, { body }).then(() => pageProfile(slug)).catch((e) => toast(e.message));
    card.querySelectorAll("[data-act]").forEach((el) => {
      const a = el.dataset.act;
      const ref = el.closest(".ref") ? el.closest(".ref").dataset.ref : null;
      if (a === "lock") el.onchange = () => act({ action: "lock_ref", ref, locked: el.checked });
      else el.onclick = () => {
        if (a === "rename") act({ action: "rename", name: $(".cname", card).value });
        if (a === "rmref" && confirm("Remove this clip?")) act({ action: "remove_ref", ref });
        if (a === "merge" && confirm("Merge the chosen character into this one?")) act({ action: "merge", other: $(".other", card).value });
        if (a === "split") {
          const refs = [...card.querySelectorAll(".ref")].filter((r) => $(".pick", r).checked).map((r) => r.dataset.ref);
          if (!refs.length) return toast("Select the clips that belong to another character first");
          act({ action: "split", refs, name: prompt("Name for the new character", "") || "" });
        }
        if (a === "delchar" && confirm("Delete this character and its clips?")) act({ action: "delete", confirm: true });
      };
    });
  });
}

async function showRules(slug) {
  const box = $("#rules");
  const r = await api(`/profiles/${encodeURIComponent(slug)}/rules`);
  const names = { romaji_lyrics: "Romaji lines are lyrics", italic_quoted_lyrics: "Italics in quotation marks are lyrics",
    tilde_signs: "“~ text ~” is a sign, not spoken", narrator_dubbed: "The narrator's opening is dubbed" };
  box.innerHTML = `<div class="row"><label>Review before dubbing <select id="ruleMode"><option value="">Default (${esc(r.default_review)})</option>
      <option value="off">Off</option><option value="unsure">When unsure</option><option value="always">Always</option></select></label></div>
    <div class="row" style="margin-top:8px">${Object.entries(names).map(([k, t]) => `<label class="small"><input type="checkbox" data-habit="${k}" ${r.habits[k] ? "checked" : ""}> ${t}</label>`).join("")}</div>
    ${r.regions.length ? `<table class="small" style="margin-top:8px"><tr><th>Opening / ending</th><th>Length</th><th>Set on</th><th></th></tr>
      ${r.regions.map((x) => `<tr><td>${esc(x.label || x.kind)}</td><td>${x.seconds} s</td><td>${esc(x.episode)}</td><td><button data-unrule="${esc(x.id)}">Remove</button></td></tr>`).join("")}</table>`
      : `<p class="small muted">No opening or ending marked yet. Mark one on an episode's review screen and tick “Use for every episode”.</p>`}`;
  $("#ruleMode").value = r.review || "";
  const save = (body) => api(`/profiles/${encodeURIComponent(slug)}/rules`, { method: "PUT", body }).then(() => toast("Saved")).catch((e) => toast(e.message));
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
  main.innerHTML = `<h2>Originals pending deletion</h2>
    <div class="panel"><p class="small muted">Each dubbed file's original is kept beside it with the extension .aidub-original.
      Nothing deletes them except you, from here. Restore puts an original back under its own name and sets the dubbed copy aside instead.</p>
      <div class="row"><span class="grow">${data.items.length} file(s), ${gb(data.total)} in total</span>
        <button id="scan">Rescan folders</button>
        <button class="danger" id="delsel" disabled>Delete selected</button>
        <button class="danger" id="delall" ${data.items.length ? "" : "disabled"}>Delete all</button></div></div>
    ${data.items.length ? `<div class="panel tablewrap"><table><tr><th></th><th>File</th><th>Kind</th><th>Size</th><th>Set aside</th><th></th></tr>
      ${data.items.map((o) => `<tr data-id="${o.id}"><td><input type="checkbox" class="pick"></td><td style="overflow-wrap:anywhere">${esc(o.path.split("/").pop())}<br><span class="small muted">${esc(o.path.split("/").slice(0, -1).join("/"))}</span></td>
        <td>${esc(o.kind)}</td><td>${gb(o.size)}</td><td class="small">${when(o.created)}${o.job_id ? ` · job ${o.job_id}` : ""}</td>
        <td><button data-act="restore">Restore</button></td></tr>`).join("")}</table></div>` : ""}`;
  const picks = () => [...main.querySelectorAll("tr[data-id]")].filter((r) => $(".pick", r).checked).map((r) => Number(r.dataset.id));
  main.querySelectorAll(".pick").forEach((cb) => (cb.onchange = () => ($("#delsel").disabled = !picks().length)));
  const del = async (ids) => {
    const size = data.items.filter((o) => ids.includes(o.id)).reduce((a, o) => a + o.size, 0);
    if (!confirm(`Permanently delete ${ids.length} file(s), ${gb(size)}? This cannot be undone.`)) return;
    const r = await api("/originals/delete", { body: { ids, confirm: true } });
    toast(`Deleted ${r.deleted} file(s), freed ${gb(r.freed)}`); pageOriginals();
  };
  $("#delsel").onclick = () => del(picks());
  $("#delall").onclick = () => del(data.items.map((o) => o.id));
  $("#scan").onclick = async () => { const r = await api("/originals/scan", { body: {} }); toast(`Found ${r.added} more`); pageOriginals(); };
  main.querySelectorAll("[data-act=restore]").forEach((b) => (b.onclick = async () => {
    if (!confirm("Restore this file under its own name? The current file is set aside in its place.")) return;
    try { toast((await api(`/originals/${b.closest("tr").dataset.id}/restore`, { body: {} })).message); } catch (e) { toast(e.message); }
    pageOriginals();
  }));
}

// Models ------------------------------------------------------------------------------------
async function pageModels() {
  const list = await api("/models");
  const busy = list.some((m) => m.job === "queued" || m.job === "installing");
  const missing = list.filter((m) => !(m.installed && m.weights)).length;
  main.innerHTML = `<h2>Models</h2>
    <div class="panel"><p class="small muted">Nothing is bundled with Ai Dub. Each model is downloaded from its official
      source into the models folder and runs in its own environment. Downloads total tens of GB; each model keeps its own licence.</p>
      <div class="row"><span class="grow">${missing ? `${missing} of ${list.length} not installed yet` : "All models installed"}</span>
        <button class="primary" id="installall" ${missing && !busy ? "" : "disabled"}>Install all missing</button>
        <button id="voices">Make generic voices</button></div></div>
    <div class="panel tablewrap"><table>
      <tr><th>Step</th><th>Model</th><th>Licence</th><th>Download</th><th>Status</th><th></th></tr>
      ${list.map((m) => {
        const st = m.job === "installing" ? `<span class="tag warn">installing…</span>` : m.job === "queued" ? `<span class="tag">queued</span>`
          : m.job === "failed" ? `<span class="tag bad">failed</span>` : m.installed && m.weights ? `<span class="tag ok">ready</span>`
          : m.installed ? `<span class="tag warn">weights missing</span>` : `<span class="tag">not installed</span>`;
        return `<tr data-name="${esc(m.name)}"><td>${esc(m.title)}</td><td class="small">${esc(m.models)}<br><span class="muted">${esc(m.model)}</span></td>
          <td class="small">${esc(m.licence)}</td><td class="small">${esc(m.download)}</td><td>${st}</td>
          <td class="row"><button data-act="install" ${m.job === "installing" || m.job === "queued" ? "disabled" : ""}>${m.installed && m.weights ? "Reinstall" : "Install"}</button>
          <button data-act="log">Log</button></td></tr>
          <tr hidden class="logrow"><td colspan="6"><pre></pre></td></tr>`;
      }).join("")}</table></div>`;
  $("#installall").onclick = async () => { await api("/models/install", { body: { names: [] } }); pageModels(); };
  $("#voices").onclick = async () => {
    if (!confirm("Design the small library of generic voices (needs the voice-design model installed)?")) return;
    await api("/voices/design", { body: { force: false } }); toast("Started; see the voice-design log");
  };
  main.querySelectorAll("tr[data-name] button").forEach((b) => (b.onclick = async () => {
    const tr = b.closest("tr"), name = tr.dataset.name;
    if (b.dataset.act === "install") { await api("/models/install", { body: { names: [name] } }); pageModels(); return; }
    const row = tr.nextElementSibling; row.hidden = !row.hidden;
    $("pre", row).textContent = (await api(`/models/${name}/log`)) || "No log yet.";
  }));
  if (busy) timer = setTimeout(() => location.hash === "#models" && pageModels(), 4000);
}

// Settings ------------------------------------------------------------------------------------
async function pageSettings() {
  const s = await api("/settings");
  const bz = ((s.effective.subtitles || {}).providers || {}).bazarr || {};
  main.innerHTML = `<h2>Settings</h2>
    <div class="panel"><h2>Bazarr</h2>
      <div class="row"><label><input type="checkbox" id="bzon" ${bz.enabled ? "checked" : ""}> Use Bazarr</label>
        <label>URL <input type="text" id="bzurl" value="${esc(bz.url || "")}" placeholder="http://bazarr:6767"></label></div>
      <p class="small muted">Path map: how Bazarr sees the same folders (one per line, "aidub-path = bazarr-path").</p>
      <textarea id="bzmap" rows="3" style="width:100%">${esc((bz.path_map || []).map((m) => `${m.aidub} = ${m.bazarr}`).join("\n"))}</textarea>
      <div class="row"><button id="bzsave">Save Bazarr settings</button></div></div>
    <div class="panel"><h2>Keys</h2><p class="small muted">Keys can be entered or replaced, but are never shown back.</p>
      ${Object.entries(s.secrets).map(([k, set]) => `<div class="row" style="margin:4px 0"><span style="min-width:220px">${esc(k)}</span>
        <span class="tag ${set ? "ok" : ""}">${set ? "set" : "not set"}</span>
        <input type="password" data-secret="${esc(k)}" placeholder="new value" autocomplete="off"><button data-save="${esc(k)}">Save</button>
        ${set ? `<button data-clear="${esc(k)}">Clear</button>` : ""}</div>`).join("")}</div>
    <div class="panel"><h2>Review before dubbing</h2>
      <p class="small muted">A job can stop after Ai Dub has worked out the songs and speakers, so you can check them on the review screen before it translates and voices anything. Each show and each job can choose differently.</p>
      <div class="row"><select id="revMode"><option value="off">Off: never stop</option><option value="unsure">When unsure: stop only if something needs a look</option>
        <option value="always">Always: stop for every episode</option></select><button id="revSave">Save</button></div></div>
    <div class="panel tablewrap"><h2>Models per step</h2><table class="small">
      ${Object.entries(s.effective.stages || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v.worker || "")}</td><td>${esc(v.model || "")}</td>
        <td>${status && status.workers[v.worker] ? '<span class="tag ok">installed</span>' : '<span class="tag warn">not installed</span>'}</td></tr>`).join("")}</table>
      <p class="small muted">Install models on the Models page. Choose different ones in config.yaml in the app data folder.</p></div>
    <div class="panel"><details><summary>Effective configuration</summary><pre>${esc(JSON.stringify(s.effective, null, 1))}</pre></details></div>`;
  $("#revMode").value = (s.effective.review || {}).mode || "unsure";
  $("#revSave").onclick = async () => {
    const o = s.overrides || {};
    o.review = { ...(o.review || {}), mode: $("#revMode").value };
    const r = await api("/settings", { body: { overrides: o } });
    toast(r.note);
  };
  $("#bzsave").onclick = async () => {
    const o = s.overrides || {};
    o.subtitles = o.subtitles || {}; o.subtitles.providers = o.subtitles.providers || {};
    const map = $("#bzmap").value.split("\n").map((l) => l.split("=").map((x) => x.trim())).filter((p) => p.length === 2 && p[0] && p[1]).map(([a, b]) => ({ aidub: a, bazarr: b }));
    o.subtitles.providers.bazarr = { ...(o.subtitles.providers.bazarr || {}), enabled: $("#bzon").checked, url: $("#bzurl").value.trim(), path_map: map, api_key: "${BAZARR_API_KEY}" };
    const r = await api("/settings", { body: { overrides: o } });
    toast(r.note);
  };
  main.querySelectorAll("[data-save]").forEach((b) => (b.onclick = async () => {
    const v = $(`[data-secret="${b.dataset.save}"]`).value;
    if (!v) return toast("Enter a value first");
    await api("/secrets", { body: { name: b.dataset.save, value: v } }); toast("Saved; restart to apply"); pageSettings();
  }));
  main.querySelectorAll("[data-clear]").forEach((b) => (b.onclick = async () => {
    if (!confirm("Clear this key?")) return;
    await api("/secrets", { body: { name: b.dataset.clear, value: "" } }); pageSettings();
  }));
}

// Routing ------------------------------------------------------------------------------------
async function route() {
  clearInterval(timer); clearTimeout(timer);
  const [page, arg] = (location.hash.slice(1) || "queue").split("/");
  main.classList.remove("wide");
  document.querySelectorAll(".rv-pop").forEach((p) => p.remove());
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === (page === "review" ? "queue" : page)));
  main.innerHTML = `<p class="muted">Loading…</p>`;
  try {
    await ({ queue: pageQueue, history: pageHistory, profiles: () => pageProfiles(arg && decodeURIComponent(arg)),
      originals: pageOriginals, models: pageModels, settings: pageSettings, review: () => pageReview(arg) }[page] || pageQueue)();
  } catch (e) { main.innerHTML = `<div class="panel tag bad">${esc(e.message)}</div>`; }
}

window.addEventListener("hashchange", route);
refreshStatus().then(route);
setInterval(refreshStatus, 15000);
