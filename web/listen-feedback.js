// Listening pages: an Acceptable / Poor choice per row, with a comment box for Poor.
// Each row is the table row holding a "<item>-orig.m4a" clip; answers are saved on the
// server next to the page (feedback.json) as soon as they change.
(() => {
  const page = decodeURIComponent(location.pathname.replace(/^\/listen\//, "").replace(/(index\.html)?$/, "").replace(/\/$/, ""));
  const css = document.createElement("style");
  css.textContent = `
    td.fb { min-width: 260px; }
    td.fb label { display: inline-flex; gap: 4px; align-items: center; margin-right: 12px; cursor: pointer; }
    td.fb textarea { display: block; width: 100%; min-height: 54px; margin-top: 6px; font: inherit; color: inherit;
      background: rgba(127,127,127,.12); border: 1px solid rgba(127,127,127,.4); border-radius: 6px; padding: 4px 6px; }
    td.fb .note { display: flex; gap: 6px; align-items: flex-start; }
    td.fb .note[hidden] { display: none; }
    td.fb .note textarea { flex: 1; }
    td.fb button { margin-top: 6px; font: inherit; padding: 4px 12px; border-radius: 6px; cursor: pointer;
      color: #0d1220; background: #7d9bff; border: 1px solid #7d9bff; }
    td.fb .st { font-size: 13px; margin-top: 4px; min-height: 1.2em; }
    td.fb .st.ok { color: #4cc17f; } td.fb .st.wait, td.fb .st.busy { color: #e3a33b; } td.fb .st.bad { color: #ff6b5e; font-weight: 600; }
    tr.fb-ok td:first-child { box-shadow: inset 4px 0 #3fa36b; }
    tr.fb-poor td:first-child { box-shadow: inset 4px 0 #d9534f; }
    #fb-bar { position: sticky; top: 0; z-index: 3; padding: 8px 12px; margin: 0 -16px 12px; backdrop-filter: blur(6px);
      background: rgba(20,23,28,.85); color: #e6e9ef; border-bottom: 1px solid rgba(127,127,127,.3); }
    #fb-bar a { color: #7d9bff; margin-left: 12px; }
    #fb-bar .good { color: #4cc17f; } #fb-bar .warn { color: #e3a33b; }`;
  document.head.appendChild(css);

  const rows = [];
  for (const tr of document.querySelectorAll("tr")) {
    const a = [...tr.querySelectorAll("audio")].find(x => /-orig\.m4a$/.test(x.getAttribute("src") || ""));
    if (a) rows.push({ tr, item: a.getAttribute("src").replace(/-orig\.m4a$/, "") });
  }
  if (!rows.length) return;

  const bar = document.createElement("div");
  bar.id = "fb-bar";
  document.body.insertBefore(bar, document.body.firstChild);
  let saved = {};

  function tally() {
    const v = Object.values(saved);
    const ok = v.filter(x => x.verdict === "acceptable").length, poor = v.filter(x => x.verdict === "poor").length;
    const next = rows.find(r => !(saved[r.item] || {}).verdict);
    bar.innerHTML = `Judged ${ok + poor} of ${rows.length} &nbsp;·&nbsp; acceptable ${ok} &nbsp;·&nbsp; poor ${poor}` +
      ` &nbsp;·&nbsp; <span class="${pending.size ? "warn" : "good"}">${pending.size ? "not everything saved yet" : "everything saved ✓"}</span>` +
      (next ? `<a href="#fb-${next.item}">next unjudged</a>` : " &nbsp;·&nbsp; all done, thank you");
  }

  // Each row shows plainly whether its answer is on the server: "Saved ✓ 4:22 PM",
  // "Not saved yet" while typing, or a red retry message. Enter or the Save button saves at once.
  const pending = new Set();
  const seq = {};
  const now = () => new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  function mark(st, kind, text) { st.className = "st " + kind; st.textContent = text; }

  async function save(item, verdict, comment, st) {
    const n = seq[item] = (seq[item] || 0) + 1;
    pending.add(item);
    mark(st, "busy", "Saving…");
    try {
      const r = await fetch("/api/listen/feedback", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ page, item, verdict, comment }), keepalive: true });
      if (!r.ok) throw new Error(r.status);
      if (n !== seq[item]) return;  // a newer save for this row is on its way
      saved[item] = { verdict, comment: comment.trim() };
      pending.delete(item);
      mark(st, "ok", `Saved ✓ ${now()}`);
    } catch (e) {
      if (n === seq[item]) mark(st, "bad", "Not saved. Press Save to try again.");
    }
    tally();
  }

  function build(saved) {
    for (const { tr, item } of rows) {
      const prev = saved[item] || {};
      const td = document.createElement("td");
      td.className = "fb";
      td.id = "fb-" + item;
      const name = "fb-" + item;
      td.innerHTML = `<label><input type="radio" name="${name}" value="acceptable"> Acceptable</label>` +
        `<label><input type="radio" name="${name}" value="poor"> Poor</label>` +
        `<div class="note" hidden><textarea placeholder="What's wrong? (voice, emotion, timing, wording, should stay original...)"></textarea>` +
        `<button type="button">Save</button></div><div class="st"></div>`;
      const st = td.querySelector(".st"), note = td.querySelector(".note"), box = td.querySelector("textarea");
      const radios = [...td.querySelectorAll("input")];
      const show = v => {
        note.hidden = v !== "poor";
        tr.classList.toggle("fb-ok", v === "acceptable");
        tr.classList.toggle("fb-poor", v === "poor");
      };
      if (prev.verdict) radios.find(r => r.value === prev.verdict).checked = true;
      box.value = prev.comment || "";
      show(prev.verdict);
      if (prev.verdict || prev.comment) mark(st, "ok", "Saved ✓");
      const verdict = () => (radios.find(r => r.checked) || {}).value || "";
      let timer;
      const saveNow = () => { clearTimeout(timer); save(item, verdict(), box.value, st); };
      for (const r of radios) r.addEventListener("change", () => {
        show(r.value);
        if (r.value === "poor") box.focus();
        saveNow();
      });
      box.addEventListener("input", () => {
        clearTimeout(timer);
        pending.add(item);
        mark(st, "wait", "Not saved yet (saves when you pause or press Enter)");
        tally();
        timer = setTimeout(saveNow, 1000);
      });
      box.addEventListener("keydown", e => {
        if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); saveNow(); }
      });
      box.addEventListener("blur", () => { if (pending.has(item)) saveNow(); });
      td.querySelector("button").addEventListener("click", saveNow);
      tr.appendChild(td);
    }
  }

  window.addEventListener("beforeunload", e => {
    if (pending.size) { e.preventDefault(); e.returnValue = ""; }  // ask before leaving with unsaved notes
  });

  fetch("/api/listen/feedback?page=" + encodeURIComponent(page))
    .then(r => r.ok ? r.json() : {})
    .catch(() => ({}))
    .then(data => { saved = data || {}; build(saved); tally(); });
})();
