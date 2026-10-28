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
    td.fb textarea[hidden] { display: none; }
    td.fb .st { font-size: 12px; opacity: .7; margin-left: 4px; }
    tr.fb-ok td:first-child { box-shadow: inset 4px 0 #3fa36b; }
    tr.fb-poor td:first-child { box-shadow: inset 4px 0 #d9534f; }
    #fb-bar { position: sticky; top: 0; z-index: 3; padding: 8px 12px; margin: 0 -16px 12px; backdrop-filter: blur(6px);
      background: rgba(20,23,28,.85); color: #e6e9ef; border-bottom: 1px solid rgba(127,127,127,.3); }
    #fb-bar a { color: #7d9bff; margin-left: 12px; }`;
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
      (next ? `<a href="#fb-${next.item}">next unjudged</a>` : " &nbsp;·&nbsp; all done, thank you");
  }

  async function save(item, verdict, comment, st) {
    st.textContent = "saving…";
    try {
      const r = await fetch("/api/listen/feedback", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ page, item, verdict, comment }) });
      if (!r.ok) throw new Error(r.status);
      saved[item] = { verdict, comment: comment.trim() };
      st.textContent = "saved";
    } catch (e) {
      st.textContent = "not saved, try again";
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
        `<label><input type="radio" name="${name}" value="poor"> Poor</label><span class="st"></span>` +
        `<textarea placeholder="What's wrong? (voice, emotion, timing, wording, should stay original...)" hidden></textarea>`;
      const st = td.querySelector(".st"), box = td.querySelector("textarea");
      const radios = [...td.querySelectorAll("input")];
      const show = v => {
        box.hidden = v !== "poor";
        tr.classList.toggle("fb-ok", v === "acceptable");
        tr.classList.toggle("fb-poor", v === "poor");
      };
      if (prev.verdict) radios.find(r => r.value === prev.verdict).checked = true;
      box.value = prev.comment || "";
      show(prev.verdict);
      const verdict = () => (radios.find(r => r.checked) || {}).value || "";
      for (const r of radios) r.addEventListener("change", () => {
        show(r.value);
        if (r.value === "poor") box.focus();
        save(item, r.value, box.value, st);
      });
      let timer;
      box.addEventListener("input", () => {
        clearTimeout(timer);
        st.textContent = "…";
        timer = setTimeout(() => save(item, verdict(), box.value, st), 700);
      });
      box.addEventListener("blur", () => { clearTimeout(timer); if ((saved[item] || {}).comment !== box.value.trim()) save(item, verdict(), box.value, st); });
      tr.appendChild(td);
    }
  }

  fetch("/api/listen/feedback?page=" + encodeURIComponent(page))
    .then(r => r.ok ? r.json() : {})
    .catch(() => ({}))
    .then(data => { saved = data || {}; build(saved); tally(); });
})();
