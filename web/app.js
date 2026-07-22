"use strict";

const state = { mode: "live", scope: "auto", document: null, profiles: [], answer: "", lastQuestion: "", history: [] };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const elements = {
  baseUrl: $("#base-url"), apiKey: $("#api-key"), model: $("#model-name"), timeout: $("#timeout"),
  profileName: $("#profile-name"), profileSelect: $("#profile-select"), connectionStatus: $("#connection-status"),
  documentStatus: $("#document-status"), fileInput: $("#document-file"), dropZone: $("#drop-zone"),
  question: $("#question"), maxWorkers: $("#max-workers"), reduceFanIn: $("#reduce-fan-in"),
  resultPanel: $("#result-panel"), resultEmpty: $(".result-empty"), resultContent: $(".result-content"),
  answerContent: $("#answer-content"), citations: $("#citations"), trace: $("#trace"),
  citationCount: $("#citation-count"), modeHint: $("#mode-hint"),
};

function apiConfig() {
  return { base_url: elements.baseUrl.value.trim(), api_key: elements.apiKey.value.trim(), model: elements.model.value.trim(), timeout_seconds: Number(elements.timeout.value || 90) };
}

function validateApiConfig() {
  const config = apiConfig();
  if (!config.base_url || !config.api_key || !config.model) throw new Error("请先填写 Base URL、API Key 和模型名称。");
  if (!/^https?:\/\//i.test(config.base_url)) throw new Error("Base URL 必须以 http:// 或 https:// 开头。");
  return config;
}

async function request(path, payload = {}) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  let data;
  try { data = await response.json(); } catch { throw new Error(`服务器返回非JSON响应（HTTP ${response.status}）`); }
  if (!response.ok || !data.ok) throw new Error(data.error || `请求失败（HTTP ${response.status}）`);
  return data.result;
}

function setLoading(button, loading) {
  button.disabled = loading;
  button.classList.toggle("loading", loading);
  button.setAttribute("aria-busy", String(loading));
}

function setStatus(element, kind, label) {
  element.className = `status-pill ${kind}`;
  element.replaceChildren(document.createElement("i"), document.createTextNode(label));
}

function toast(message, kind = "default") {
  const item = document.createElement("div");
  item.className = `toast ${kind === "error" ? "error" : ""}`;
  item.textContent = message;
  $("#toast-region").append(item);
  window.setTimeout(() => item.remove(), 3600);
}

const number = (value) => new Intl.NumberFormat("zh-CN").format(value || 0);
function bytes(value) {
  if (!Number.isFinite(value)) return "—";
  if (value >= 1024 * 1024) return `${(value / 1024 / 1024).toFixed(2)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${value} B`;
}

function renderDocument(info, announce = true) {
  state.document = info;
  $("#metric-bytes").textContent = bytes(info.bytes);
  $("#metric-tokens").textContent = number(info.estimated_tokens);
  $("#metric-chunks").textContent = `${number(info.chunks)} / ${number(info.sections)}`;
  setStatus(elements.documentStatus, "success", `${String(info.source_format || "text").toUpperCase()} 已索引`);
  const verdict = $("#limit-verdict");
  verdict.className = `limit-verdict ${info.exceeds_64k_tokens ? "success" : "neutral"}`;
  verdict.lastChild.textContent = info.exceeds_64k_tokens
    ? `资料约 ${number(info.estimated_tokens)} Token，将由多个隔离 Agent 分担`
    : "资料未超过64K，也可验证多 Agent 执行链路";
  $("#empty-title").textContent = "资料已就绪，可以开始提问";
  $("#empty-copy").textContent = `${info.name} 已建立 ${number(info.chunks)} 个证据块；自动模式会优先使用文档 Agent。`;
  updateComposerHint();
  if (announce) toast(`已解析 ${info.name}，建立 ${info.chunks} 个可检索证据块。`);
}

async function restoreCurrentDocument() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    if (response.ok && status.document) renderDocument(status.document, false);
  } catch {
    // The upload flow remains available when no previous server-side document exists.
  }
}

function loadProfiles() {
  try { state.profiles = JSON.parse(sessionStorage.getItem("context-atlas-profiles") || "[]"); } catch { state.profiles = []; }
  const activeId = sessionStorage.getItem("context-atlas-active-profile") || "";
  const active = state.profiles.find((profile) => profile.id === activeId) || null;
  renderProfileOptions(active?.id || "");
  if (active) { applyProfile(active); setMode("live"); }
}

function renderProfileOptions(selectedId = "") {
  elements.profileSelect.replaceChildren(new Option("临时配置", ""));
  state.profiles.forEach((profile) => elements.profileSelect.append(new Option(profile.name, profile.id)));
  elements.profileSelect.value = selectedId;
  $("#delete-profile").disabled = !selectedId;
}

function applyProfile(profile) {
  elements.baseUrl.value = profile?.base_url || "";
  elements.apiKey.value = profile?.api_key || "";
  elements.model.value = profile?.model || "";
  elements.timeout.value = profile?.timeout_seconds || 90;
  elements.profileName.value = profile?.name || "";
  setStatus(elements.connectionStatus, "neutral", "未测试");
}

function saveProfile() {
  try {
    const config = validateApiConfig();
    const name = elements.profileName.value.trim();
    if (!name) throw new Error("请填写配置档案名称。");
    const id = elements.profileSelect.value || `profile_${Date.now()}`;
    state.profiles = [...state.profiles.filter((item) => item.id !== id), { id, name, ...config }];
    sessionStorage.setItem("context-atlas-profiles", JSON.stringify(state.profiles));
    sessionStorage.setItem("context-atlas-active-profile", id);
    renderProfileOptions(id);
    toast(`已保存“${name}”到当前浏览器会话。`);
  } catch (error) { toast(error.message, "error"); }
}

function deleteProfile() {
  const id = elements.profileSelect.value;
  state.profiles = state.profiles.filter((item) => item.id !== id);
  sessionStorage.setItem("context-atlas-profiles", JSON.stringify(state.profiles));
  sessionStorage.removeItem("context-atlas-active-profile");
  renderProfileOptions(); applyProfile(null); toast("已删除当前 API 配置。");
}

async function testConnection() {
  const button = $("#test-connection");
  try {
    setLoading(button, true); setStatus(elements.connectionStatus, "loading", "测试中");
    const result = await request("/api/test-connection", { api: validateApiConfig() });
    const verifiedProfile = {
      profile_token: result.profile_token,
      base_url: result.base_url,
      model: result.model,
      timeout_seconds: result.timeout_seconds,
      verified_at: result.verified_at,
      expires_at: result.expires_at,
      name: elements.profileName.value.trim() || elements.profileSelect.selectedOptions[0]?.textContent || "已验证 API",
    };
    localStorage.setItem("context-atlas-verified-api", JSON.stringify(verifiedProfile));
    setStatus(elements.connectionStatus, "success", "连接正常");
    setMode("live");
    toast(`模型返回：${result.response}；测试中心将复用此 API。`);
  } catch (error) { setStatus(elements.connectionStatus, "error", "连接失败"); toast(error.message, "error"); }
  finally { setLoading(button, false); }
}

async function uploadDocument(file) {
  if (!file) return;
  if (file.size > 20 * 1024 * 1024) return toast("文件超过20 MB限制。", "error");
  setStatus(elements.documentStatus, "loading", "正在解析");
  try {
    const data = new Uint8Array(await file.arrayBuffer());
    let binary = "";
    for (let offset = 0; offset < data.length; offset += 0x8000) binary += String.fromCharCode(...data.subarray(offset, offset + 0x8000));
    renderDocument(await request("/api/documents", { name: file.name, data_base64: window.btoa(binary) }));
  } catch (error) { setStatus(elements.documentStatus, "error", "导入失败"); toast(error.message, "error"); }
  finally { elements.fileInput.value = ""; }
}

function setMode(mode) {
  state.mode = mode;
  $$('[data-mode]').forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  updateComposerHint();
}

function setScope(scope) {
  state.scope = scope;
  $$('[data-scope]').forEach((button) => {
    const active = button.dataset.scope === scope;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  const placeholders = {
    auto: "直接提问，或上传文档后针对资料提问…",
    general: "像普通智能体一样提问、写作、解释或编程…",
    document: "针对已上传文档提问，可直接指定页码…",
  };
  elements.question.placeholder = placeholders[scope] || placeholders.auto;
  updateComposerHint();
}

function updateComposerHint() {
  const webEnabled = Boolean($("#web-search-toggle")?.checked);
  const modeText = "智能回答";
  let routeText;
  if (state.scope === "general") routeText = "直接使用 LLM 通用能力";
  else if (state.scope === "document") routeText = state.document ? "仅基于当前文档调度 Agent" : "需要先上传文档";
  else routeText = state.document ? "自动优先文档问答，普通聊天直连 LLM" : "自动使用通用问答";
  const webText = webEnabled ? " · 联网检索已开启（最多 5 个来源）" : "";
  elements.modeHint.textContent = `${modeText} · ${routeText}${webText}`;
  const webState = $("#web-search-state");
  if (webState) webState.textContent = webEnabled ? "已开启 · 5个来源" : "关闭";
}

function renderResult(result) {
  state.answer = result.answer || "";
  elements.resultEmpty.hidden = true; elements.resultContent.hidden = false;
  $("#result-question").textContent = elements.question.value.trim();
  elements.answerContent.textContent = state.answer;
  const documentRead = result.document_read;
  const generalChat = result.execution_mode === "general_chat";
  const offlineDemo = result.execution_mode === "offline_demo";
  $("#answer-mode-label").textContent = documentRead ? "PDF / 文档原文" : (generalChat ? "LLM 直接回答" : (offlineDemo ? "流程演示结果" : "多 Agent 智能回答"));
  setStatus(
    $("#result-status"),
    result.validation?.approved ? "success" : "warning",
    documentRead ? "原文直接读取" : (generalChat ? "直接回答" : (offlineDemo ? "测试模型" : (result.validation?.approved ? "Validator 通过" : "Validator 拒绝"))),
  );
  const readNav = $("#document-read-nav");
  readNav.hidden = !documentRead;
  if (documentRead) {
    $("#read-progress").textContent = `字符 ${number(documentRead.start_character + 1)}–${number(documentRead.end_character)} / ${number(documentRead.total_characters)}`;
    $("#read-previous").disabled = !documentRead.has_previous;
    $("#read-next").disabled = !documentRead.has_more;
    $("#read-previous").dataset.offset = String(documentRead.previous_offset || 0);
    $("#read-next").dataset.offset = String(documentRead.next_offset || 0);
  }

  const report = result.capacity_report || {};
  $("#context-proof-card").hidden = generalChat;
  $("#result-document-tokens").textContent = `${number(report.document_tokens_estimate)} Token`;
  $("#result-prompt-tokens").textContent = `${number(report.max_single_agent_prompt_tokens)} Token`;
  $("#result-window-usage").textContent = `${Number(report.max_window_utilization_percent || 0).toFixed(1)}%`;
  $("#result-task-count").textContent = `${number(report.task_count)} 个`;
  $("#result-isolation").textContent = report.isolated_specialist_contexts ? "是" : "否";
  $("#result-supervisor-raw").textContent = report.supervisor_received_raw_document ? "是" : "否";
  $("#result-call-count").textContent = `${number(report.model_calls)} 次`;
  const proof = $("#context-proof-state");
  if (documentRead) {
    proof.className = "proof-pass";
    proof.textContent = "直接读取解析文本；未调用模型，不占用 LLM 上下文";
  } else if (report.divide_and_conquer_verified) {
    proof.className = "proof-pass";
    proof.textContent = "资料超过64K；所有Agent调用均未越界";
  } else if (report.all_agent_calls_within_limit) {
    proof.className = "proof-neutral";
    proof.textContent = "所有Agent调用未越界；当前资料未超过64K";
  } else {
    proof.className = "proof-fail"; proof.textContent = "检测到单Agent上下文越界风险";
  }

  elements.citations.replaceChildren();
  const citations = result.citations || [];
  elements.citationCount.textContent = `${citations.length} 条`;
  citations.forEach((citation) => {
    const card = document.createElement("article"); card.className = "citation-card";
    const meta = document.createElement("div"); meta.className = "citation-meta";
    const sourceLabel = citation.url ? document.createElement("a") : document.createElement("span");
    sourceLabel.textContent = `${citation.document} · ${citation.section || citation.chunk_id}`;
    if (citation.url) { sourceLabel.href = citation.url; sourceLabel.target = "_blank"; sourceLabel.rel = "noopener noreferrer"; }
    meta.append(sourceLabel, Object.assign(document.createElement("span"), { textContent: citation.artifact_id }));
    card.append(meta, Object.assign(document.createElement("p"), { textContent: citation.excerpt })); elements.citations.append(card);
  });
  if (!citations.length) elements.citations.append(Object.assign(document.createElement("p"), { textContent: generalChat ? "通用回答未使用文档证据。" : "没有返回可验证的证据引用。" }));

  elements.trace.replaceChildren();
  (result.trace || []).forEach((step, index) => {
    const item = document.createElement("div"); item.className = "trace-step"; item.dataset.step = String(index + 1);
    item.append(Object.assign(document.createElement("strong"), { textContent: `${step.role || step.node} · ${step.status}` }), Object.assign(document.createElement("span"), { textContent: step.detail || "" }));
    elements.trace.append(item);
  });
  elements.resultPanel.scrollTo({ top: 0, behavior: "smooth" });
}

async function runResearch() {
  const button = $("#run-research");
  try {
    const question = elements.question.value.trim();
    if (!question) throw new Error("请输入调研问题。");
    state.lastQuestion = question;
    const payload = { mode: state.mode, answer_scope: state.scope, web_search: $("#web-search-toggle").checked, question, history: state.history, max_workers: Number(elements.maxWorkers.value), reduce_fan_in: Number(elements.reduceFanIn.value) };
    if (state.mode === "live") {
      const config = apiConfig();
      if (config.base_url && config.api_key && config.model) payload.api = validateApiConfig();
    }
    setLoading(button, true);
    const result = await request("/api/ask", payload);
    renderResult(result);
    if (result.execution_mode === "general_chat") {
      state.history.push({ role: "user", content: question }, { role: "assistant", content: result.answer || "" });
      state.history = state.history.slice(-20);
    }
  } catch (error) { toast(error.message, "error"); }
  finally { setLoading(button, false); }
}

async function readDocumentPage(offset) {
  const button = $("#run-research");
  try {
    setLoading(button, true);
    renderResult(await request("/api/ask", {
      mode: state.mode,
      question: state.lastQuestion || "请输出全文。",
      document_read: true,
      read_offset: Number(offset || 0),
    }));
    elements.resultPanel.scrollTo({ top: 0, behavior: "smooth" });
  } catch (error) { toast(error.message, "error"); }
  finally { setLoading(button, false); }
}

elements.profileSelect.addEventListener("change", () => { const profile = state.profiles.find((item) => item.id === elements.profileSelect.value); applyProfile(profile || null); $("#delete-profile").disabled = !profile; if (profile) { sessionStorage.setItem("context-atlas-active-profile", profile.id); setMode("live"); } else sessionStorage.removeItem("context-atlas-active-profile"); });
$("#save-profile").addEventListener("click", saveProfile); $("#delete-profile").addEventListener("click", deleteProfile); $("#test-connection").addEventListener("click", testConnection);
$("#toggle-secret").addEventListener("click", () => { const visible = elements.apiKey.type === "text"; elements.apiKey.type = visible ? "password" : "text"; $("#toggle-secret").textContent = visible ? "显示" : "隐藏"; });
elements.fileInput.addEventListener("change", () => uploadDocument(elements.fileInput.files[0]));
["dragenter", "dragover"].forEach((name) => elements.dropZone.addEventListener(name, (event) => { event.preventDefault(); elements.dropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((name) => elements.dropZone.addEventListener(name, (event) => { event.preventDefault(); elements.dropZone.classList.remove("dragging"); }));
elements.dropZone.addEventListener("drop", (event) => uploadDocument(event.dataTransfer.files[0]));
$$('[data-scope]').forEach((button) => button.addEventListener("click", () => setScope(button.dataset.scope)));
$("#web-search-toggle").addEventListener("change", (event) => { if (event.currentTarget.checked) setMode("live"); else updateComposerHint(); });
$$('[data-question]').forEach((button) => button.addEventListener("click", () => { elements.question.value = button.dataset.question; elements.question.focus(); }));
$("#run-research").addEventListener("click", runResearch);
elements.question.addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") runResearch(); });
$("#copy-answer").addEventListener("click", async () => { try { await navigator.clipboard.writeText(state.answer); toast("答案已复制。"); } catch { toast("浏览器不允许自动复制。", "error"); } });
$("#read-previous").addEventListener("click", (event) => readDocumentPage(event.currentTarget.dataset.offset));
$("#read-next").addEventListener("click", (event) => readDocumentPage(event.currentTarget.dataset.offset));

setMode("live"); setScope("auto"); loadProfiles(); restoreCurrentDocument();
