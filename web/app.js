"use strict";

const CONVERSATION_STORAGE_KEY = "context-atlas-conversation-v1";
const MAX_STORED_TURNS = 30;
const MAX_STORED_CHARACTERS = 2_500_000;
const state = { mode: "live", scope: "auto", document: null, profiles: [], lastQuestion: "", turns: [] };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const elements = {
  baseUrl: $("#base-url"), apiKey: $("#api-key"), model: $("#model-name"), timeout: $("#timeout"),
  profileName: $("#profile-name"), profileSelect: $("#profile-select"), connectionStatus: $("#connection-status"),
  documentStatus: $("#document-status"), fileInput: $("#document-file"), dropZone: $("#drop-zone"),
  question: $("#question"), defaultAgents: $("#default-agents"), maxWorkers: $("#max-workers"), reduceFanIn: $("#reduce-fan-in"),
  resultPanel: $("#result-panel"), resultEmpty: $(".result-empty"), resultContent: $("#conversation-history"),
  turnCount: $("#turn-count"), clearConversation: $("#clear-conversation"), modeHint: $("#mode-hint"),
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

function node(tag, className = "", text = "") {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== "") item.textContent = text;
  return item;
}

function icon(pathData) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", pathData);
  svg.append(path);
  return svg;
}

function compactResult(result) {
  return {
    answer: String(result.answer || "").slice(0, 80_000),
    execution_mode: result.execution_mode || "",
    stop_reason: result.stop_reason || "",
    validation: result.validation || {},
    capacity_report: result.capacity_report || {},
    document_read: result.document_read || null,
    citations: (result.citations || []).slice(0, 40).map((citation) => ({
      ...citation,
      excerpt: String(citation.excerpt || "").slice(0, 1_200),
    })),
    trace: (result.trace || []).slice(0, 80).map((step) => ({
      node: step.node,
      role: step.role,
      status: step.status,
      detail: String(step.detail || "").slice(0, 1_500),
    })),
  };
}

function persistConversation() {
  let turns = state.turns.slice(-MAX_STORED_TURNS);
  let serialized = JSON.stringify(turns);
  while (turns.length > 1 && serialized.length > MAX_STORED_CHARACTERS) {
    turns.shift();
    serialized = JSON.stringify(turns);
  }
  state.turns = turns;
  try { localStorage.setItem(CONVERSATION_STORAGE_KEY, serialized); }
  catch { toast("对话记录过大，最新内容仅保留在当前页面。", "error"); }
}

function modelHistory() {
  return state.turns.slice(-10).flatMap((turn) => [
    { role: "user", content: String(turn.question || "").slice(0, 2_000) },
    { role: "assistant", content: String(turn.result?.answer || "").slice(0, 4_000) },
  ]);
}

function updateConversationState() {
  const count = state.turns.length;
  elements.turnCount.textContent = `${number(count)} 轮对话`;
  elements.clearConversation.disabled = count === 0;
  elements.resultEmpty.hidden = count > 0;
  elements.resultContent.hidden = count === 0;
}

function restoreConversation() {
  try {
    const parsed = JSON.parse(localStorage.getItem(CONVERSATION_STORAGE_KEY) || "[]");
    state.turns = Array.isArray(parsed) ? parsed.slice(-MAX_STORED_TURNS).filter((turn) => turn?.question && turn?.result) : [];
  } catch { state.turns = []; }
  renderConversation();
}

function renderDocument(info, announce = true) {
  state.document = info;
  $("#metric-bytes").textContent = bytes(info.bytes);
  $("#metric-tokens").textContent = number(info.estimated_tokens);
  $("#metric-chunks").textContent = `${number(info.chunks)} / ${number(info.sections)}`;
  setStatus(elements.documentStatus, "success", `${String(info.source_format || "text").toUpperCase()} 已索引`);
  const verdict = $("#limit-verdict");
  const needsSharding = Boolean(info.exceeds_shard_byte_limit || info.exceeds_64k_tokens);
  verdict.className = `limit-verdict ${needsSharding ? "success" : "neutral"}`;
  verdict.lastChild.textContent = info.exceeds_shard_byte_limit
    ? `索引文本 ${bytes(info.indexed_bytes)} 超过 ${bytes(info.shard_byte_limit)}，回答时将自动分片并分配多个 Agent`
    : (info.exceeds_64k_tokens
      ? `资料约 ${number(info.estimated_tokens)} Token，将由多个隔离 Agent 分担`
      : "资料规模较小，仍会启用默认数量的隔离 Agent");
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

function resultLabels(result) {
  const documentRead = Boolean(result.document_read);
  const generalChat = result.execution_mode === "general_chat";
  const offlineDemo = result.execution_mode === "offline_demo";
  return {
    documentRead,
    generalChat,
    mode: documentRead ? "PDF / 文档原文" : (generalChat ? "LLM 直接回答" : (offlineDemo ? "流程演示结果" : "多 Agent 智能回答")),
    status: documentRead ? "原文直接读取" : (generalChat ? "直接回答" : (offlineDemo ? "测试模型" : (result.validation?.approved ? "Validator 通过" : "Validator 拒绝"))),
  };
}

function createContextProof(result) {
  const report = result.capacity_report || {};
  const card = node("section", "context-proof-card");
  const head = node("div", "proof-card-head");
  const title = node("div");
  title.append(node("span", "section-kicker", "Context isolation"), node("h3", "", "多 Agent 上下文隔离证明"));
  const proof = node("span", "proof-neutral", "等待统计");
  if (result.document_read) {
    proof.className = "proof-pass";
    proof.textContent = "直接读取解析文本；未调用模型，不占用 LLM 上下文";
  } else if (report.source_exceeds_shard_limit && report.multi_agent_sharding_active && report.all_agent_calls_within_limit) {
    proof.className = "proof-pass";
    proof.textContent = `单一来源超过 ${bytes(report.shard_byte_limit)}；${number(report.allocated_agents)} 个 Agent 已完成分片处理`;
  } else if (report.divide_and_conquer_verified) {
    proof.className = "proof-pass";
    proof.textContent = "资料超过 64K；所有 Agent 调用均未越界";
  } else if (report.all_agent_calls_within_limit) {
    proof.className = "proof-neutral";
    proof.textContent = "所有 Agent 调用未越界";
  } else {
    proof.className = "proof-fail";
    proof.textContent = "检测到单 Agent 上下文越界风险";
  }
  head.append(title, proof);

  const flow = node("div", "context-flow");
  const flowItems = [
    ["完整资料", `${number(report.document_tokens_estimate)} Token`],
    ["最大单次 Prompt", `${number(report.max_single_agent_prompt_tokens)} Token`],
    ["64K 模型窗口", `${Number(report.max_window_utilization_percent || 0).toFixed(1)}%`],
  ];
  flowItems.forEach(([label, value], index) => {
    const metric = node("div"); metric.append(node("span", "", label), node("strong", "", value)); flow.append(metric);
    if (index < flowItems.length - 1) {
      const arrow = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      arrow.setAttribute("viewBox", "0 0 48 24");
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path"); path.setAttribute("d", "M2 12h42m-7-7l7 7-7 7"); arrow.append(path); flow.append(arrow);
    }
  });

  const meta = node("div", "proof-meta");
  [
    ["自动 Agent", `${number(report.default_agents)} → ${number(report.allocated_agents)} 个`],
    ["子任务", `${number(report.task_count)} 个`],
    ["专业 Agent 隔离", report.isolated_specialist_contexts ? "是" : "否"],
    ["主 Agent 接收原文", report.supervisor_received_raw_document ? "是" : "否"],
    ["模型调用", `${number(report.model_calls)} 次`],
  ].forEach(([label, value]) => { const item = node("span", "", `${label} `); item.append(node("strong", "", value)); meta.append(item); });
  card.append(head, flow, meta);
  if (report.agent_allocation_reason) card.append(node("p", "allocation-note", `分配依据：${report.agent_allocation_reason}`));
  return card;
}

function createCitationSection(result, generalChat) {
  const section = node("section");
  const citations = result.citations || [];
  const heading = node("div", "result-subheading");
  heading.append(node("h3", "", "检索来源"), node("span", "", `${citations.length} 条`));
  const container = node("div", "citations");
  citations.forEach((citation) => {
    const card = node("article", "citation-card");
    const meta = node("div", "citation-meta");
    const sourceLabel = node(citation.url ? "a" : "span", "", `${citation.document} · ${citation.section || citation.chunk_id}`);
    if (citation.url) { sourceLabel.href = citation.url; sourceLabel.target = "_blank"; sourceLabel.rel = "noopener noreferrer"; }
    meta.append(sourceLabel, node("span", "", citation.artifact_id || "evidence"));
    const stats = node("div", "citation-stats");
    const statItems = [
      `文件 ${bytes(Number(citation.document_bytes || citation.indexed_bytes || 0))}`,
      `索引 ${bytes(Number(citation.indexed_bytes || 0))}`,
      `证据块 ${bytes(Number(citation.chunk_bytes || 0))}`,
    ];
    if (Number(citation.shard_count || 0) > 1) statItems.push(`Agent ${number(citation.shard_index)} / ${number(citation.shard_count)}`);
    if (citation.source_exceeds_shard_limit) statItems.push(`超过 ${bytes(Number(citation.shard_byte_limit || 0))} · 已分片`);
    statItems.forEach((label, index) => stats.append(node("span", index === statItems.length - 1 && citation.source_exceeds_shard_limit ? "split" : "", label)));
    card.append(meta, stats, node("p", "", citation.excerpt || ""));
    container.append(card);
  });
  if (!citations.length) container.append(node("p", "empty-evidence", generalChat ? "通用回答未使用文档证据。" : "没有返回可验证的证据引用。"));
  section.append(heading, container);
  return section;
}

function createTrace(result) {
  const details = node("details", "trace-details");
  details.append(node("summary", "", "查看多 Agent 图执行轨迹"));
  const trace = node("div", "trace");
  (result.trace || []).forEach((step, index) => {
    const item = node("div", "trace-step"); item.dataset.step = String(index + 1);
    item.append(node("strong", "", `${step.role || step.node} · ${step.status}`), node("span", "", step.detail || ""));
    trace.append(item);
  });
  details.append(trace);
  return details;
}

function renderTurn(turn, isLatest) {
  const result = turn.result;
  const labels = resultLabels(result);
  const article = node("article", "conversation-turn"); article.dataset.turnId = turn.id;
  const question = node("div", "message question-message");
  const questionMeta = node("div", "question-meta");
  questionMeta.append(node("span", "message-label", "你的问题"), node("time", "", new Date(turn.created_at).toLocaleString("zh-CN", { hour12: false })));
  question.append(questionMeta, node("p", "", turn.question));

  const answer = node("div", "message answer-message");
  const answerHead = node("div", "message-head");
  const actions = node("div");
  const status = node("span", "status-pill");
  setStatus(status, result.validation?.approved === false ? "warning" : "success", labels.status);
  const copy = node("button", "icon-text-button", "复制"); copy.type = "button"; copy.dataset.action = "copy-turn"; copy.dataset.turnId = turn.id;
  copy.prepend(icon("M8 8h11v11H8zM5 16H4V5h11v1"));
  actions.append(status, copy);
  answerHead.append(node("span", "message-label", labels.mode), actions);
  answer.append(answerHead, node("div", "answer-content", result.answer || ""));

  const read = result.document_read;
  if (read) {
    const nav = node("div", "document-read-nav");
    const previous = node("button", "button secondary", "上一段"); previous.type = "button"; previous.disabled = !read.has_previous; previous.dataset.action = "read-document"; previous.dataset.offset = String(read.previous_offset || 0); previous.dataset.question = turn.question;
    const next = node("button", "button secondary", "下一段"); next.type = "button"; next.disabled = !read.has_more; next.dataset.action = "read-document"; next.dataset.offset = String(read.next_offset || 0); next.dataset.question = turn.question;
    nav.append(previous, node("span", "", `字符 ${number(read.start_character + 1)}–${number(read.end_character)} / ${number(read.total_characters)}`), next);
    answer.append(nav);
  }

  const evidence = node("details", "turn-evidence"); evidence.open = isLatest;
  const citationCount = (result.citations || []).length;
  const summary = node("summary");
  summary.append(node("span", "", "执行详情"), node("small", "", `${citationCount} 条来源 · ${number(result.capacity_report?.model_calls)} 次模型调用`));
  const body = node("div", "turn-evidence-body");
  if (!labels.generalChat) body.append(createContextProof(result));
  body.append(createCitationSection(result, labels.generalChat), createTrace(result));
  evidence.append(summary, body);
  article.append(question, answer, evidence);
  return article;
}

function renderConversation() {
  elements.resultContent.replaceChildren(...state.turns.map((turn, index) => renderTurn(turn, index === state.turns.length - 1)));
  updateConversationState();
}

function renderResult(result, question) {
  const turn = {
    id: (globalThis.crypto?.randomUUID?.() || `turn_${Date.now()}_${Math.random().toString(16).slice(2)}`),
    question,
    created_at: new Date().toISOString(),
    document_name: state.document?.name || null,
    result: compactResult(result),
  };
  state.turns.push(turn);
  persistConversation();
  renderConversation();
  window.requestAnimationFrame(() => elements.resultPanel.scrollTo({ top: elements.resultPanel.scrollHeight, behavior: "smooth" }));
}

async function runResearch() {
  const button = $("#run-research");
  try {
    const question = elements.question.value.trim();
    if (!question) throw new Error("请输入调研问题。");
    state.lastQuestion = question;
    const defaultAgents = Number(elements.defaultAgents.value);
    const maxWorkers = Number(elements.maxWorkers.value);
    if (!Number.isInteger(defaultAgents) || defaultAgents < 1 || defaultAgents > 32) throw new Error("默认 Agent 数必须在 1至32 之间。");
    if (!Number.isInteger(maxWorkers) || maxWorkers < defaultAgents || maxWorkers > 32) throw new Error("最大 Agent 数必须大于等于默认 Agent 数，且不超过32。");
    const payload = { mode: state.mode, answer_scope: state.scope, web_search: $("#web-search-toggle").checked, question, history: modelHistory(), default_agents: defaultAgents, max_workers: maxWorkers, reduce_fan_in: Number(elements.reduceFanIn.value) };
    if (state.mode === "live") {
      const config = apiConfig();
      if (config.base_url && config.api_key && config.model) payload.api = validateApiConfig();
    }
    setLoading(button, true);
    const result = await request("/api/ask", payload);
    renderResult(result, question);
    elements.question.value = "";
  } catch (error) { toast(error.message, "error"); }
  finally { setLoading(button, false); }
}

async function readDocumentPage(offset, question) {
  const button = $("#run-research");
  try {
    setLoading(button, true);
    const result = await request("/api/ask", {
      mode: state.mode,
      question: question || state.lastQuestion || "请输出全文。",
      document_read: true,
      read_offset: Number(offset || 0),
    });
    renderResult(result, question || state.lastQuestion || "请输出全文。");
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
elements.resultContent.addEventListener("click", async (event) => {
  const target = event.target.closest("[data-action]");
  if (!target) return;
  if (target.dataset.action === "copy-turn") {
    const turn = state.turns.find((item) => item.id === target.dataset.turnId);
    try { await navigator.clipboard.writeText(turn?.result?.answer || ""); toast("答案已复制。"); }
    catch { toast("浏览器不允许自动复制。", "error"); }
  }
  if (target.dataset.action === "read-document") await readDocumentPage(target.dataset.offset, target.dataset.question);
});
elements.clearConversation.addEventListener("click", () => {
  if (!state.turns.length || !window.confirm("确定清空当前浏览器中的全部对话记录吗？")) return;
  state.turns = [];
  localStorage.removeItem(CONVERSATION_STORAGE_KEY);
  renderConversation();
  toast("对话记录已清空。再提问会开始新的会话。");
});

setMode("live"); setScope("auto"); loadProfiles(); restoreConversation(); restoreCurrentDocument();
