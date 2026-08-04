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
  agentMonitor: $("#agent-monitor"), agentMonitorToggle: $("#agent-monitor-toggle"), agentMonitorSummary: $("#agent-monitor-summary"),
  agentMonitorContext: $("#agent-monitor-context"), agentMonitorContent: $("#agent-monitor-content"),
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
    tasks: (result.tasks || []).slice(0, 32).map((task) => ({
      task_id: task.task_id,
      agent_instance_id: task.agent_instance_id,
      agent_name: task.agent_name,
      agent_instruction: task.agent_instruction,
      objective: task.objective,
      capabilities: task.capabilities || [],
      tools: task.tools || [],
      source_policy: task.source_policy || "none",
      shard_index: task.shard_index,
      shard_count: task.shard_count,
      shard_bytes: task.shard_bytes,
      shard_tokens: task.shard_tokens,
      shard_chunks: task.shard_chunks,
      input_budget: task.input_budget,
      depends_on: task.depends_on || [],
      phase: task.phase || 0,
      required_targets: task.required_targets || [],
    })),
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
      ? `资料约 ${number(info.estimated_tokens)} Token，可由动态 Agent 分片处理`
      : "资料规模较小，可作为动态 Agent 的可选上下文");
  $("#empty-title").textContent = "资料已就绪，可以开始提问";
  $("#empty-copy").textContent = `${info.name} 已建立 ${number(info.chunks)} 个资料块；Supervisor 会按问题决定哪些 Agent 需要使用它。`;
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
    auto: "直接提问；系统会判断使用单模型还是动态多 Agent…",
    general: "使用底层 LLM 直接回答、写作、解释或编程…",
    agent: "描述目标，Supervisor 将即时生成并调度所需 Agent…",
    document: "针对已上传资料提问，可直接指定页码…",
  };
  elements.question.placeholder = placeholders[scope] || placeholders.auto;
  updateComposerHint();
}

function updateComposerHint() {
  const webEnabled = Boolean($("#web-search-toggle")?.checked);
  const modeText = "智能回答";
  let routeText;
  if (state.scope === "general") routeText = "单模型直接回答";
  else if (state.scope === "agent") routeText = state.document ? "动态生成 Agent；资料作为可选工具" : "动态生成 Agent；使用模型通用能力";
  else if (state.scope === "document") routeText = state.document ? "动态 Agent 受当前资料约束" : "需要先上传资料";
  else routeText = state.document ? "自动选择执行方式；资料仅在需要时使用" : "普通聊天直答，其他问题动态生成 Agent";
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
    mode: documentRead ? "PDF / 文档原文" : (generalChat ? "LLM 直接回答" : (offlineDemo ? "流程演示结果" : "动态多 Agent 回答")),
    status: documentRead ? "原文直接读取" : (generalChat ? "直接回答" : (offlineDemo ? "测试模型" : (result.validation?.approved ? "Validator 通过" : "Validator 拒绝"))),
  };
}

function createContextProof(result) {
  const report = result.capacity_report || {};
  const card = node("section", "context-proof-card");
  const head = node("div", "proof-card-head");
  const title = node("div");
  title.append(node("span", "section-kicker", "Context isolation"), node("h3", "", "动态 Agent 上下文隔离证明"));
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
    ["动态 Worker 隔离", report.isolated_worker_contexts ? "是" : "否"],
    ["主 Agent 接收原文", report.supervisor_received_raw_document ? "是" : "否"],
    ["模型调用", `${number(report.model_calls)} 次`],
  ].forEach(([label, value]) => { const item = node("span", "", `${label} `); item.append(node("strong", "", value)); meta.append(item); });
  card.append(head, flow, meta);
  if (report.agent_allocation_reason) card.append(node("p", "allocation-note", `分配依据：${report.agent_allocation_reason}`));
  return card;
}

function createAgentVisualization(result) {
  const report = result.capacity_report || {};
  const tasks = result.tasks || [];
  const allocated = Number(report.allocated_agents || tasks.length || 0);
  if (allocated < 2 || tasks.length < 2) return null;

  const desired = Number(report.desired_agents || allocated);
  const maximum = Math.max(allocated, Number(report.max_agents || allocated));
  const defaultAgents = Number(report.default_agents || 0);
  const capped = desired > allocated;
  const saturated = allocated >= maximum;
  const section = node("section", "agent-allocation-panel");
  section.setAttribute("aria-label", `Agent 调度视图：系统需要 ${desired} 个，实际分配 ${allocated} 个`);

  const head = node("div", "agent-allocation-head");
  const title = node("div");
  title.append(node("span", "section-kicker", "Agent orchestration"), node("h3", "", "Agent 调度视图"));
  const stateLabel = capped ? "受最大数量限制" : (saturated ? "已用满配置上限" : "自动分配完成");
  head.append(title, node("span", `allocation-state ${capped ? "capped" : (saturated ? "warning" : "success")}`, stateLabel));

  const metrics = node("div", "allocation-metrics");
  [
    ["默认下限", `${number(defaultAgents)} 个`],
    ["系统需求", `${number(desired)} 个`],
    ["实际分配", `${number(allocated)} 个`],
    ["配置上限", `${number(maximum)} 个`],
  ].forEach(([label, value]) => {
    const metric = node("div"); metric.append(node("span", "", label), node("strong", "", value)); metrics.append(metric);
  });

  const capacity = node("div", "allocation-capacity");
  const capacityLabels = node("div");
  capacityLabels.append(node("span", "", `容量占用 ${number(allocated)} / ${number(maximum)}`), node("strong", "", `${Math.round(allocated / Math.max(1, maximum) * 100)}%`));
  const track = node("div", "allocation-track");
  const fill = node("i"); fill.style.width = `${Math.min(100, allocated / Math.max(1, maximum) * 100)}%`; track.append(fill);
  capacity.append(capacityLabels, track);

  const coverageMonitor = createCoverageMonitor(result);

  const pipeline = node("div", "agent-pipeline");
  const supervisor = node("div", "pipeline-node supervisor-node");
  supervisor.append(
    node("span", "pipeline-node-index", "01"),
    node("strong", "", "Supervisor"),
    node("small", "", `规划 ${number(tasks.length)} 个隔离子任务 · ${number(report.dependency_phases || 1)} 个依赖波次`),
  );
  const firstArrow = node("div", "pipeline-arrow"); firstArrow.append(icon("M5 12h14m-5-5 5 5-5 5"));

  const agentStage = node("div", "agent-stage");
  const agentStageHead = node("div", "agent-stage-head");
  const sourceSummary = Number(report.source_indexed_bytes || 0) > 0
    ? `${number(report.source_indexed_bytes)} Bytes 外部资料按需使用`
    : "不依赖外部资料，保留模型通用能力";
  agentStageHead.append(node("span", "pipeline-node-index", "02"), node("strong", "", "运行时生成的 Agent"), node("small", "", sourceSummary));
  const grid = node("div", "agent-grid");
  tasks.forEach((task, index) => {
    const card = node("article", "agent-mini-card dynamic-agent-card");
    const cardHead = node("div");
    cardHead.append(node("span", "agent-number", `Agent ${String(index + 1).padStart(2, "0")}`), node("span", "agent-role", task.agent_name || "动态任务 Agent"));
    const shardIndex = Number(task.shard_index || index + 1);
    const shardCount = Number(task.shard_count || tasks.length);
    const sourceLine = Number(task.shard_chunks || 0) > 0
      ? `分片 ${number(shardIndex)} / ${number(shardCount)} · ${bytes(Number(task.shard_bytes || 0))} · 约 ${number(task.shard_tokens)} Token · ${number(task.shard_chunks)} 个资料块 · 独立 64K 窗口 / 安全输入 ${number(task.input_budget)} Token`
      : `独立 64K 窗口 · 最大安全输入 ${number(task.input_budget)} Token`;
    const tools = (task.tools || []).join(" · ") || "model_reasoning";
    const capabilities = (task.capabilities || []).join(" · ") || "通用推理";
    card.append(
      cardHead,
      node("strong", "agent-shard", task.objective || "独立完成当前子任务"),
      node("p", "agent-instruction", task.agent_instruction || "根据当前问题独立完成任务。"),
      node("p", "", sourceLine),
      node("small", "agent-tools", `能力：${capabilities}`),
      node("small", "agent-tools", `工具：${tools}`),
      node("small", "agent-tools", `依赖：${(task.depends_on || []).join("、") || "无"} · 波次 ${number(Number(task.phase || 0) + 1)}`),
      node("span", "agent-complete", "已完成"),
    );
    grid.append(card);
  });
  agentStage.append(agentStageHead, grid);

  const secondArrow = node("div", "pipeline-arrow"); secondArrow.append(icon("M5 12h14m-5-5 5 5-5 5"));
  const finish = node("div", "pipeline-finish");
  const reducer = node("div", "pipeline-node"); reducer.append(node("span", "pipeline-node-index", "03"), node("strong", "", "Tree Reducer"), node("small", "", "固定扇入分层汇总"));
  const validator = node("div", `pipeline-node validator-node ${result.validation?.approved ? "approved" : "rejected"}`);
  validator.append(node("span", "pipeline-node-index", "04"), node("strong", "", "Validator"), node("small", "", result.validation?.approved ? "验证通过" : "发现未通过项"));
  finish.append(reducer, validator);
  pipeline.append(supervisor, firstArrow, agentStage, secondArrow, finish);

  section.append(head, metrics, capacity);
  if (coverageMonitor) section.append(coverageMonitor);
  section.append(pipeline);
  if (report.agent_strategy) section.append(node("p", "allocation-reason", `生成策略：${report.agent_strategy}`));
  if (report.agent_allocation_reason) section.append(node("p", "allocation-reason", `数量依据：${report.agent_allocation_reason}`));
  if (capped) section.append(node("p", "allocation-warning", `系统计算需要 ${number(desired)} 个 Agent，但配置上限为 ${number(maximum)}；建议提高最大 Agent 数或缩小单次任务范围。`));
  return section;
}

function createCoverageMonitor(result) {
  const report = result.capacity_report || {};
  const coverage = report.coverage_report || {};
  const gates = report.quality_gates || {};
  const accounting = report.token_accounting || {};
  const hasCoverage = Boolean(coverage.required);
  if (!hasCoverage && !Object.keys(gates).length) return null;

  const panel = node("section", "coverage-monitor");
  panel.setAttribute("aria-label", "覆盖率与质量门禁");
  const head = node("div", "coverage-monitor-head");
  const title = node("div");
  title.append(node("span", "section-kicker", "Coverage & quality"), node("h4", "", "覆盖率与质量门禁"));
  const percent = Number(coverage.coverage_percent ?? 100);
  head.append(title, node("strong", coverage.complete === false ? "coverage-fail" : "coverage-pass", hasCoverage ? `${percent.toFixed(1)}%` : "无需资料"));

  const gateGrid = node("div", "quality-gate-grid");
  [
    ["容量", gates.capacity_passed !== false, `${number(report.max_accounted_total_tokens || report.max_single_agent_prompt_tokens)} / ${number(report.context_limit_tokens || 64_000)} Token`],
    ["覆盖", gates.coverage_passed !== false, hasCoverage ? `${number(coverage.covered_chunks)} / ${number(coverage.document_chunks)} 块` : "不需要来源"],
    ["答案", gates.answer_passed !== false, result.validation?.approved ? "Validator 通过" : "Validator 拒绝"],
  ].forEach(([label, passed, detail]) => {
    const item = node("div", passed ? "gate-pass" : "gate-fail");
    item.append(node("i", "", passed ? "✓" : "×"), node("span", "", label), node("strong", "", detail));
    gateGrid.append(item);
  });

  panel.append(head, gateGrid);
  if (hasCoverage) {
    const strip = node("div", "coverage-shard-strip");
    strip.setAttribute("role", "list");
    strip.setAttribute("aria-label", `${coverage.total_shards || 0} 个资料分片的覆盖状态`);
    (coverage.shards || []).forEach((shard) => {
      const segment = node("div", shard.complete ? "complete" : "incomplete");
      segment.setAttribute("role", "listitem");
      segment.setAttribute("title", `分片 ${shard.shard_index}：${shard.scanned_chunks}/${shard.assigned_chunks} 块；约 ${number(shard.packed_tokens)} Token`);
      segment.append(node("span", "", String(shard.shard_index)), node("small", "", shard.complete ? "完成" : "缺口"));
      strip.append(segment);
    });
    panel.append(strip);
  }

  const targetList = node("div", "coverage-targets");
  (coverage.targets || []).forEach((target) => {
    const statusMap = {
      resolved: ["已解决", "resolved"],
      not_found_after_full_coverage: ["全覆盖未找到", "not-found"],
      unresolved: ["候选存在但未回答", "unresolved"],
      coverage_incomplete: ["覆盖不足", "unresolved"],
    };
    const [label, className] = statusMap[target.status] || [target.status || "未知", "unresolved"];
    const item = node("div", className);
    item.append(node("strong", "", target.target), node("span", "", `${label} · 候选 ${number(target.candidate_matches)}`));
    targetList.append(item);
  });
  if (targetList.childElementCount) panel.append(targetList);

  const tokenMode = accounting.mode === "api_usage" ? "API 实测" : (accounting.mode === "mixed" ? "API实测 + 估算" : "保守估算");
  panel.append(node("p", "token-accounting-note", `Token 计量：${tokenMode} · 输出预留 ${number(accounting.output_reserve_tokens || 2_000)} · 安全余量 ${number(accounting.safety_margin_tokens || 1_000)}`));
  return panel;
}

function createCitationSection(result, generalChat) {
  const section = node("section");
  const citations = result.citations || [];
  const heading = node("div", "result-subheading");
  heading.append(node("h3", "", "证据明细"), node("span", "", `${citations.length} 条`));
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

function createSourceDisclosure(result, generalChat) {
  const citations = result.citations || [];
  if (!citations.length) return null;
  const details = node("details", "turn-source-disclosure");
  const summary = node("summary");
  const identity = node("span", "source-disclosure-title");
  identity.append(icon("M4 6.5h16M4 12h16M4 17.5h10"));
  const labels = node("span");
  labels.append(node("strong", "", "外部来源"), node("small", "", "展开查看 Agent 实际使用的资料与工具结果"));
  identity.append(labels);
  summary.append(identity, node("span", "source-count-badge", `${number(citations.length)} 条`));
  const body = node("div", "source-disclosure-body");
  body.append(createCitationSection(result, generalChat));
  details.append(summary, body);
  return details;
}

function createTrace(result) {
  const details = node("details", "trace-details");
  details.append(node("summary", "", "查看 Agent Loop 执行轨迹"));
  const trace = node("div", "trace");
  (result.trace || []).forEach((step, index) => {
    const item = node("div", "trace-step"); item.dataset.step = String(index + 1);
    item.append(node("strong", "", `${step.role || step.node} · ${step.status}`), node("span", "", step.detail || ""));
    trace.append(item);
  });
  details.append(trace);
  return details;
}

function renderTurn(turn) {
  const result = turn.result;
  const labels = resultLabels(result);
  const article = node("article", "conversation-turn"); article.dataset.turnId = turn.id;
  const question = node("div", "message question-message");
  const questionMeta = node("div", "question-meta");
  questionMeta.append(node("span", "message-label", "你的问题"), node("time", "", new Date(turn.created_at).toLocaleString("zh-CN", { hour12: false })));
  question.append(questionMeta, node("p", "", turn.question));

  const sourceDisclosure = createSourceDisclosure(result, labels.generalChat);

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

  const evidence = node("details", "turn-evidence");
  const citationCount = (result.citations || []).length;
  const summary = node("summary");
  summary.append(node("span", "", "执行详情"), node("small", "", `${citationCount} 条来源 · ${number(result.capacity_report?.model_calls)} 次模型调用`));
  const body = node("div", "turn-evidence-body");
  if (!labels.generalChat) body.append(createContextProof(result));
  body.append(createTrace(result));
  evidence.append(summary, body);
  article.append(question);
  if (sourceDisclosure) article.append(sourceDisclosure);
  article.append(evidence, answer);
  return article;
}

function renderAgentMonitor() {
  const turn = [...state.turns].reverse().find((item) => {
    const result = item?.result || {};
    return Number(result.capacity_report?.allocated_agents || 0) >= 2 && (result.tasks || []).length >= 2;
  });
  const visualization = turn ? createAgentVisualization(turn.result) : null;
  if (!turn || !visualization) {
    setAgentMonitorOpen(false);
    elements.agentMonitorToggle.disabled = true;
    elements.agentMonitorToggle.classList.remove("ready");
    elements.agentMonitorSummary.textContent = "暂无运行";
    elements.agentMonitorContext.textContent = "";
    elements.agentMonitorContent.replaceChildren();
    return;
  }
  const allocated = Number(turn.result.capacity_report?.allocated_agents || turn.result.tasks.length);
  const question = String(turn.question || "").trim();
  elements.agentMonitorToggle.disabled = false;
  elements.agentMonitorToggle.classList.add("ready");
  elements.agentMonitorSummary.textContent = `${number(allocated)} Agent`;
  elements.agentMonitorContext.textContent = `最近一次多 Agent 任务：${question.length > 90 ? `${question.slice(0, 90)}…` : question}`;
  elements.agentMonitorContent.replaceChildren(visualization);
}

function setAgentMonitorOpen(open) {
  const shouldOpen = Boolean(open && !elements.agentMonitorToggle.disabled);
  const shouldReturnFocus = !shouldOpen && document.activeElement === $("#agent-monitor-close");
  elements.agentMonitor.hidden = !shouldOpen;
  elements.agentMonitorToggle.setAttribute("aria-expanded", String(shouldOpen));
  if (shouldOpen) $("#agent-monitor-close").focus();
  else if (shouldReturnFocus) elements.agentMonitorToggle.focus();
}

function renderConversation() {
  elements.resultContent.replaceChildren(...state.turns.map((turn) => renderTurn(turn)));
  renderAgentMonitor();
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
elements.agentMonitorToggle.addEventListener("click", () => setAgentMonitorOpen(elements.agentMonitor.hidden));
$("#agent-monitor-close").addEventListener("click", () => setAgentMonitorOpen(false));
document.addEventListener("keydown", (event) => { if (event.key === "Escape" && !elements.agentMonitor.hidden) setAgentMonitorOpen(false); });

setMode("live"); setScope("auto"); loadProfiles(); restoreConversation(); restoreCurrentDocument();
