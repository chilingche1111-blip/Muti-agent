"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const number = (value) => new Intl.NumberFormat("zh-CN").format(Number(value) || 0);
const roleNames = {
  supervisor: "Supervisor",
  fact_extractor: "事实抽取 Agent",
  analyst: "分析 Agent",
  risk_reviewer: "风险审查 Agent",
  comparator: "对比 Agent",
  reducer: "Reducer",
  validator: "Validator",
  finalizer: "Finalizer",
};

let mode = "offline";
let latestReport = null;
let verifiedApi = null;

async function get(path) {
  const response = await fetch(path);
  const data = await response.json();
  if (!response.ok || !data.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data.result;
}

async function post(path, payload = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data.result;
}

function node(tag, className = "", text = "") {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== "") item.textContent = text;
  return item;
}

function toast(message, error = false) {
  const item = node("div", `toast ${error ? "error" : ""}`, message);
  $("#toast-region").append(item);
  window.setTimeout(() => item.remove(), 3500);
}

function setStatus(element, kind, label) {
  element.className = `status-pill ${kind}`;
  element.replaceChildren(node("i"), document.createTextNode(label));
}

function setLoading(button, loading) {
  button.disabled = loading;
  button.classList.toggle("loading", loading);
}

function loadVerifiedApi() {
  try {
    verifiedApi = JSON.parse(localStorage.getItem("context-atlas-verified-api") || "null");
  } catch {
    verifiedApi = null;
  }
  if (verifiedApi && Number(verifiedApi.expires_at || 0) <= Math.floor(Date.now() / 1000)) {
    localStorage.removeItem("context-atlas-verified-api");
    verifiedApi = null;
  }
  $("#verified-api-name").textContent = verifiedApi?.name || "未找到已验证配置";
  $("#verified-api-base-url").textContent = verifiedApi?.base_url || "—";
  $("#verified-api-model").textContent = verifiedApi?.model || "—";
  $("#verified-api-time").textContent = verifiedApi?.verified_at
    ? new Date(verifiedApi.verified_at * 1000).toLocaleString("zh-CN")
    : "—";
  $("#test-api-connection").disabled = !verifiedApi;
  $("#clear-verified-api").disabled = !verifiedApi;
  setStatus(
    $("#test-connection-status"),
    verifiedApi ? "success" : "error",
    verifiedApi ? "已复用" : "尚未验证",
  );
  $("#verified-api-hint").textContent = verifiedApi
    ? "该配置来自研究系统最近一次成功的连接测试；真实 API 基准将直接使用同一临时句柄。"
    : "请先返回研究系统填写 API，并点击“测试连接”。测试中心不接受另一套临时 API。";
  return verifiedApi;
}

function requireVerifiedApi() {
  if (!verifiedApi?.profile_token) throw new Error("没有可复用的已验证 API，请先在研究系统完成连接测试。");
  return verifiedApi.profile_token;
}

async function confirmVerifiedApiHandle() {
  if (!verifiedApi?.profile_token) return false;
  try {
    await post("/api/test/api-profile", { api_profile_token: verifiedApi.profile_token });
    return true;
  } catch {
    localStorage.removeItem("context-atlas-verified-api");
    loadVerifiedApi();
    return false;
  }
}

function benchmarkConfig() {
  const config = {
    max_workers: Number($("#benchmark-workers").value),
    reduce_fan_in: Number($("#benchmark-fan-in").value),
    max_replans: Number($("#benchmark-replans").value),
  };
  if (!Number.isInteger(config.max_workers) || config.max_workers < 1 || config.max_workers > 32) throw new Error("最大专业 Agent 必须在 1至32 之间。");
  if (!Number.isInteger(config.reduce_fan_in) || config.reduce_fan_in < 2 || config.reduce_fan_in > 8) throw new Error("Reducer 扇入必须在 2至8 之间。");
  if (!Number.isInteger(config.max_replans) || config.max_replans < 0 || config.max_replans > 3) throw new Error("最大重规划必须在 0至3 之间。");
  return config;
}

async function showWorkspace(user) {
  $("#login-panel").hidden = true;
  $("#test-workspace").hidden = false;
  $("#session-user").textContent = user;
  const info = await post("/api/test/document");
  $("#test-bytes").textContent = `${(info.bytes / 1024 / 1024).toFixed(2)} MB`;
  $("#test-chunks").textContent = `${number(info.chunks)} / ${number(info.sections)}`;
  $("#test-document-tokens").textContent = number(info.estimated_tokens);
  setStatus($("#test-document-status"), info.exceeds_64k_tokens ? "success" : "error", info.exceeds_64k_tokens ? "资料 > 64K" : "资料不足 64K");
  loadVerifiedApi();
  await confirmVerifiedApiHandle();
}

function renderGates(assertions) {
  const container = $("#acceptance-gates");
  container.replaceChildren();
  assertions.forEach((assertion, index) => {
    const row = node("article", `gate-row ${assertion.passed ? "pass" : "fail"}`);
    const mark = node("span", "gate-mark", assertion.passed ? "✓" : "×");
    const copy = node("div", "gate-copy");
    copy.append(node("strong", "", `${index + 1}. ${assertion.label}`), node("small", "", `期望：${assertion.expected}`));
    const actual = node("div", "gate-actual");
    actual.append(node("strong", "", assertion.actual), node("span", "", assertion.passed ? "通过" : "失败"));
    row.append(mark, copy, actual);
    container.append(row);
  });
  const passed = assertions.filter((item) => item.passed).length;
  setStatus($("#gate-score"), passed === assertions.length ? "success" : "error", `${passed} / ${assertions.length}`);
}

function renderFailureSummary(report) {
  const container = $("#failure-summary");
  const failedCases = report.results.filter((item) => !item.passed);
  container.replaceChildren();
  container.hidden = failedCases.length === 0;
  if (!failedCases.length) return;
  const header = node("div", "failure-summary-head");
  header.append(
    node("strong", "", `${failedCases.length} 个用例未通过`),
    node("span", "", "这表示模型质量或证据链存在问题，不等同于 64K 上下文越界。"),
  );
  const list = node("div", "failure-summary-list");
  failedCases.forEach((result) => {
    const primary = result.failure_reasons?.[0];
    const item = node("button", "failure-summary-item");
    item.type = "button";
    item.dataset.failureCase = String(report.results.indexOf(result));
    const copy = node("div");
    copy.append(
      node("strong", "", `${result.id} · ${primary?.stage || "未知阶段"}`),
      node("span", "", primary?.message || "没有返回具体失败原因"),
    );
    item.append(copy, node("code", "", primary?.code || "unknown_failure"), node("b", "", "查看审计 →"));
    list.append(item);
  });
  container.append(header, list);
}

function renderCases(results) {
  const container = $("#benchmark-results");
  container.replaceChildren();
  results.forEach((result, index) => {
    const metrics = result.context_metrics;
    const utilization = Number(metrics.max_window_utilization_percent || 0);
    const card = node("button", `benchmark-case ${result.passed ? "pass" : "fail"}`);
    card.type = "button";
    card.dataset.caseIndex = String(index);
    const head = node("div", "benchmark-case-head");
    head.append(node("strong", "", result.id), node("span", `case-state ${result.passed ? "pass" : "fail"}`, result.passed ? "通过" : "失败"));
    const question = node("p", "case-question", result.question);
    const position = node("div", "case-position");
    const positionHead = node("div");
    positionHead.append(node("span", "", "最大窗口占用"), node("strong", "", `${utilization.toFixed(2)}%`));
    const track = node("div", "case-mini-track");
    const bar = node("i");
    bar.style.width = `${Math.min(100, utilization)}%`;
    track.append(bar);
    position.append(positionHead, track);
    const stats = node("div", "case-metrics");
    [["子任务", metrics.task_count], ["引用", result.citations.length], ["模型调用", metrics.model_calls], ["Validator", result.validation.approved ? "通过" : "拒绝"]].forEach(([label, value]) => {
      const item = node("div"); item.append(node("span", "", label), node("strong", "", String(value))); stats.append(item);
    });
    card.append(head, question, position, stats);
    if (!result.passed) {
      const primary = result.failure_reasons?.[0];
      card.append(node("p", "case-failure", `原因：${primary?.message || "未返回具体原因"}`));
    }
    card.append(node("span", "case-open", "查看完整审计 →"));
    container.append(card);
  });
}

function renderRoleAccounting(results) {
  const roles = new Map();
  results.forEach((result) => {
    Object.entries(result.context_metrics.by_role || {}).forEach(([role, data]) => {
      const current = roles.get(role) || { calls: 0, max: 0 };
      current.calls += Number(data.calls || 0);
      current.max = Math.max(current.max, Number(data.max_prompt_tokens || 0));
      roles.set(role, current);
    });
  });
  const container = $("#role-accounting");
  container.replaceChildren();
  [...roles.entries()].sort((a, b) => b[1].max - a[1].max).forEach(([role, data]) => {
    const percent = data.max / 64_000 * 100;
    const row = node("div", "role-row");
    const meta = node("div", "role-meta");
    meta.append(node("strong", "", roleNames[role] || role), node("span", "", `${data.calls} 次调用 · max ${number(data.max)} Token`));
    const track = node("div", "role-track");
    const bar = node("i"); bar.style.width = `${Math.min(100, percent)}%`; track.append(bar);
    row.append(meta, track, node("b", "", `${percent.toFixed(2)}%`));
    container.append(row);
  });
}

function renderInspector(index) {
  if (!latestReport) return;
  const result = latestReport.results[index];
  const container = $("#case-inspector");
  container.replaceChildren();

  const summary = node("div", "inspector-summary");
  const question = node("div"); question.append(node("span", "", "测试问题"), node("strong", "", result.question));
  const answer = node("div"); answer.append(node("span", "", "最终回答"), node("strong", "", result.answer || "—"));
  const expected = node("div"); expected.append(node("span", "", "预期事实"), node("strong", "", result.expected_terms.join("；")));
  summary.append(question, answer, expected);

  const failureReasons = result.failure_reasons || [];
  const failurePanel = node("section", "case-failure-panel");
  if (failureReasons.length) {
    failurePanel.append(node("h3", "", `未通过原因（${failureReasons.length}）`));
    const reasonList = node("div", "reason-list");
    failureReasons.forEach((reason) => {
      const item = node("article");
      const heading = node("div");
      heading.append(node("strong", "", reason.stage), node("code", "", reason.code));
      item.append(heading, node("p", "", reason.message), node("span", reason.retryable ? "retryable" : "blocked", reason.retryable ? "可以重试" : "需要修复后再测"));
      reasonList.append(item);
    });
    failurePanel.append(reasonList);
  }

  const taskSection = node("section", "audit-section");
  taskSection.append(node("h3", "", "主 Agent 任务路由"));
  const taskGrid = node("div", "task-grid");
  result.tasks.forEach((task) => {
    const item = node("article");
    item.append(node("span", "task-id", task.task_id), node("strong", "", task.objective), node("small", "", `${roleNames[task.agent_type] || task.agent_type} · 优先级 ${task.priority} · 预算 ${number(task.input_budget)} Token`));
    taskGrid.append(item);
  });
  taskSection.append(taskGrid);

  const validationSection = node("section", "audit-section");
  validationSection.append(node("h3", "", "Validator 双层验证"));
  const validationGrid = node("div", "validation-grid");
  Object.entries(result.validation.hard_checks.checks || {}).forEach(([key, passed]) => {
    const item = node("div", passed ? "pass" : "fail");
    item.append(node("b", "", passed ? "✓" : "×"), node("span", "", key.replaceAll("_", " ")));
    validationGrid.append(item);
  });
  const semantic = node("div", result.validation.semantic_checks.passed ? "pass" : "fail");
  semantic.append(node("b", "", result.validation.semantic_checks.passed ? "✓" : "×"), node("span", "", `semantic check · ${result.validation.semantic_checks.notes}`));
  validationGrid.append(semantic);
  validationSection.append(validationGrid);

  const traceSection = node("section", "audit-section");
  traceSection.append(node("h3", "", "LangGraph 执行轨迹"));
  const timeline = node("ol", "trace-timeline");
  result.trace.forEach((step) => {
    const item = node("li");
    const title = node("div"); title.append(node("strong", "", step.role || step.node), node("span", "", step.status));
    item.append(title, node("p", "", step.detail || "—"));
    timeline.append(item);
  });
  traceSection.append(timeline);

  const citationSection = node("section", "audit-section");
  citationSection.append(node("h3", "", `证据引用（${result.citations.length}）`));
  const citations = node("div", "citation-list");
  result.citations.forEach((citation, citationIndex) => {
    const detail = node("details");
    const summaryNode = node("summary", "", `${citationIndex + 1}. ${citation.section || citation.chunk_id}`);
    detail.append(summaryNode, node("p", "", citation.excerpt || "无摘录"), node("code", "", citation.artifact_id));
    citations.append(detail);
  });
  citationSection.append(citations);

  container.append(summary);
  if (failureReasons.length) container.append(failurePanel);
  container.append(taskSection, validationSection, traceSection, citationSection);
  $$('[data-case-index]').forEach((card) => card.classList.toggle("selected", Number(card.dataset.caseIndex) === index));
}

function renderReport(report) {
  latestReport = report;
  const proof = report.context_proof;
  const passed = report.passed;
  $("#verdict-hero").className = `verdict-strip ${passed ? "pass" : "fail"}`;
  $("#verdict-eyebrow").textContent = passed ? "全部验收门禁通过" : "至少一项门禁失败";
  $("#verdict-title").textContent = passed ? "多 Agent 分治验证通过" : "多 Agent 分治验证未通过";
  $("#verdict-description").textContent = passed
    ? `完整资料 ${number(proof.logical_document_tokens)} Token；主 Agent 未读取原文，全部调用都在 64K 安全边界内。`
    : (() => {
        const failed = report.results.find((item) => !item.passed);
        const reason = failed?.failure_reasons?.[0]?.message || "请进入用例审计查看失败原因。";
        return `${failed?.id || "用例"} 未通过：${reason}`;
      })();
  $("#verdict-seal-text").textContent = passed ? "通过" : "失败";
  $("#kpi-logical").textContent = number(proof.logical_document_tokens);
  $("#kpi-prompt").textContent = number(proof.max_single_agent_prompt_tokens);
  $("#kpi-tasks").textContent = number(Math.max(...report.results.map((item) => item.context_metrics.task_count), 0));
  $("#chart-logical-value").textContent = `${number(proof.logical_document_tokens)} Token`;
  $("#chart-prompt-value").textContent = `${number(proof.max_single_agent_prompt_tokens)} Token`;
  const scale = Math.max(proof.logical_document_tokens, proof.physical_context_limit_tokens);
  $("#chart-logical-bar").style.width = `${proof.logical_document_tokens / scale * 100}%`;
  $("#chart-prompt-bar").style.width = `${proof.max_single_agent_prompt_tokens / scale * 100}%`;
  $(".bar.window").style.width = `${proof.physical_context_limit_tokens / scale * 100}%`;
  $(".limit-marker").style.left = `${proof.physical_context_limit_tokens / scale * 100}%`;
  const percent = Math.min(100, proof.max_single_agent_prompt_tokens / proof.physical_context_limit_tokens * 100);
  $("#gauge-value").textContent = `${percent.toFixed(2)}%`;
  $("#gauge-caption").textContent = proof.all_agent_calls_within_limit ? "所有 Agent 调用都保留了安全余量。" : "至少一个 Agent 调用存在越界风险。";
  $("#case-score strong").textContent = `${proof.passed_cases} / ${proof.total_cases}`;
  $("#run-meta").textContent = `${report.mode === "offline" ? "离线" : "真实 API"} · ${number(report.duration_ms)} ms · ${new Date(report.timestamp).toLocaleTimeString("zh-CN")}`;

  renderGates(report.assertions);
  renderFailureSummary(report);
  renderCases(report.results);
  renderRoleAccounting(report.results);
  const selector = $("#case-select");
  selector.disabled = false;
  selector.replaceChildren(...report.results.map((result, index) => {
    const option = node("option", "", `${result.id} · ${result.passed ? "通过" : "失败"}`);
    option.value = String(index);
    return option;
  }));
  renderInspector(0);
  $("#export-report").disabled = false;
  showResultView("overview");
}

function showResultView(name) {
  $$('[data-result-tab]').forEach((button) => {
    const active = button.dataset.resultTab === name;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  $$('[data-result-view]').forEach((view) => { view.hidden = view.dataset.resultView !== name; });
}

$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const result = await post("/api/auth/login", { username: $("#login-user").value, password: $("#login-password").value });
    $("#login-error").hidden = true;
    await showWorkspace(result.user);
  } catch (error) {
    $("#login-error").textContent = error.message;
    $("#login-error").hidden = false;
  }
});

$("#logout").addEventListener("click", async () => { await post("/api/auth/logout"); window.location.reload(); });

$$('[data-test-mode]').forEach((button) => button.addEventListener("click", () => {
  mode = button.dataset.testMode;
  $$('[data-test-mode]').forEach((item) => item.classList.toggle("active", item === button));
  if (mode === "live") {
    $("#verified-api-card").scrollIntoView({ behavior: "smooth", block: "center" });
    $("#verified-api-card").classList.add("attention");
    window.setTimeout(() => $("#verified-api-card").classList.remove("attention"), 1200);
    if (!verifiedApi) toast("请先在研究系统测试 API 连接，测试中心会自动复用。", true);
  }
}));

$("#clear-verified-api").addEventListener("click", () => {
  localStorage.removeItem("context-atlas-verified-api");
  loadVerifiedApi();
  toast("已清除测试中心的 API 复用句柄。");
});

window.addEventListener("storage", (event) => {
  if (event.key === "context-atlas-verified-api") loadVerifiedApi();
});

$("#test-api-connection").addEventListener("click", async () => {
  try {
    setStatus($("#test-connection-status"), "loading", "测试中");
    await post("/api/test/api-connection", { api_profile_token: requireVerifiedApi() });
    setStatus($("#test-connection-status"), "success", "复用正常");
  } catch (error) {
    setStatus($("#test-connection-status"), "error", "连接失败");
    if (String(error.message).includes("不存在或已过期")) {
      localStorage.removeItem("context-atlas-verified-api");
      loadVerifiedApi();
    }
    toast(error.message, true);
  }
});

$("#run-benchmark").addEventListener("click", async () => {
  const button = $("#run-benchmark");
  try {
    setLoading(button, true);
    $("#run-meta").textContent = "正在执行 Supervisor → Specialists → Validator…";
    $("#verdict-eyebrow").textContent = "验收运行中";
    $("#verdict-title").textContent = "正在收集可审计证据";
    const payload = { mode, ...benchmarkConfig() };
    if (mode === "live") payload.api_profile_token = requireVerifiedApi();
    renderReport(await post("/api/test/benchmark", payload));
    toast("完整验收已完成。");
  } catch (error) {
    $("#run-meta").textContent = "运行失败";
    toast(error.message, true);
  } finally {
    setLoading(button, false);
  }
});

$("#benchmark-results").addEventListener("click", (event) => {
  const card = event.target.closest("[data-case-index]");
  if (!card) return;
  const index = Number(card.dataset.caseIndex);
  $("#case-select").value = String(index);
  renderInspector(index);
  showResultView("inspector");
  $(".results-workbench").scrollIntoView({ behavior: "smooth", block: "start" });
});

$("#failure-summary").addEventListener("click", (event) => {
  const item = event.target.closest("[data-failure-case]");
  if (!item) return;
  const index = Number(item.dataset.failureCase);
  $("#case-select").value = String(index);
  renderInspector(index);
  showResultView("inspector");
});

$("#case-select").addEventListener("change", (event) => renderInspector(Number(event.target.value)));

$$('[data-result-tab]').forEach((button) => button.addEventListener("click", () => showResultView(button.dataset.resultTab)));

$("#export-report").addEventListener("click", () => {
  if (!latestReport) return;
  const blob = new Blob([JSON.stringify(latestReport, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `context-atlas-report-${new Date().toISOString().replaceAll(":", "-")}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});

(async () => {
  try {
    const session = await get("/api/auth/me");
    if (session.authenticated) await showWorkspace(session.user);
  } catch (error) {
    toast(error.message, true);
  }
})();
