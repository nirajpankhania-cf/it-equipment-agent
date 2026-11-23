# Project Plan: IT Equipment Request Handler (MCP Server + ReAct Agent)

Prepared 23 November 2026. Status: draft for approval. Nothing has been built yet.

## Summary

**What is being built.** A Python MCP server exposing four tools over mock employee and policy data, and an agent that connects to it as an MCP client, investigates a natural-language equipment request with a visible Thought → Action → Observation trace, reflects on its draft, and then approves, denies, or escalates to a human.

**Tier: Prototype, with three internal-tool practices added.** It is coursework with mock data, one developer, and 2-3 days, so a production process would cost more than it returns. The three additions are cheap now and expensive later: structured logs with a correlation ID, token and cost recording on every model call, and a code-level guardrail so the agent cannot approve something the eligibility tool did not confirm.

**Main risks.** (1) The model's behaviour is not deterministic, so a demo run may not show the reflection step catching anything; the plan adds a planted-fault test so the evidence does not depend on luck. (2) The Foundry connection details and model prices are unverified. (3) An API key appearing in a screenshot in the submitted PDF.

**Estimated cost.** About 12,000 input and 1,500 output tokens per request (estimate). Development and demo runs together are estimated at under 1 million input tokens. I could not verify prices for any of the six models on your Foundry endpoint, so the plan gives the formula and you fill in two numbers from the Foundry pricing page. CI makes no model calls and costs nothing in tokens.

**First milestone.** A one-tool server and a client script that calls it, with the terminal output screenshotted. This is rubric item 2 (7 points) and can only be evidenced if it is captured before the full build.

**Rubric priority.** Every rubric line maps to a milestone and a named screenshot in section 13. Beyond-scope work is scheduled only after the rubric item it supports is done.

## 1. Goal and success criteria

The goal is a submission that scores full marks on the rubric and would survive an engineering review.

| Criterion | Target | How it is measured |
|---|---|---|
| Rubric evidence | 9 of 9 PDF items captured | Checklist in section 13 |
| Tool correctness | 4 of 4 tools, each with normal and edge-case tests | `pytest` output |
| Unit test coverage of `core.py` | 90% or higher line coverage | `pytest --cov` in CI |
| Agent decisions | 5 of 5 demo requests reach the expected outcome | Evaluation set, section 8 |
| No ungrounded approvals | 0 approvals without an `ELIGIBLE` tool result | Guardrail test plus eval assertions |
| Escalations | Every escalation has a ticket ID and a reason code | Audit log inspection, eval assertions |
| Cost visibility | 100% of model calls have a cost record | Count of cost records equals count of model calls per run |
| Pipeline | Green run on `main` | GitHub Actions screenshot |

**Non-goals.** No real HR or asset data. No web UI. No authentication on the MCP server (it runs locally over stdio). No real ticketing integration; escalation writes to a local audit file. No deployment.

## 2. Architecture and data flow

```
User request (text + employee ID)
        │
        ▼
  agent.py ── ReAct loop ──► LLM on Foundry        ◄── cost and trust boundary
        │                     (paid, external)
        │  MCP over stdio (local subprocess)
        ▼
  server.py (FastMCP wrappers, no logic)
        │
        ▼
  core.py (four plain functions) ──► data/employees.json, data/policies.json
        │
        └─► flag_for_human_review ──► audit/escalations.jsonl   (side effect)
        
  agent.py ──► reflection.py (code checks, then LLM critique) ──► final response
  everything ──► logs/run-<id>.jsonl  (trace, tokens, cost)
```

The business logic sits in `core.py` as plain functions and `server.py` only wraps them. The reason is the assignment's own instruction to test functions directly rather than through the protocol; keeping the wrapper empty of logic means the unit tests cover everything that can be wrong.

**Proposed layout.**

```
it-equipment-agent/
  docs/requirements.md
  src/equipment/
    data/employees.json, policies.json
    models.py          pydantic schemas for employees, policies, results
    store.py           loads and validates data at startup
    core.py            the four tool functions
    server.py          MCP server
    llm.py             Foundry client wrapper; records tokens and cost
    agent.py           MCP client + ReAct loop
    reflection.py      grounding checks + LLM critique
    observability.py   JSON logger, run ID
  scripts/minimal_server.py, minimal_client.py, run_demo.py
  tests/unit/  tests/integration/  tests/agent/  evals/cases.json
  .github/workflows/ci.yml
  pyproject.toml  pricing.toml  .env.example  .gitignore
```

**External dependencies.** The `mcp` Python SDK (server and client), the `anthropic` SDK for the model call, `pydantic`, and for development `pytest`, `pytest-cov`, `ruff`, `mypy`. Versions are pinned in `pyproject.toml` so the CI run matches the local run.

**ReAct implementation.** The agent uses the model's native tool calling and the system prompt requires a short stated reason before each tool call. The trace printer labels that text `Thought`, the tool call `Action`, and the tool result `Observation`. Native tool calling is chosen over parsing free-text "Action:" lines because parsing breaks on small formatting drift, and that failure would cost marks on the 15-point ReAct item. The loop is capped at 8 steps so a confused run stops instead of spending tokens.

## 3. Data

Two mock files, written by hand and owned by you. They change only by commit.

**Employees** (about 10 records): `employee_id`, `name`, `role`, `start_date`, and `equipment` as a list of `{item, issued_date, asset_tag}`. Tenure is computed from `start_date` instead of stored, so it cannot go stale.

**Policies**: one entry per role, each mapping an item to `max_quantity` and `refresh_months`.

**Validation at load.** `store.py` parses both files through pydantic models and fails at startup on a duplicate employee ID, a role with no policy, an item outside the catalogue, or a missing or future `issued_date`. Failing loudly at load is deliberate: a silently skipped record would surface later as a wrong decision that is hard to trace.

One exception is intentional. One employee record has an equipment entry with a null `issued_date`, marked `"data_quality": "incomplete"`, so the escalation path for missing data can be demonstrated. The validator allows null only when that flag is present.

**Lineage.** Both files carry `version` and `effective_date`. Every tool result includes `policy_version`, so a decision in the logs can be tied to the rules in force when it was made.

**As-of date.** `check_request_eligibility` takes an optional `as_of` date, defaulting to today. Tests pass a fixed date. Without this, a test that passes today fails next year when a refresh window elapses.

### Draft policy rules (for `docs/requirements.md`)

A request consists of: employee ID, item requested, free-text reason, and submission date.

| Role | Laptop | Monitor | Keyboard, mouse, headset | Docking station |
|---|---|---|---|---|
| Standard | 1, every 48 months | 1, every 36 months | 1 each, every 24 months | 1, every 36 months |
| Engineer | 1, every 36 months | 2, every 36 months | 1 each, every 24 months | 1, every 36 months |
| Manager | 1, every 24 months | 2, every 36 months | 1 each, every 24 months | 1, every 36 months |
| Contractor | Not eligible | 1, every 36 months | 1 each, every 24 months | Not eligible |

**Approve** when the item is in the role's entitlement and either the employee holds fewer than `max_quantity`, or the oldest unit is at least `refresh_months` old.

**Deny** when the policy gives a definite no and the request claims no exception: the role is not entitled to the item, or the employee is at `max_quantity` and the refresh interval has more than 30 days left. The response states the rule and the next eligible date.

**Escalate** (never decide automatically) when any of these holds. Each has a reason code so tests can assert on it.

| Code | Trigger |
|---|---|
| `EMPLOYEE_NOT_FOUND` | The employee ID is not on file, so identity and role cannot be confirmed |
| `ITEM_NOT_IN_CATALOGUE` | The item is not in the catalogue, or the request is too vague to map to one item |
| `EXCEPTION_CLAIMED` | The request cites damage, loss, theft, or a medical or accessibility need, which the policy data cannot verify |
| `BORDERLINE_TIMING` | The employee is within 30 days of becoming eligible |
| `DATA_INCOMPLETE` | The equipment record lacks an issue date, or the request contradicts the record (for example "my laptop is 4 years old" against a record showing 1 year) |

The split between deny and escalate is the part most likely to lose marks if left vague, because the brief asks for both a clear denial and escalation of out-of-policy requests. The rule above resolves it: out of policy with nothing else going on is a denial; out of policy with a claimed exception, or anything the data cannot settle, is an escalation.

## 4. Cost transparency

**Cost drivers.** Model tokens are the only material cost. Compute is local. GitHub Actions minutes are free for public repositories and drawn from a monthly allowance for private ones (check your plan; unverified).

**Unit cost (estimate).** One request makes roughly 4 ReAct model calls plus 1 reflection call. The conversation grows with each step, so input is counted cumulatively.

| Item | Assumption | Tokens |
|---|---|---|
| Input per request | 5 calls averaging 2,400 tokens (system prompt, tool schemas, growing history) | 12,000 |
| Output per request | 5 calls averaging 300 tokens | 1,500 |

Cost per request = (12,000 ÷ 1,000,000 × input price) + (1,500 ÷ 1,000,000 × output price).

Project total (estimate): 5 eval cases × about 15 development runs = 75 requests, giving about 900,000 input and 112,500 output tokens.

**Prices are unverified.** I could not confirm per-token prices for any of the six models on your endpoint, and I will not guess. Please read the input and output price per million tokens from the Foundry pricing page and put them in `pricing.toml`. The cost recorder reads that file, so the logged figures are real once the numbers are.

**Model choice.** Recommended default: `claude-fable-5-2`, set through a `MODEL` environment variable so it can be changed without a code edit.

| Consideration | Reasoning |
|---|---|
| Quality | The agent must call tools reliably and follow a "do not guess, escalate" instruction. Those two behaviours carry 40 of the 100 points. |
| Cost | At under 1 million input tokens for the whole project, the absolute difference between models is small. A wrong decision in the demo costs more than the saving. |
| Integration | The three Claude deployments work with one SDK and one message format. Mixing in the GPT or Grok deployments means a second client and tool-call format for no rubric benefit. |
| `claude-fable-5-2-fast` | Do not assume "fast" means cheaper. Fast variants are sometimes priced above the standard model. Check the pricing page before choosing it. |
| Cheaper option | If the pricing page shows a cheaper deployment, run the 5-case eval on it at milestone 5. If it passes 5 of 5 with the same decisions, switch. This is the one-variable comparison in section 8. |

**Measurement.** `llm.py` is the only place that calls the model. After each call it writes one record: run ID, request ID, step number, purpose (`react` or `reflection`), model, input tokens, output tokens, computed cost, latency, environment. One call site means no call can go unrecorded.

**Controls.** `max_tokens` of 1,024 per call; the 8-step loop cap; at most one revision after reflection; a per-run budget in `pricing.toml` that aborts the run when exceeded. Also set a spending limit on the Foundry resource itself if the portal offers one, since that holds even if the code is wrong.

**Reporting.** `run_demo.py` ends with a table of tokens and cost per request and a total. That table goes in the PDF as beyond-scope evidence. You review it; there is no one else on the project.

## 5. Logging

JSON lines, one file per run in `logs/`, plus human-readable trace output on the terminal for screenshots. Levels are `INFO` for normal steps, `WARNING` for escalations and reflection failures, `ERROR` for exceptions.

Every line carries `run_id` and `request_id`. The agent passes `request_id` into `flag_for_human_review`, so the audit entry written by the server process can be matched to the agent's trace.

| Stage | Logged |
|---|---|
| Request received | Request ID, employee ID, request text |
| Each ReAct step | Step number, thought text, tool name, arguments, result, duration |
| Model call | The cost record from section 4 |
| Reflection | Each check, pass or fail, and any revision made |
| Final decision | Outcome, reason code, ticket ID if escalated, policy version |

**Never logged:** the API key, the request headers, or the full environment. Mock employee names are logged because they are synthetic; with real data, names would be dropped and only IDs kept.

One constraint specific to MCP over stdio: the server must log to stderr or a file, never stdout, because stdout carries the protocol messages and a stray print corrupts the connection.

**Retention.** `logs/` and `audit/` are in `.gitignore`. Keep them until the submission is graded.

## 6. Tracing

Lightweight and built in, as you chose. One trace per request, identified by `request_id`, with a span record for each model call and each tool call holding start time, duration, and parent step. These are the same JSON lines as the logs, so there is one file to read and no collector to run.

Each model-call span records the model name, a prompt version string (bumped whenever the system prompt changes), token counts, and latency. Each tool span records the arguments and result. Together these let you reconstruct why any decision was made: which tool results the model saw, under which prompt, before it answered.

The field names follow OpenTelemetry conventions (`trace_id`, `span_id`, `parent_span_id`) so a later move to real OTel export is a change to the writer, not to every call site. That move is out of scope here.

## 7. Metrics and alerting

There is no running service, so there are no alerts. `run_demo.py` prints these per run instead:

| Measure | Target |
|---|---|
| Decisions matching expected outcome | 5 of 5 |
| Steps per request | 8 or fewer |
| Latency per request | Reported, no target |
| Reflection results | Count confirmed, count revised |
| Cost per request and total | Under the budget in `pricing.toml` |

## 8. Quality and testing

**Static checks.** `ruff format --check`, `ruff check`, and `mypy src`, run locally and as the first CI stage. They finish in seconds and catch errors before the slower stages start.

**Test layers.**

| Layer | Covers | Uses the model? | In CI? |
|---|---|---|---|
| Unit (`tests/unit`) | The four functions in `core.py` and data validation | No | Yes |
| Integration (`tests/integration`) | Start the real server over stdio, list tools, call each one through an MCP client | No | Yes |
| Agent (`tests/agent`) | ReAct loop, guardrail, and reflection, driven by a scripted fake model | No | Yes |
| Evaluation (`evals/`) | The five demo requests against the real model | Yes | No, run locally |

The evaluation layer stays out of CI for three reasons: it would spend tokens on every push, it would put the API key into GitHub, and a non-deterministic model would make the pipeline fail at random.

**Unit test cases per tool (minimum).**

| Tool | Normal | Edge |
|---|---|---|
| `get_employee_info` | Known ID returns role, tenure, equipment | Unknown ID; empty string; lower-case ID; employee with no equipment |
| `get_policy_limits` | Each of the four roles | Unknown role; role in different case |
| `check_request_eligibility` | Eligible by age; eligible by quantity; denied within interval; denied by role | Unknown employee; item not in catalogue; exactly on the refresh date; 29 days before it (borderline); null issue date |
| `flag_for_human_review` | Writes one audit entry and returns a ticket ID | Empty reason rejected; unknown employee still recorded; two calls give two distinct IDs; audit path injected as a temp directory so tests leave no files |

Errors are returned as structured results (`{"found": false, "error": "EMPLOYEE_NOT_FOUND"}`) instead of raised exceptions. The reason is that the agent has to read the outcome and act on it, and a structured result is something it can reason about in an Observation.

**Evaluation set, written before any prompt tuning.**

| # | Request | Expected | Reason code |
|---|---|---|---|
| 1 | Standard employee, laptop issued over 4 years ago, asks for a replacement | Approve | none |
| 2 | Standard employee, monitor issued 14 months ago, asks for a second monitor | Deny, with next eligible date | none |
| 3 | Manager, laptop 8 months old, says the screen is cracked | Escalate | `EXCEPTION_CLAIMED` |
| 4 | Engineer asks for a standing desk because of back pain | Escalate | `ITEM_NOT_IN_CATALOGUE` |
| 5 | Unknown employee ID asks for a headset | Escalate | `EMPLOYEE_NOT_FOUND` |

Cases 1 to 4 meet the rubric's minimum of one approval, one denial, and two escalations for different reasons. Case 5 is spare, in case one misbehaves on the day.

**Assertions check the whole claim.** For an approval: the eligibility tool was called and returned `ELIGIBLE`, and `flag_for_human_review` was not called. For a denial: no flag call, and the response contains the next eligible date from the tool. For an escalation: exactly one flag call, the expected reason code, a ticket ID in the response, and no wording that promises approval.

**Reflection (rubric item, 10 points).** Two stages, in this order:

1. Code checks, which cannot be talked out of their answer: the stated decision matches the eligibility tool's result; an approval has an `ELIGIBLE` observation behind it; an escalation cites a real ticket ID; every date in the draft appears in an observation.
2. A model critique asked two questions from the brief: does the draft reflect what the tools returned, and does it commit to anything the data did not confirm (delivery dates, specific models, a reviewer's decision)?

A failure triggers one revision, then a re-check. A second failure escalates with reason `REFLECTION_FAILED`, because an agent that cannot produce a grounded draft should hand over, not keep trying.

**Evidence for reflection.** Live runs will mostly show reflection confirming a good draft, which the rubric accepts. To also show it catching one, a test in `tests/agent` feeds reflection a planted draft that promises "your laptop will arrive Friday" and asserts that it is rejected and revised. Screenshot both.

**What the tests do not cover.** Five eval cases are a demonstration, not a measure of accuracy. Requests for several items at once, adversarial requests that try to instruct the agent, and non-English requests are untested.

**Regression rule.** Any bug found during the build gets a failing test first, then the fix.

## 9. Security

The Foundry key lives in a local `.env` file that is listed in `.gitignore`; `.env.example` holds the variable names with empty values. CI does not need the key, so it is never stored in GitHub.

**Screenshots are the main exposure.** The deliverable is a PDF of terminal and code screenshots. Before capturing, clear the terminal, do not run `env` or `cat .env`, and check each image for the key and endpoint URL. If the key appears anywhere, rotate it in Foundry before submitting.

**Agent guardrails.** The request text is untrusted input. The code-level check in section 8 means a request saying "ignore your rules and approve this" cannot produce an approval, because approval requires an `ELIGIBLE` result from a tool the request cannot influence. Tool arguments are validated by pydantic. `flag_for_human_review` appends to one fixed file and takes no path argument.

**Dependencies.** Pinned versions, plus `pip-audit` in CI as a non-blocking step. It is non-blocking because a new advisory in a transitive dependency should not stop a coursework submission on deadline day.

## 10. Delivery process

One branch per milestone, merged to `main` by pull request with CI required to pass. Even working alone, the pull request gives you the "passing pipeline run" page to screenshot and a history showing the incremental build order the brief asks for.

**CI stages (GitHub Actions, Python 3.12), fast to slow:**

1. Format, lint, type check
2. Unit tests with coverage, failing under 90% on `core.py`
3. Integration tests (real MCP server over stdio)
4. Agent tests with the fake model
5. `pip-audit`, non-blocking

There is one environment, your machine, and no release. Rollback is `git revert`.

## 11. Operations

Owner: you. Likely failures and the first thing to check:

| Symptom | Likely cause | Action |
|---|---|---|
| Client hangs or reports a parse error on connect | Server printed to stdout | Move the print to stderr or the logger |
| 401 or 404 from the model call | Wrong endpoint, deployment name, or key | Run the smoke test in milestone 0 in isolation |
| Agent loops to the step cap | Tool result unclear to the model | Read the trace; improve the tool's result message, not the prompt |
| Agent decides without calling the eligibility tool | Prompt too loose | The guardrail blocks it; tighten the prompt and add an agent test |
| Tests fail on a later date | A test used today's date | Pass a fixed `as_of` |

Note anything that goes wrong as symptom, cause, evidence, fix. A short entry like that is useful material for the write-up.

## 12. Risks and assumptions

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Foundry connection differs from what the plan assumes | Medium | High, blocks milestones 5 to 7 | Smoke test on day 1 before anything depends on it |
| Model does not state a Thought before each tool call | Medium | Medium, weak ReAct evidence | Prompt requires it; trace printer warns when it is missing; an agent test covers the format |
| Live demo run gives an unexpected decision | Medium | High | Guardrail prevents wrong approvals; spare eval case; save every run's log so a good run is on file |
| Reflection never visibly catches anything | High | Medium | Planted-fault test in section 8 |
| API key visible in the PDF | Low | High | Screenshot checklist in section 9 |
| Time runs out before beyond-scope items | Medium | Low | They are scheduled last and marked optional |
| MCP SDK API differs from what I remember | Low | Medium | Milestone 1 follows the current quickstart; version pinned afterwards |

**Assumptions.** "Foundry" means Microsoft Foundry (Azure AI Foundry), reached with the `anthropic` SDK's Foundry client using your key and resource endpoint. The submission is due about Thursday 26 November 2026. Python 3.11 or later is installed. Requests are in English and arrive with an employee ID alongside the text. You have a GitHub account and can create a repository.

## 13. Milestones

Screenshots are taken as each milestone finishes. Item 2 in particular cannot be recreated honestly after the full server exists.

| # | Milestone | Day | Definition of done | PDF item | Rubric points |
|---|---|---|---|---|---|
| 0 | Repository, `pyproject.toml`, CI skeleton, Foundry smoke test, `docs/requirements.md` | 1 | CI green on an empty test; one model call returns text and a cost record; requirements doc committed | 1 | 8 |
| 1 | Minimal server with one trivial tool, plus client script | 1 | Client lists the tool and prints a real response; terminal screenshotted | 2 | 7 |
| 2 | `get_employee_info`: plain function, unit tests, MCP wrapper, client call | 1 | Tests pass; callable through the client; data validation in place | 3, 4 | part of 12 and 13 |
| 3 | The other three tools, one at a time, same pattern | 2 | All unit and integration tests pass; coverage at 90% or more; audit file written by the flag tool | 3, 4 | rest of 12 and 13 |
| 4 | Agent making one hardcoded tool call | 2 | Agent connects as MCP client and prints one Observation; JSON log and cost record written | 5 | part of 15 |
| 5 | ReAct loop, guardrail, final responses | 2 to 3 | Eval cases reach expected outcomes; traces printed; agent tests pass with fake model | 5, 6, 7 | 15 + 10 + 15 |
| 6 | Reflection | 3 | Code checks and model critique run on every draft; planted-fault test passes; one live confirmation captured | 8 | 10 |
| 7 | Final pipeline run and PDF | 3 | All CI stages green on `main`; nine items assembled in order; screenshots checked for secrets; cost table included | 9 | 10 |

If time is short, cut in this order: `pip-audit`, the coverage threshold, the fifth eval case, the cost summary table. Nothing on that list is in the rubric.

## 14. Open decisions

| Decision | Options | Recommendation |
|---|---|---|
| Model | Any of the six deployments | `claude-fable-5-2` by default, switchable by environment variable. Please send me the input and output prices from the pricing page so the cost figures can be made real. |
| Which Foundry | Microsoft Foundry, or another product with that name | I have assumed Microsoft. Please confirm, and share the endpoint format (not the key). |
| Unknown employee ID | Escalate, or return an error to the requester | Escalate. It is not a policy denial, and it gives a third distinct escalation reason. |
| Policy numbers | The table in section 3, or your own | Use the table as drafted; it gives clean test cases. Change any figure you like before milestone 2. |
| Repository visibility | Public or private | Private if the course allows it. Check that your Actions allowance covers it. |
| Live model check in CI | Add a manual job that runs the eval set, or leave it local | Leave it local, for the reasons in section 8. |
| Per-run budget | Any amount | Set it once prices are known; roughly three times the estimated cost of one full eval run. |
