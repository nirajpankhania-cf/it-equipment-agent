# Requirements: IT Equipment Request Handler

Derived from sections 1 and 3 of `docs/project-plan.md`. Policy numbers and reason codes here are the source of truth for `core.py` and its tests; change them only by commit.

## 1. Purpose

An agent receives a natural-language request for IT equipment, looks up the employee and the policy for their role through MCP tools, and returns one of three outcomes: **approve**, **deny** with the rule and next eligible date, or **escalate** to a human with a ticket ID and a reason code. The agent never approves without an `ELIGIBLE` result from the eligibility tool.

## 2. Request fields

| Field | Type | Notes |
|---|---|---|
| `employee_id` | string | Supplied alongside the text; identifies the requester |
| `item` | string | The equipment requested, as written by the requester; the agent maps it to a catalogue item |
| `reason` | free text | May claim an exception (damage, loss, theft, medical or accessibility need) |
| `submitted_on` | date | Passed to the eligibility check as its `as_of` date so decisions are reproducible |

## 3. Catalogue

Laptop, monitor, keyboard, mouse, headset, docking station. A request that cannot be mapped to exactly one of these is out of catalogue.

## 4. Policy table

Each cell gives `max_quantity` and `refresh_months` for the role and item.

| Role | Laptop | Monitor | Keyboard, mouse, headset | Docking station |
|---|---|---|---|---|
| Standard | 1, every 48 months | 1, every 36 months | 1 each, every 24 months | 1, every 36 months |
| Engineer | 1, every 36 months | 2, every 36 months | 1 each, every 24 months | 1, every 36 months |
| Manager | 1, every 24 months | 2, every 36 months | 1 each, every 24 months | 1, every 36 months |
| Contractor | Not eligible | 1, every 36 months | 1 each, every 24 months | Not eligible |

Tenure and equipment age are computed from `start_date` and `issued_date` against the `as_of` date, never stored.

## 5. Decision rules

Rules are checked in the order escalate, deny, approve. An escalation trigger takes precedence over everything else.

### Approve

The item is in the role's entitlement, and either:

- the employee holds fewer than `max_quantity` of the item, or
- the oldest unit held is at least `refresh_months` old on the `as_of` date.

### Deny

The policy gives a definite no and the request claims no exception. Either:

- the role is not entitled to the item, or
- the employee is at `max_quantity` and the refresh interval has more than 30 days left.

The response states the rule applied and the next eligible date.

### Escalate

Never decide automatically when any of these holds. Each has a reason code so tests can assert on it.

| Code | Trigger |
|---|---|
| `EMPLOYEE_NOT_FOUND` | The employee ID is not on file, so identity and role cannot be confirmed |
| `ITEM_NOT_IN_CATALOGUE` | The item is not in the catalogue, or the request is too vague to map to one item |
| `EXCEPTION_CLAIMED` | The request cites damage, loss, theft, or a medical or accessibility need, which the policy data cannot verify |
| `BORDERLINE_TIMING` | The employee is within 30 days of becoming eligible |
| `DATA_INCOMPLETE` | The equipment record lacks an issue date, or the request contradicts the record (for example "my laptop is 4 years old" against a record showing 1 year) |
| `REFLECTION_FAILED` | The agent could not produce a grounded draft after one revision |

Every escalation writes one audit entry and returns a ticket ID, which the response must include.

### Deny versus escalate

Out of policy with nothing else going on is a denial. Out of policy with a claimed exception, or anything the data cannot settle, is an escalation.

## 6. Success criteria

| Criterion | Target | How it is measured |
|---|---|---|
| Rubric evidence | 9 of 9 PDF items captured | Checklist in plan section 13 |
| Tool correctness | 4 of 4 tools, each with normal and edge-case tests | `pytest` output |
| Unit test coverage of `core.py` | 90% or higher line coverage | `pytest --cov` in CI |
| Agent decisions | 5 of 5 demo requests reach the expected outcome | Evaluation set, plan section 8 |
| No ungrounded approvals | 0 approvals without an `ELIGIBLE` tool result | Guardrail test plus eval assertions |
| Escalations | Every escalation has a ticket ID and a reason code | Audit log inspection, eval assertions |
| Cost visibility | 100% of model calls have a cost record | Count of cost records equals count of model calls per run |
| Pipeline | Green run on `main` | GitHub Actions screenshot |

### Non-goals

No real HR or asset data. No web UI. No authentication on the MCP server (it runs locally over stdio). No real ticketing integration; escalation writes to a local audit file. No deployment.
