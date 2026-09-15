# C1 — GPU first run

**Owner: C** · unblocks Dimitris and B.  
**Status:** CLI install in progress. **No GPU cluster. No spend.** Isaac Sim does not run on laptops.

This is the Solar Sheep walkthrough of Nebius `first-run-setup`. Do not skip a gate because a later step looks more interesting. Project ids live in `~/.npa/` after configure — do not copy them into new files.

Sibling workbench (already cloned, `npa` on PATH): `../nebius-physical-ai`.

---

## Evidence (this machine, 15 Sep 2026)

| Gate | State | Evidence |
|---|---|---|
| Python 3.12 / Terraform 1.x | **VERIFIED** | `python3` 3.12.2 · Terraform 1.16.2 |
| `npa` CLI | **VERIFIED** | `npa 0.1.0` at `~/.local/bin/npa` |
| `npa configure` project stanza | **FAILED** | `npa configure --show` → no projects in `~/.npa/config.yaml` |
| Nebius CLI on PATH | **FAILED** (pre-install) | `npa workbench health preflight --checks nebius` → `Nebius CLI is not available` |
| Token Factory key | **NOT VERIFIED** | START-HERE still-to-do; separate `v1.` key, not IAM |
| RTX PRO 6000 quota (API list) | **VERIFIED** in START-HERE (15 Sep) | Listing availability ≠ a running box |
| Isaac Sim / cluster | **NOT STARTED** | Do not provision until preflight is green **and** C says spend |

This Mac is Darwin arm64. Isaac Sim Docker is native-Linux x86_64. The box is a Nebius Linux GPU node; this laptop only runs `npa` / `nebius` over the API.

---

## Skills (C1)

Vendored beside B’s skills (`.agents/skills/`, symlinked from `.claude/skills/`):

| Skill | Role |
|---|---|
| `first-run-setup` | Ordered gates. Stop at the first red one. |
| `gpu-selection` | Isaac / cameras need **RT cores** → RTX PRO 6000 (`gpu-rtx6000`). Never H100 / H200 / B200. |
| `gpu-cluster-provisioning` | `npa cluster up --gpu-workload-profile rtx-rendering` |
| `token-factory` | Hosted-inference proof **before** a cluster. Key starts with `v1.` |
| `teardown-and-cost` | Cancel jobs, then `npa cluster down`. A first run is unfinished while the box is still up. |
| `isaac-lab` | Gen-3 is the payload-clean **container** on that cluster, not `--runtime vm` on the laptop. |

`gpu-rtx6000` is not serverless. Do not pass `--runtime serverless`.

---

## Gates (run in order)

### 0 — Nebius CLI binary

START-HERE install (WSL or macOS POSIX):

```bash
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
# new shell so PATH picks up `nebius`
nebius --help
```

**Gate:** `nebius` is on PATH. Then:

```bash
npa workbench health preflight --checks nebius --json
```

Until that check is no longer “CLI is not available”, do not configure, and do not provision.

### 1 — `npa` already installed

```bash
npa --version    # expect 0.1.0 (or whatever this machine actually prints)
npa --help
```

No cloud credentials required.

### 2 — Configure (HITL — browser login)

Interactive; federation opens a browser. Use the tenant/project already recorded in START-HERE. Prefer `--no-provision` so configure does **not** create a bucket as a side effect:

```bash
npa configure --show          # writes nothing
npa configure --no-provision  # interactive; creates/reuses the CLI profile
```

**Gate:** `npa configure --show` prints a real project stanza (alias, region, project id).  
**Stop here in the agent session** until a human finishes the browser step.

Unattended (still no bucket, still no GPU) only if the human already has the Nebius CLI profile:

```bash
npa configure --no-interactive --no-provision --save-env-credentials \
  --tenant-id <id> --project-id <id> --region <region> --project-alias solar
```

Record HTTP/CLI outcomes in gitignored `docs/access.md`. Never commit `~/.npa/credentials.yaml`.

### 3 — Preflight (no spend)

```bash
npa workbench health preflight --json
npa workbench health preflight --checks nebius --json
```

**Gate:** both exit 0. Token Factory / Hugging Face / NGC warnings are recorded; a missing `v1.` Token Factory key is expected until START-HERE’s still-to-do is done. The **Nebius** check must pass before any cluster.

### 4 — Cheapest real artifact (still no Isaac GPU)

```bash
npa workbench token-factory verify
```

**Gate:** verify succeeds. This proves Token Factory, not Isaac. Skip only if the team explicitly defers the `v1.` key; do not treat a skip as “GPU ready”.

### 5 — GPU cluster (second yes — not this session)

```bash
npa workbench workflow gpus --cluster <name> --json   # after a cluster exists
npa cluster up --gpu-workload-profile rtx-rendering
```

That profile selects `gpu-rtx6000` and defaults to `1gpu-24vcpu-218gb`. Preemptible is the cheap quota tier; confirm with the human before `--preemptible` because the node can vanish mid-demo.

After up: Isaac bootstrap is `/isaac-sim/python.sh` inside the workbench image (`isaac-lab` skill). Loopback Isaac IPC stays on `127.0.0.1:8226` and is reached with an SSH tunnel. Never publish that port.

### 6 — Stop the spend

```bash
npa cluster down --project <alias> --force
```

Full order (cancel jobs first): `.agents/skills/teardown-and-cost/SKILL.md`.

---

## Out of scope (later C slices)

- `scene/SPEC.md`, `pasture.usda`, `world.usda` composition (C2)
- `orchestrator/` Nemotron JSON loop (C3)
- NVIDIA skills `isaac-sim-headless-deployment` and `usd-composition-architecture` (not in `nebius-physical-ai`)
