# C1 — GPU first run

**Owner: C** · unblocks Dimitris and B.  
**Status:** Nebius CLI is installed and authenticated. **`npa configure` has a project stanza.** **No GPU cluster. No spend.** Isaac Sim does not run on laptops.

This is the Solar Sheep walkthrough of Nebius `first-run-setup`. Do not skip a gate because a later step looks more interesting. Do not copy tenant/project ids into new tracked files; they belong in `~/.npa/` after configure.

Sibling workbench (already cloned, `npa` on PATH): `../nebius-physical-ai`.

---

## Evidence (this machine, 15 Sep 2026)

| Gate | State | Evidence |
|---|---|---|
| Python 3.12 / Terraform 1.x | **VERIFIED** | `python3` 3.12.2 · Terraform 1.16.2 |
| `npa` CLI | **VERIFIED** | `npa 0.1.0` at `~/.local/bin/npa` |
| Nebius CLI on PATH | **VERIFIED** | `~/.nebius/bin/nebius` · `nebius version` → `0.12.277` (darwin/arm64) |
| `nebius iam whoami` | **VERIFIED** | GitHub federation user, **own** tenant `ACTIVE` (not Dimitris’s START-HERE tenant) |
| `npa workbench health preflight --checks nebius` | **VERIFIED** | `PASS` — “Default Nebius CLI profile is authenticated.” |
| `npa configure --show` project stanza | **VERIFIED** | alias `eu-north1`, same project id as START-HERE, region **`eu-north1`**, S3 unset |
| START-HERE GPU listing | **VERIFIED** 15 Sep on **us-central1** / Dimitris tenant | Listing ≠ this npa stanza. **HITL before any cluster.** |
| Default health preflight | **VERIFIED** 15 Sep 20:37 GMT+1 | `ok: true`. PASS: HF, Token Factory (23 models). WARN: NGC unset, S3 unset |
| Token Factory `verify` | **VERIFIED** | `authenticated: true`, 23 models. Includes `nvidia/Nemotron-3_5-Lightning` (default text) |
| Isaac Sim / cluster | **NOT STARTED** | Do not provision until C confirms tenant/region **and** spend |

This Mac is Darwin arm64. Isaac Sim Docker is native-Linux x86_64. The box is a Nebius Linux GPU node; this laptop only runs `npa` / `nebius` over the API.

New shells need `export PATH="$HOME/.nebius/bin:$PATH"` (or `exec -l $SHELL`) so `nebius` stays on PATH.

---

## HITL — do not provision on the wrong account

START-HERE’s quota probe used Dimitris’s tenant and **us-central1**. This laptop’s `npa` stanza is **eu-north1** on the GitHub-federation user’s own tenant. Same project *id string* as START-HERE does not prove it is the same billing/quota pool.

**Stop.** C confirms one of:

1. Use **this** npa project (eu-north1) and re-check `gpu-rtx6000` availability there, or  
2. Reconfigure npa to Dimitris’s START-HERE tenant/region after being added as collaborator.

Do not run `npa cluster up` until that is written in gitignored `docs/access.md`.

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

### 0 — Nebius CLI binary — done on this Mac

```bash
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
export PATH="$HOME/.nebius/bin:$PATH"
nebius version
npa workbench health preflight --checks nebius --json
```

**Gate:** `nebius` on PATH and the `nebius` health check `PASS`.

### 1 — `npa` already installed — done

```bash
npa --version
npa --help
```

### 2 — Configure — stanza exists; confirm it is the intended project

```bash
npa configure --show
```

If the alias/region/tenant are wrong, fix with interactive configure (**`--no-provision`** so it does not create a bucket):

```bash
npa configure --no-provision
```

**Gate:** `npa configure --show` prints the **intended** project. Record the check in `docs/access.md` (gitignored). Never commit `~/.npa/credentials.yaml`.

### 3 — Preflight (no spend)

```bash
npa workbench health preflight --json
npa workbench health preflight --checks nebius --json
```

**Gate:** Nebius check exits 0. Token Factory / Hugging Face / NGC warnings are recorded; they do not authorize a cluster.

### 4 — Cheapest real artifact (still no Isaac GPU)

```bash
npa workbench token-factory verify
```

**Gate:** verify succeeds. This proves Token Factory, not Isaac.

### 5 — GPU cluster (second yes — not this session)

Only after HITL on tenant/region. `npa` looks for Terraform at `./deploy/cluster` **relative to the current working directory**. This repo does not ship that tree — it lives in the sibling workbench. Running the short form from `solar-sheep-copy` fails with `Cannot find deploy/cluster`.

```bash
export PATH="$HOME/.nebius/bin:$PATH"
unset NEBIUS_IAM_TOKEN NPA_NEBIUS_IAM_TOKEN
npa cluster up \
  --project eu-north1 \
  --gpu-workload-profile rtx-rendering \
  --terraform-dir ../nebius-physical-ai/deploy/cluster
```

Equivalent: `cd ../nebius-physical-ai` then the short form. `npa configure --show` currently has **S3 unset**; Terraform remote state needs a bucket, so the next failure after a missing `deploy/cluster` is often storage. Fix with `npa configure` (or `npa storage`) before a long apply.

That profile selects `gpu-rtx6000` and defaults to `1gpu-24vcpu-218gb`. Preemptible is the cheap quota tier; confirm before `--preemptible` because the node can vanish mid-demo.

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
