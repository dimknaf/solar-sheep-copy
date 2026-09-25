# Third-party notices — vendored agent skills

The skill directories under `.claude/skills/` are copied unmodified from the
upstream repositories below, at the pinned commits. `skills-lock.json` at the
repo root lists each skill with its source path, commit, licence and SHA-256.

| Source | Ref / commit | Licence | Skills |
|---|---|---|---|
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | v3.0.0-EA · ae37b028ea415c91ea2bc32609efcd759ed2b974 | BSD-3-Clause (text below) | `isaaclab-*` (11) |
| [isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim) | v6.1.0 · 7c206f75bdadd9e05fc457f19863ca4c3f0cb693 | Apache-2.0 | urdf-mjcf-to-usd-conversion, physics-simulation, usd-articulation, navigation-primitives, isaac-sim-headless-deployment, isaac-sim-troubleshooting, isaac-sim-workflow, isaac-sim-validator, isaac-sim-rendering, isaac-camera, isaac-sim-robot-navigation |
| [nebius/nebius-physical-ai](https://github.com/nebius/nebius-physical-ai) | bf4788a0a94ce2c2e842f54c5dc39932795d6cc5 | Apache-2.0 | token-factory, teardown-and-cost, third-party-eula-preflight, gpu-selection, health-preflight, vm-nebius-auth, protect-nebius-infra-details |

Apache-2.0 text: https://www.apache.org/licenses/LICENSE-2.0

Proprietary NVIDIA skills (NVIDIA-Omniverse/ovrtx, ovstage, kit-cae,
kit-app-template) are **not** redistributable and must never be committed here;
`.gitignore` guards against it.

The Isaac Lab skills link to docs with relative paths (`../../../docs/...`).
Those resolve inside a checkout of isaac-sim/IsaacLab at v3.0.0-EA, not in this
repo.

## isaac-sim/IsaacLab — BSD-3-Clause

```
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).

All rights reserved.

SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
