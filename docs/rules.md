# Rules — VERBATIM

Nebius x NVIDIA Global AI Hackathon · https://nebiusglobalaihackathon.devpost.com/
**Argue from this file, never from memory.** Everything below is quoted exactly.

---

## Dates

```
Submission period   "August 26 at 9:00am PDT"  ->  "October 30 at 10:00am PDT"
Judging period      "November 02 at 9:00am PST" -> "December 15 at 12:00pm PST"
Winners announced   "January 11 at 12:00pm PST"
```

Deadline in our timezone: **30 Oct 2026, 17:00 GMT.**
Our internal schedule: feature freeze 27 Oct · video 28 Oct · **submit 29 Oct** (a full day of
buffer — never submit on the day, on a network everyone else is also using).

---

## The eligibility gate

> "create a working software application that runs on either Nebius Token Factory or Nebius AI
> Cloud and uses at least one NVIDIA open source model."

Both halves are mandatory. Note it says **on Nebius** — using NVIDIA's own hosted endpoint at
`build.nvidia.com` would *not* satisfy this.

## Physical AI track

> "embodied and edge agents that sense and act in the real world"

Demo video requirement:

> "For Physical AI Track submissions, should include at least one minute of footage showing the
> physical hardware/robot actually operating, or — if the Project has no physical hardware
> component — the key application modules in action."

**Simulation is explicitly allowed.** Also: a hosted demo URL is **not required** for Physical AI,
unlike the other three tracks.

## Pre-existing code — applies to us

> "Projects must be either newly created by the Entrant or, if the Entrant's Project existed prior
> to the Hackathon Submission Period, must have been significantly updated after the start of the
> Hackathon Submission Period."

> "If your project existed before the Submission Period, include a written explanation of what was
> significantly updated during the Submission Period."

The JS sim predates 26 Aug 2026, so **we must write that explanation**.

## Repository

> "The repository must be public and open source by including an open source license file (such as
> Apache 2.0, MIT, or MPL 2.0)."

---

## Judging criteria — HOW SCORING ACTUALLY WORKS

From the organiser update *"Here's how judging works"*:

> "Submissions first go through a pass/fail check for baseline viability and fit with the theme.
> Everything that passes is then **scored 1–5 on four equally weighted criteria**."

**So: 25% each, 1–5 each, max 20 points.** The four, verbatim from `/rules`:

> **"Technological Implementation:** How well is the project built, and how effectively does it use
> Nebius Token Factory or AI Cloud model(s), and NVIDIA Nemotron or other NVIDIA open source models
> as part of the solution?"

> **"Design:** Does the project deliver a complete, coherent product experience not just a technical
> proof of concept?"

> **"Potential Impact:** Does the project make a credible, specific case for solving a real problem
> for a real audience and does the solution actually address it based on what's demonstrated?"

> **"Quality of the Idea:** Is this a creative, non-obvious use of Nebius Token Factory or AI Cloud
> model(s), and NVIDIA Nemotron or other NVIDIA open source models and does the team show genuine
> understanding of the problem space?"

### ⚠️ The tiebreak order is the real rubric

> "if two or more Submissions are tied, the tied Submission with the highest score in **the first
> applicable criterion listed above** will be considered the higher scoring Submission."

Four criteria scored 1–5 across thousands of entrants **guarantees ties at the top**, and ties break
on **Technological Implementation first**, then Design, then Impact, then Idea. In practice Tech
Implementation is worth more than 25%.

**Consequence for the video:** lead with the Nebius/NVIDIA depth in the **first 30 seconds**.
> "Judges are not required to watch beyond three minutes."

Also note, on testing:
> "Judges are not required to test the Project and may choose to judge based solely on the text
> description, images, and video provided in the Submission."

**The video is the deliverable.** Budget for it accordingly.

---

## Submission checklist

- [ ] A working project using NVIDIA models on Nebius Token Factory or AI Cloud
- [ ] A category (track) — **Physical AI**
- [ ] A project description
- [ ] ~~A working demo URL~~ — **not required for Physical AI**
- [ ] A demo video — **YouTube, ≤ 3 minutes, with audio explaining the Nebius/NVIDIA model usage**
- [ ] A public code repository with an **open source licence file**
- [ ] README with setup instructions
- [ ] Feedback on the platforms/tools used ← *also a $100 prize, see below*
- [ ] Written explanation of what was significantly updated during the submission period

---

## Prizes

| Prize | Amount | Winners |
|---|---|---|
| Grand Prize | $20,000 cash | 1 |
| 2nd Place | $10,000 cash | 1 |
| 3rd Place | $6,000 cash | 1 |
| Track Winners | NVIDIA Jetson Orin Nano | 4 |
| Best Use of Tavily | $3,000 cash | 1 |
| **City Winner Award** | **$500 cash** | **20** |
| Most Valuable Feedback | $100 cash + NVIDIA swag | 10 |

Total "$50,000+ in prizes".

**City Winner — we qualify:**

> "All Eligible Submissions from Entrants attending one of the following participating IRL city
> events (Builders & Brews): Tokyo, Da Nang, Seoul, Kuala Lumpur, Singapore, Taipei, Tel Aviv,
> **London**, Copenhagen, Stockholm, Warsaw, Amsterdam, Berlin, Paris, Mexico City, New York City,
> Toronto, Boston, San Francisco, and Los Angeles."

We attended **Builders & Brews London, 15 Sept 2026**. Attendance + a valid submission is the
qualification. Nearly free money — do not forget to submit.

**Most Valuable Feedback** is also nearly free: the platform-feedback field is a required
submission field anyway. Write it properly rather than dashing it off.

---

## Credits

- **Builders & Brews attendance (London, 15 Sep)** → **$100 Token Factory + $100 Nebius AI Cloud**,
  plus Tavily and Zapier credits. ⭐ **The $100 AI Cloud is the GPU money** — roughly 130 h of L40S
  preemptible, which covers the whole project. Claim it at the event.
- **`NEBIUS-DEVPOST-GLOBAL26`** → $25 Token Factory. Form:
  `https://nebius.com/promo-code?utm_promo_event_code=2026-devpost-global-ai-hack&utm_promo_code_type=Token_Factory&utm_promo_activation_code=NEBIUS-DEVPOST-GLOBAL26`
- **Nebius Builders Program** (`dev.nebius.com/builders`) → another $25 Token Factory + Tavily +
  Academy credits.

⚠️ The two promo codes are **Token Factory (inference) only**, and the **AI Cloud free trial was
suspended 13 July 2026**. Without the event's $100 AI Cloud credit, GPU hours come out of pocket
(L40S preemptible ≈ $0.74/h, RTX PRO 6000 on-demand ≈ $1.80/h). AI Cloud promo codes are applied
separately: console → Billing → "Apply promo code".

---

## Prize strategy — the stacking rule

> "Each Project is eligible for one (1) Overall Award **OR** one (1) Track Award and one (1) Bonus
> Award."

So we cannot win the Grand Prize *and* a bonus. Two paths:

- **Path A (better odds):** Physical AI **Track Award** (Jetson Orin Nano — competing only against
  other Physical AI entries) **+ Best Use of Tavily ($3,000)**.
- **Path B:** Grand Prize ($20,000) alone.

**Path A is the play.** ~5,800 registered, but the gallery is unpublished and comparable NVIDIA
Devpost events drew 21–24 submissions; a four-way track split makes Physical AI very winnable.

**Best Use of Tavily**, verbatim: *"All Eligible Submissions that make a functional, runtime call to
the Tavily API as part of its solution."* One award, low competition, ~a day's work. The winning
pattern is Tavily output **changing the plan the NVIDIA model emits** — e.g. live weather and
dust-forecast data rerouting the cleaning schedule — not a sidebar of links.

**Most Valuable Feedback:** the feedback field is a *required* submission field anyway, so every
submission is auto-eligible. The competition is between people who typed "great platform!" and
people who wrote a real bug report. Score against the three named axes — *completeness, viability,
potential impact*. ⚠️ Unresolved: whether MVF counts as a "Bonus Award" and therefore competes with
Tavily. Worth asking on the forum.

---

## Sim-only: allowed, and it wins

Hard precedent from **NVIDIA's own Cosmos Cookoff** (1,600+ participants, a Physical AI challenge):
**1st and 2nd place were both Isaac Sim.** 2nd place (ResQ-AI) was a **100% simulated drone, no
hardware at all**.

But the organisers of *this* event signalled otherwise, twice, verbatim:
> "🦾 Physical AI: **a scripted demo is table stakes; hardware reacting live to real-world input is
> where this track can really shine.**"
> "🦾 Physical AI: embodied agents, robotics, or IoT with **a real hardware demo**."

**Cheap hedge:** get ≥60 s of *any* real device reacting live to real-world input — a webcam plus an
ESP32 or a Pi satisfies "robotics, IoT, and on-device intelligence" and moves us from the rule's
fallback branch to its primary branch. A previous NVIDIA-sponsored **Grand Prize** winner used an
ESP32, a pulse sensor and a pressure film. Then use the simulation as the *scale* story on Nebius
Serverless Jobs — which is what earns Technological Implementation, the de facto #1 tiebreaker.

Also: **provide a demo URL anyway**, even though Physical AI is exempt. A judge who can click
something scores Design higher, and Design is tiebreak #2.

⚠️ **Date conflict:** `/details/dates` says judging begins 2 Nov; the **rules** say 1–15 Dec. The
rules bind. Either way **everything must stay live until 15 Dec 2026** — keys, endpoints, repo.

---

## Sources

- https://nebiusglobalaihackathon.devpost.com/
- https://nebiusglobalaihackathon.devpost.com/rules
- https://nebiusglobalaihackathon.devpost.com/details/dates
- https://nebiusglobalaihackathon.devpost.com/resources
