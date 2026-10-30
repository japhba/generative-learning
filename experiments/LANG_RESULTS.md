# LANG confirmatory results (7 October 2026)

Protocol: `LANG_PROTOCOL.md`. Data: `results/task_search/lang_main` (29 of 40 episodes usable). Figure:
`reports/lang-confirmatory.png`. Six ~8B models write worked solutions in their own language (English, Japanese,
Chinese) to language-neutral alien-unit problems. Practice is all correct, on identical problems; the reader solves new
problems in its own language.

Contrasts are on the double-centred reader × source matrix, i.e. with reader skill and general source quality removed.
Each cell gives the estimate, the 95% CI and the two-sided p, over 29 episodes.

| | N = 1 | N = 2 | N = 4 |
|---|---|---|---|
| **Same-language peer − other language** (primary at N = 1) | **+4.1 pp [+0.9, +6.8], p = 0.012** | +5.3 [+3.3, +7.4] | +4.0 [+2.3, +5.6] |
| Own − same-language peer | −2.3 [−5.2, +0.9] | −5.5 [−8.6, −2.6] | −6.0 [−8.3, −3.7] |
| Recognition: same-language peer − other language | +11.4 [+6.8, +15.8] | +7.5 [+2.6, +12.2] | +10.5 [+6.8, +14.2] |
| Application given recognition (interaction scale) | +2.3 [−1.9, +5.9] | +5.6 [+2.5, +8.7] | +4.1 [+2.5, +5.8] |
| Uplift over no practice: own / same / other | +36 / +39 / +35 | +41 / +47 / +41 | +44 / +50 / +46 |

- **Language matters.** A worked solution from a model of the reader's language helps it more than its general quality
  predicts, by about 4–5 pp. The gap does not close with more examples: the trend C₁ − C₄ is +0.1 pp [−2.7, +2.8].
  Other-language solutions still help a lot.
- **Self does not.** The reader's own solutions help *less* than a same-language peer's, by 5–6 pp at N = 2 and 4. The
  best source is a different model writing in the reader's language.
- **Recognition vs application.** Other-language solutions are recognised less often (46% vs 57% at N = 1). Even
  among recognised cases, applying them works worse at N = 2 and 4 (+4–6 pp for the same-language peer, with source
  quality removed). This is partial support for "recognised but applied worse".
- **By reader:** the effect is carried by English and Japanese readers (Llama +12 to +15, OLMo +8, Swallows +2 to +7).
  The Chinese readers show none: GLM about 0, InternLM negative.
