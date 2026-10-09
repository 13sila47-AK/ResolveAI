# Decision Log

Plain list of non-obvious calls made while building this, and why. Numbered
so the report can reference them. Part 2 will append #10 onward.

1. **Brand: Spotify (SpotifyCares), not Amazon/Apple/Uber.** Amazon and
   Apple have far more volume (170k/107k agent tweets) but span too many
   product categories (retail, devices, cloud, media) to have a clean small
   intent taxonomy. Spotify is a single-product support surface with real
   back-and-forth resolution threads (not just "please call us"), and 43k
   pairs is plenty for both weak-label training and a 200-example golden set.

2. **Banking77 was used as a methodology reference, not a label source.**
   Its domain (banking) doesn't transfer to music streaming. What *did*
   transfer: the idea of a tight, mutually-exclusive intent set with
   explicit decision boundaries between similar-sounding classes, built by
   reading real examples rather than guessing from the brand's marketing
   copy.

3. **Weak labeling (regex/keyword rules) for the 43k-row training set,
   hand labeling only for the 200-row golden set.** Hand-labeling 43k
   tweets isn't feasible; hand-labeling 0 tweets means you can't trust any
   metric. The split is deliberate: weak labels are *only* used to train the
   TF-IDF baseline (and, in Part 2, possibly as few-shot examples) — they
   are never used as ground truth for evaluation.

4. **Pair extraction is single-turn, not full-thread.** Each training/golden
   example is (customer tweet, first direct agent reply), not the full
   multi-turn thread. This is a real limitation — a few golden examples are
   only interpretable with earlier thread context the extraction didn't
   capture (e.g. "No difference ☹️"). Chose this over full-thread
   reconstruction because: (a) the agent's job in this project is to handle
   the *first* incoming message, not referee a whole thread, and (b) thread
   reconstruction on this dataset is genuinely messy (branching replies,
   quote-tweets, multiple customers CC'd) and would have eaten the time
   budget for a marginal gain. Documented as a limitation, not hidden.

5. **Escalation is a policy applied to (intent, content), not its own
   intent class.** The same intent (e.g. `billing_payment_subscription`)
   can be auto-handleable ("how do I apply the student discount" —
   informational) or must escalate ("someone's using my grandmother's card"
   — needs account verification). Modeling escalate as an 10th intent would
   have hidden this and produced a taxonomy that doesn't match how the
   decision actually needs to be made.

6. **Golden set is stratified-then-topped-up, not purely random.** A purely
   random 200-sample would have ~120 chitchat examples and single digits for
   several real support intents — not enough to say anything meaningful
   about per-class performance. The tradeoff (documented explicitly) is that
   the golden set's intent distribution is NOT representative of true
   traffic mix, so it must not be used to estimate volume/prevalence.

7. **Escalation policy is a judgment call, encoded as an explicit written
   rule, applied consistently — not "vibes per example."** The 36.5%
   escalate rate is a direct function of where that rule draws the line
   (e.g., treating all payment/account-lookup cases as escalate=True). A
   business could reasonably draw it stricter (escalate more, safer, more
   human load) or looser (escalate less, cheaper, more risk). This number
   is flagged in the report as one of the "misleading headline number" risks.

8. **Baseline escalate labels come from a DM-mention regex on the
   *historical* agent reply, not from the golden hand labels.** This keeps
   the "simple" baseline honestly weak-label-trained end-to-end (same
   standard as the intent classifier) rather than accidentally peeking at
   golden-set-quality signal for one of its two heads.

9. **Leakage check: golden-set tweet IDs are explicitly excluded from
   baseline training data**, not just "sampled before training" (which
   would still risk accidental overlap if scripts are re-run out of order).
   `tfidf_baseline.py` filters by ID set at train time as a hard guarantee.

10. **Text cleaning strips agent sign-off codes (e.g. "/LS") and URLs, but
    keeps @mentions and emoji.** Sign-offs are agent-identifier noise with
    no semantic content — stripping them measurably reduces sparse,
    meaningless TF-IDF features. @mentions and emoji were kept because they
    sometimes carry signal (e.g. which brand is being addressed, sentiment).
