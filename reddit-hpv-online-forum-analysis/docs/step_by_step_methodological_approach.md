**Step-by-Step Methodological Approach**

1. Retrieve Reddit threads and preserve the original post plus associated comment tree per thread.
2. Apply a cervical-screening relevance filter so the analytic corpus is restricted to abnormal Pap / cervical HPV narratives.
3. Segment the corpus into sentence units while retaining thread identifiers for leakage-safe splitting.
4. Define a reaction-oriented label space for stigma, distress, and coping rather than relying on baseline clinical-result labels alone.
5. Construct weak positive seeds from theory-guided lexical cues and construct stronger negative seeds from:
   - explicit anti-patterns within the same label,
   - hard negatives drawn from sentences that match other labels but not the target label,
   - neutral negatives drawn from unlabeled sentences when needed for class balance.
6. Train one classifier per label rather than a single shared multilabel head. This reduces cross-label bleed and makes thresholding label-specific.
7. Represent sentences using local transformer embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and fit balanced logistic classifiers on silver-labeled sentence sets.
8. Tune thresholds on held-out silver calibration sets at the thread level, then apply lexical-semantic gating so a sentence must either:
   - contain an explicit target cue, or
   - exceed the learned threshold by a meaningful confidence margin.
9. Aggregate predictions back to sentence-level and thread-level prevalence summaries; rule-based comparison can be added only when a matched reference run exists for the same corpus.
10. Use the resulting annotation workbook and active-learning queue to create a dual-coded gold-standard set for the actual large-scale study.

**What Changed Relative to the Earlier Modern Pipeline**

- Abstentions are now kept as missing rather than as pseudo-0.5 labels.
- Snorkel smoothing is not used when conflict is minimal; direct vote-based seeds are retained.
- Negatives are actively mined instead of assuming unlabeled examples behave as neutral training data.
- Calibration is performed separately for each label.
- Final prediction requires lexical-semantic agreement or very high model confidence.

**Current V2 Pilot Output Pattern**

Top V2 sentence-level labels: procedure_anxiety (6.28%), cancer_fear (5.92%), lifestyle_change (3.46%), supplement_use (3.41%), relational_stigma (3.33%), information_seeking (2.69%).

**Interpretation**

This V2 pipeline is intended to move the pilot toward more realistic prevalence estimates while preserving a modern, scalable architecture. It is still not a replacement for a gold-standard adjudicated training set, but it is methodologically stronger than the earlier degenerate weak-supervision-plus-transformer run because it explicitly addresses sparse coverage, missing negatives, and threshold collapse.
