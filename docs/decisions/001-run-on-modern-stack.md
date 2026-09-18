# Decision 001: Run On A Modern Stack

## Context
- `Exoplanets Classifier.py` is a 2018 Jupyter export and had not been run since. On Python
  3.11.14 / pandas 3.0.5 / scikit-learn 1.9.1 it aborted before reaching any model.
- Three failures blocked the run: `get_ipython()` at line 12 (no kernel outside Jupyter),
  `.any(1)` at line 169 (positional `axis` removed in pandas 2.0), and `target_1` at line 218
  (name never defined; the name in scope is `target`).
- Two further bugs did not abort the run but were wrong: `clean_dataset(df)` at line 172 discarded
  its return value, so the inf filter never reached `df`; and `evaluation(y_true, y_pred)` at
  line 180 ignored its `y_true` parameter and read the global `y_test`.
- `DecisionTreeClassifier` and `RandomForestClassifier` were unseeded, so accuracy moved
  93.18-93.85% and 95.77-96.19% between runs and no reported number was reproducible.
- `exoplanets_2018.csv` is not in the repo, so the run used the NASA Exoplanet Archive
  `cumulative` KOI table fetched 2026-09-17 (9,564 rows x 49 columns, 2026 dispositions).

## Decision
Fix the file in place — 7 changes across 10 lines — rather than carry compatibility shims outside
it, and seed both tree estimators with `random_state=1` to match the split at line 210.

## Reason
- The three breakages are environment and typo bugs, not modelling choices; leaving them behind a
  shim layer means the repo stays unrunnable for anyone who clones it.
- Every change is confined to a single line, so the line numbers cited elsewhere stay stable.
- Seeding costs nothing and is the difference between a number that can be quoted and a number
  that moves every run. `random_state=1` reuses the value already chosen for the split.
- The feature set was deliberately **not** touched. Whether `DispositionScore` (`koi_score`,
  line 67) belongs in the features when the label is built from `koi_pdisposition` (line 118) is a
  modelling decision, and it is left open.

## Consequences
- `Exoplanets Classifier.py`: lines 12, 169, 172, 184-187, 218, 257, 273. Runs with no shims,
  exit 0, 4.2 s.
- The `clean_dataset` fix is a no-op on this data — LogisticRegression and KNN are bit-identical
  before and after — so the frame is unchanged at 7,803 rows x 37 features.
- Reported accuracy, now reproducible run to run:

  | model | line | accuracy | precision | recall | F1 |
  |---|---|---|---|---|---|
  | LogisticRegression | 227 | 82.38% | 80.76% | 86.92% | 83.73% |
  | KNeighborsClassifier | 242 | 80.17% | 79.15% | 84.15% | 81.57% |
  | DecisionTreeClassifier | 257 | 93.27% | 93.50% | 93.61% | 93.55% |
  | RandomForestClassifier | 273 | 96.12% | 97.78% | 94.72% | 96.22% |

- Still open: `DispositionScore` alone scores 94.73% (AUC 0.9710) and is 39.80% of RandomForest
  feature importance; holding it out drops the forest from 96.11% to 87.99% over 8 seeds.
- Not reproducible from a clone until the KOI table fetch and a `pyproject.toml` are committed.
