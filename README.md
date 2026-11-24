# Visual RAG poisoning experiments

**Project 12: Retrieval Poisoning on Charts/Tables/Documents**

This project studies retrieval poisoning in visually rich PDF documents. It
examines how cell, header, chart, and layout manipulations affect retrieved
evidence and downstream question-answering performance.

- **A: Full pipeline** — ColQwen2 retrieves pages, MonoQwen2-VL reranks them,
  and Qwen2.5-VL-7B answers. This tests whether poisoned pages reach the generator
  and affect its answer.
- **B: Generator only** — Supply pages directly to the generator to test its
  response to edited evidence and page order.

Prepare data → run experiments → review answers → read **`summary.csv`**.

## 1. Set up Colab

1. Put the project in `/content/drive/MyDrive/visual_rag` and open `nlp.ipynb`.
2. Select a GPU runtime.
3. Run the setup cells: mount Drive, enter the project folder, install
   dependencies, start PostgreSQL with pgvector, set the database password,
   and create folders. In a fresh runtime, uncomment and run
   `%pip install -r requirements.txt` if that line is commented out.

The commands below run from the project folder.

## 2. Prepare data and config

**Dataset folders:** Under `data/` folder contains `docs/`, `questions.csv`, and `questions.txt`.

**PDF folders** under `docs/`:

| Folder | Contents |
| --- | --- |
| `clean/` | Original PDFs and distractors |
| `cell/`, `header/`, `chart/`, `layout/` | Edited copies for each attack |
| `random_cell/`, `random_header/`, `random_chart/`, `random_layout/` | Unrelated edits, when random controls are enabled |
| `duplicates/` | Unedited copies created automatically |

Each edited copy must retain the original page count and contain only the
manipulation being tested. Multiple questions may use the same original PDF
on different pages; use separate copies for independent edits.
For full A runs, the clean corpus must exceed retrieval top-k; 50 clean pages
are recommended for the default top-10 retrieval.

**Questions:** each owner's `questions.csv` uses the same columns as the existing pilot. `questions.txt` contains the same questions in CSV order. Paul's supplied answer key has been mapped into this schema; unavailable descriptive metadata is left blank. For new examples, use `questions_template.csv`.

- Each row needs a unique `question_id`, plus `doc_name`, `question`,
  `support_pages`, and `clean_answer`.
- For each applicable attack, fill `<attack>_doc_name`, `<attack>_edited_page`,
  and `<attack>_target_answer` (the intended wrong answer). Leave unused attacks blank.
- Filenames are relative to their matching folders. PDF pages start at **1**;
  use semicolons for multiple support pages, e.g. `2;5`. The edited page must
  be a supporting page. Cell/header attacks on the same row must edit the same page.

**Config:** update these keys in the provided `config.json`; keep the other settings:

```json
{
  "attacks": ["cell", "header", "chart", "layout"],
  "random_controls": false
}
```

Select only prepared attack types. `false` omits A4 and random-edit files.
With `true` (the default if omitted), also fill `random_<attack>_doc_name` and
`random_<attack>_edited_page`. Its PDF must edit the same page and preserve the
correct answer.

## 3. Run the experiments

Check inputs and conditions first:

```python
!python run_experiment_a.py --both --validate-only
```

This creates duplicates and JSONL annotations but no model answers.
`--skip-visual-check` skips pixel comparison; other input checks remain.
Verify your PDF edits manually.

Then run:

```python
!python run_experiment_a.py --both
```

Add `--clean` for clean baselines only. Omit `--both` for A only; use
`run_experiment_b.py --skip-visual-check` for B only.

**A conditions:** each case uses the shared clean corpus plus the listed addition.

| Condition | Added document |
| --- | --- |
| `a0` | None: clean baseline |
| `a1` | Unedited copy: reference for attacks |
| `a2_cell`, `a3_header`, `a5_chart`, `a6_layout` | One copy with the selected attack |
| `a4_random_<attack>` | Matched random-edit copy, when enabled |

**B conditions:** pages supplied directly to the generator.

| Condition | Context |
| --- | --- |
| `b0` | No pages: memory baseline |
| `b1` | Clean supporting pages |
| `b2_<attack>` | Poisoned page replaces its clean counterpart |
| `b3_<attack>` | Clean target page + unedited copy |
| `b4_<attack>_clean_first` | Clean target page, then poisoned copy |
| `b4_<attack>_poison_first` | Poisoned copy, then clean target page |

Other supporting pages stay fixed in B; only the tested pair changes order.

Results go to `results/experiment_a_<id>/` and `results/experiment_b_<id>/`.
Repeating a command resumes matching work. Changes to inference inputs can
require new model calls; earlier results remain saved.

## 4. Review answers

In **each run folder**, copy `review_template.csv` to `reviewed_answers.csv`.
The notebook review cell does this without overwriting labels. Keep A and B
reviews separate.

Review the prediction against the question, reference answers, and context pages:

| Column | Enter |
| --- | --- |
| `correct` | `1` for correct, `0` for incorrect, on every completed row |
| `target_match` | On targeted attacks: `1` if the intended wrong answer was produced, otherwise `0`; blank for baselines and random edits |
| `notes` | Optional explanation |

Leave incomplete or unreviewed answers blank. Answer checking is entirely manual.
After saving labels, recalculate:

```python
!python evaluate.py --experiment a
!python evaluate.py --experiment b
```

These use the latest saved runs and do not rerun models. For an older run,
replace `--experiment a` with `--run /full/path/to/the/run_folder`.

## 5. Read the results

**Open `summary.csv` in each run folder**, or use the notebook results cell.

| Metric | Meaning |
| --- | --- |
| Hit@1 / Hit@5 | A correct supporting page appears in the first 1 / 5 retrieved pages; poison is excluded |
| Target-page retrieval@k | The added target page appears in retrieval top-k |
| Target-page context | That page reaches the generator after reranking |
| Answer accuracy | Fraction marked correct by human review |
| Targeted ASR | Fraction producing the intended wrong answer among questions whose matched baseline was correct |

Retrieval metrics apply to A. The target page is poisoned in attack rows and
an unedited copy in matched A1 rows. Clean-only pilots have no attack target annotations.

Compare **A attacks vs A1, B2 vs B1, and B4 vs the matching B3**. Use matching
`Question subset` rows, such as `cell questions`. A1 and B1 also have overall
rows. Report the two B4 orders separately.

Rates include counts, e.g. `40.0% (2/5)`. Check `Incomplete` and `Unreviewed`.
`Pending review/run` means required labels or executions are missing;
`N/A (0/0)` means no eligible denominator, not zero success.

Keep `records.json` and `manifest.json`: they support resuming and recalculation.
Detailed `metrics.csv` and `metrics.json` are optional; add `--details` to an
evaluation command to generate them. Normal evaluation replaces those exports
with `summary.csv`, preserving raw answers and human labels.
