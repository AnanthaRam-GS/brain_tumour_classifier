# Phase 2 walkthrough notebooks

Faculty-demo notebooks: one per architecture, each showing the full
pipeline end to end and able to run the official final test live during a
demo. The notebook files (`<model>_walkthrough.ipynb`) are generated --
never hand-edited -- from `make_walkthrough.py`, which is the single
source of truth for their content.

## One-time setup

From `phase2/`, with the project's `.venv` active:

```bash
pip install -e ".[notebook]"
python -m ipykernel install --user --name btdl-phase2 --display-name "Python (btdl phase2)"
```

This registers the `btdl-phase2` Jupyter kernel (editable install of the
`btdl` package, so the notebook always imports the checked-out code).

## Demo-day steps

1. Copy the notebook you need into `phase2/notebooks/_live/` (gitignored
   -- this is where a notebook actually gets *executed*, so running it
   never dirties the tracked copy or `test_access_log.csv` by accident):

   ```bash
   mkdir -p phase2/notebooks/_live
   cp phase2/notebooks/efficientnet_b0_walkthrough.ipynb phase2/notebooks/_live/
   cd phase2 && jupyter lab notebooks/_live/efficientnet_b0_walkthrough.ipynb
   ```

2. Select the **Python (btdl phase2)** kernel if it isn't already
   selected.

3. Practise beforehand with the default settings (`TEST_MODE =
   "rehearsal"`, `CONFIRM_FINAL_TEST = False`) -- section 9 then runs the
   identical final-test pipeline on the VALIDATION split only, and never
   writes `test_access_log.csv`.

4. During the actual demo, in section 1's settings cell, set:

   ```python
   TEST_MODE = "final"
   CONFIRM_FINAL_TEST = True
   ```

   and re-run from that cell onward. Section 9 will then run the real,
   gated `final_test` for each final seed (docs/DECISIONS.md D18 --
   requires this model's frozen registration to already be committed),
   export each result, and print the exact `git add`/`git commit`
   commands for the log and results.

5. After the demo, commit `phase2/artifacts/test_access_log.csv` and
   `phase2/artifacts/results/<model>/` from the REAL working tree (not
   from inside `_live/`) -- i.e. run those `git` commands from
   `phase2/notebooks/<model>_walkthrough.ipynb`'s real counterpart, the
   tracked repo, not the gitignored live copy.

## How teammates generate their own

Each teammate who owns an architecture generates (or regenerates, after a
change to `make_walkthrough.py`) their own notebook with:

```bash
cd phase2
python notebooks/make_walkthrough.py --model <name>
```

This overwrites `phase2/notebooks/<name>_walkthrough.ipynb` with outputs
cleared. Commit the regenerated file as usual -- never hand-edit the
`.ipynb` directly, since the next regeneration would silently discard any
manual edits.
