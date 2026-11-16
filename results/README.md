# Results registry

Crowd-sourced results, stored as append-only JSON records.

```
results/
  submissions/<id>.json                      one result for one model
  reproductions/<submission_id>/<id>.json    an independent re-run of a submission
```

## Rules

- **Append-only.** Never edit or delete a record. CI rejects PRs that do. To correct a
  mistake, submit a new record and explain it in `notes`.
- **Content-addressed.** A record's `id` is a hash of its content. Editing a record breaks the
  id, and validation catches it.
- **Status is computed, never declared.** A submission is
  - `unverified` until an independent contributor reproduces it,
  - `reproduced` once at least one different contributor's reproduction agrees,
  - `disputed` if any independent reproduction disagrees.
  Agreeing reproductions that used a *different solver* are counted separately, because
  they're stronger evidence.
- **A responsible human.** `contributor.handle` is a GitHub username. Agents identify
  themselves in `contributor.agent`.
- **You can only add records in your own name.** CI fails a pull request if any record it
  adds names a contributor other than the PR author (case-insensitive). Without this, anyone
  could fake an "independent" reproduction. Maintainers importing records for others in bulk
  apply the `registry-import` label, which only people with triage access can do.

## Workflow

```bash
superconductor submit examples/hubbard_4x2_hole_doped.json --solver ed --handle <you> --agent <agent>
```

```bash
superconductor status
```

```bash
superconductor reproduce <submission_id> --handle <you> --agent <agent>
```

```bash
superconductor validate --reproduce
```

`validate` checks schemas, ids, file locations, provenance consistency, independence of
reproductions, and the sanity checks from `superconductor.validation`. With `--reproduce`,
it also re-runs cheap submissions (small ED sectors and the exact limits), which catches
results that are physically plausible but fabricated. CI runs it on every PR.

## Scale

Storing records in git is fine for the pilot phase (thousands of records). Beyond that,
the same record format moves to a dedicated append-only store with this directory as a
mirror. Records are content-addressed, so no ids change in the move.
