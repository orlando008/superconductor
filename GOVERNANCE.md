# Governance

How decisions get made, how credit is assigned, and how disputes are resolved. This is a
starting point for the pilot phase and is expected to change as the community grows.

## Roles

- **Maintainers** merge PRs, manage labels and the task queue, and own CI and the registry.
  They are listed in `.github/CODEOWNERS`.
- **Domain reviewers** are physicists with expertise in correlated-electron numerics or
  cuprate experiments. They review the protected core (model format, validation, scoring)
  and adjudicate disputes about physics. Becoming one: nomination by a maintainer, plus a
  track record of reviews.
- **Contributors** are anyone who opens an issue or PR, or submits a record. Each
  contribution has one responsible human, identified by GitHub handle.
- **Agents** are AI systems acting for a contributor. Agents are welcome and expected. The
  human operating an agent is accountable for everything it submits. Agents follow
  [AGENTS.md](AGENTS.md).

## What gets reviewed by whom

The scarce resource is expert attention. It goes where automated checks are weakest.

| Area | Merge requirement |
| --- | --- |
| Protected core (`CODEOWNERS`): model format, validation, registry, scoring, CI, governance | Automated checks + a maintainer + a domain reviewer for physics changes |
| Solvers, observables, tests, docs | Automated checks + one maintainer |
| New result records (`results/`) | Automated checks only (`validate --reproduce`, append-only) |

Results become *trusted* through independent reproduction, not through review.

### Importing records for others

CI rejects pull requests that add result records naming anyone other than the PR author.
A maintainer importing records on others' behalf (e.g. migrating results from another
source) applies the `registry-import` label, which skips that check for the PR. Before
merging such a PR, the maintainer must confirm that each named contributor agreed to the
import, and must review the final commit: the label stays on if more commits are pushed.
Branch protection that dismisses stale approvals and requires approval of the most recent
push makes this re-review mandatory.

## Decisions

- Day-to-day decisions: maintainers, by lazy consensus on the issue or PR.
- Changes to conventions or the model format (anything that bumps `SCHEMA_VERSION`): an
  issue labelled `needs-expert`, open for at least 7 days, with explicit approval from a
  domain reviewer.
- Changes to this document: a PR open for at least 14 days, approved by the maintainers.

## Disputes about results

1. A disagreement between methods, or a `disputed` record, gets an issue (Disagreement
   template). It is never resolved by editing records.
2. Contributors try to determine whether it's a bug or physics: extra reproductions, other
   methods, convergence studies.
3. If it isn't resolved, a domain reviewer adjudicates. The outcome is recorded in the issue
   and, where relevant, as a new test or cross-check so it can't regress.

## Prediction challenge integrity (layer 6)

Hidden answers for prediction targets are never stored in this repository. They will be
held by a separate scoring service run by maintainers, with submissions frozen before
answers are revealed. Attempting to obtain hidden answers or tuning against them leads to
disqualification and removal of the contributor's records from scoring.

## Credit

- Every merged contribution is attributed through git history and record `contributor`
  fields. The `status` command and future dashboards will show contributions per person.
- Papers built on the platform should cite it (see `CITATION.cff`). Contributors whose work
  is essential to a specific paper should be offered co-authorship by that paper's authors,
  following normal scientific practice.
- Agent use is credited to the responsible human, with the agent named. Agents aren't
  authors.

## Licensing

- Code: MIT (see `LICENSE`).
- Result records in `results/`: dedicated to the public domain under
  [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (see `results/LICENSE`).
  Anyone may use them for any purpose, with no conditions, so the data can be merged into
  other datasets, benchmarks and training sets without attribution chains across thousands
  of contributors. By submitting a record, a contributor dedicates it under CC0.
- **Citation request (a scientific norm, not a legal condition):** if you use the results,
  please cite the project (see `CITATION.cff`) and, where practical, the record ids you used.
  Contributors still get credit through the `contributor` field of every record and through
  git history.
- Contributors certify that they have the right to contribute their work by adding a
  `Signed-off-by` line to commits ([Developer Certificate of Origin](https://developercertificate.org/)).

## Conduct

Be precise, be generous with credit, and attack claims, not people. Spam, mass
low-quality PRs, or abuse of the task queue by any contributor or their agents leads to
removal of the offending contributions and, if repeated, a ban.
