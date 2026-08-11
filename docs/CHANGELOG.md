# Changelog

What changed in this repository. **One chapter per date, `DD-MM-YYYY`, newest at the
top.** Terse: what changed, and why if it is not obvious. All rules and rule changes are
recorded here.

This file is the road *taken*. The roads *closed* — what was tried and failed — go in
[`AGENTS_MEMORY.md`](AGENTS_MEMORY.md). Keep them separate: mixing them makes both
unreadable.

Conventions:

- **Newest date first.** A reader wants the current state, not the archaeology.
- Dates are `DD-MM-YYYY`. Not ISO, not `Jan 5`. One format, sortable by eye.
- One bullet per change. If the *why* is not obvious from the *what*, add it after an
  em dash.
- Group a date's bullets under `### Added` / `### Changed` / `### Removed` / `### Fixed`
  only once there are enough to need it.

---

## {{DATE}}

- Repository created from
  [andreasgoethals/0.-Template](https://github.com/andreasgoethals/0.-Template).
- `docs/TEMPLATE.md` copied in verbatim and its hash baked into
  `tests/test_template_compliance.py` — a local edit to the template now fails the test
  suite, which is the point.
- `tfm-library/` added as a read-only submodule. Record the pin here whenever it moves,
  because a result depends on the literature it was checked against.
