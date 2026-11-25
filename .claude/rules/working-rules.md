# Working rules

These are Joon's working rules, copied from his global `CLAUDE.md` of 2026-11-21, with this
repository's names in place of the general ones. A
cloud session does not load that file, so the copy here is what such a session reads.
The reasoning behind the rules is in the `code-conventions` skill. Load that skill
before designing a module or contract, refactoring, or writing a change document, and
load the `clear-prose` skill before writing or revising a document, docstring or commit
message.

## Working together

- **IMPORTANT: while a design is under discussion, stay in discussion.** Propose,
  weigh options, refine wording. Edit files once Joon says to go.
- Work a review table to its end. A correction raised mid-round is appended to the
  table rather than jumped to.
- A mistake is a gap in the instructions. Name the line, check or skill that would
  have prevented it, and propose that edit.
- Facts are yours to find, decisions are Joon's: read the file, run the command or
  check the paper rather than asking.
- The review and the merge are Joon's. Claude does not propose skipping a review round
  or merging ahead of one.

## Before and after an edit

- Read what is already red first: the branch change document's *Tests* and *Open
  issues*, and any pinned `xfail`. Then run only the test files covering the modules
  you will touch, at the `dev` profile. The full tier runs once at the end, and its
  counts update that document.
- A failure in that baseline belongs to the branch that owns it. A new failure the
  baseline does not explain belongs to this change, unless a dependency or checker
  version moved, in which case re-check the record.
- A statement about the code cites `file:line` read this session, even when the same
  statement was true an hour ago.
