# Assessment Rule Reference

## Scope
This document defines the rule-based assessment flow.
All runtime rules must be loaded from JSON files only. Do not hardcode JSON content in this README.

## Source Files
- `role-skill-framework.json`
- `skill-references.json`

## Responsibilities
### `role-skill-framework.json`
- Canonical role aliases and role normalization.
- Role-to-skill mapping by career level (`fresher`, `junior`, `middle`, `senior`).
- Allowed skill boundary for each role.

### `skill-references.json`
- Skill semantic descriptions per level.
- Level-to-SFIA mapping and skill metadata used in scoring/evaluation.
- Option text source for assessment answers.

## Runtime Flow
1. Normalize input role using aliases in `role-skill-framework.json`.
2. Resolve level key (`fresher|junior|middle|senior`).
3. Load allowed skills for the normalized role+level from `role-skill-framework.json`.
4. Prioritize skills by signal order:
- `freeText`
- `techstack`
- `domain`
- remaining role skills
5. Generate 15 questions total:
- `phaseA`: 10 core skill questions
- `phaseB`: 5 practical scenario questions
6. Build options from semantic level descriptions in `skill-references.json`.
7. Evaluate submitted answers with rule-based scoring:
- selected level score
- semantic matching score against skill level description
- blended final score
8. Return summary by phase and overall skill breakdown.

## API Contract
### Generate
- Endpoint: `POST /api/generate-assessment`
- Output: `contextQuestion`, `phaseA`, `phaseB`

### Evaluate
- Endpoint: `POST /api/evaluate-assessment`
- Input: `SurveyResponsesDto`
- Output: `SurveySummaryResultDto`
  - `summaryText`
  - `summaryObject` (phase summary + overall + skill scores + gap)

## Required Rules
- Skills must be selected only from the normalized role in `role-skill-framework.json`.
- Skill options and level semantics must come from `skill-references.json`.
- No category-inference fallback for role-to-skill discovery.
- Question count must always be 15 (10 core + 5 practical).

## Maintenance Notes
- To change role coverage or aliases, edit `role-skill-framework.json`.
- To change semantic descriptions or SFIA mapping, edit `skill-references.json`.
- Keep this README as specification only; do not embed large JSON examples.
