# Security Notes

## Speech Data And Metrics

- Runtime metrics are written to `metrics.jsonl` by default.
- Development defaults are text-on:
  - `metrics.log_transcripts: true`
  - `metrics.log_llm_text: true`
- For shared or production-like environments, disable raw text logging (`false`) unless you have explicit consent and retention controls.

## Operational Guidance

- Keep metrics files out of version control.
- Store metrics in a secured location with appropriate access controls.
- Rotate and delete old metrics logs regularly.
