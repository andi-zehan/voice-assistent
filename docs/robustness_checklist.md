# Robustness Checklist

## Goal
Validate Leonardo voice assistant reliability across main flows and hard corner cases before shipping changes.

## Preconditions
- Hardware: stable mic + speaker/headphones setup
- Environment: `LEO_OPENROUTER_API_KEY` set and model files available
- Build: latest `main`, clean working tree, dependencies installed

## Baseline Validation
1. Run tests: `python3 -m pytest -q`
2. Start assistant: `python3 main.py`
3. Verify basic path: wake -> ask -> response -> follow-up -> passive

## Main Use-Case Scenarios
1. Standard question/answer cycle (10+ turns)
2. Follow-up without wake word within window
3. Long utterance near `max_utterance_s`
4. Language switch (EN -> DE and DE -> EN)
5. Barge-in while assistant is speaking (with headphones)

## Corner Cases
1. Silence after wake (expect listening timeout and passive transition)
2. Background noise bursts near VAD threshold
3. Very short utterances (single words)
4. Empty/low-confidence STT transcript rejection path
5. LLM empty response path
6. Temporary network loss during LLM request
7. Slow LLM response near timeout boundary
8. TTS failure and recovery path
9. Audio callback pressure (observe dropped frame metric)
10. Rapid repeated wake words and interrupted turns
11. Utterance metadata sample mismatch: small mismatch accepted, large mismatch rejected
12. Client reconnect while sending wake/utterance/barge-in events (verify buffered outbound replay)

## Fault Injection (Manual)
1. STT failure: force exception in `WhisperSTT.transcribe` and verify follow-up recovery
2. LLM failure: block outbound network and verify error earcon + follow-up recovery
3. TTS failure: break selected TTS voice/model and verify pipeline error handling
4. Metrics failure: set metrics path to unwritable location and verify assistant keeps running
5. Trigger protocol errors: send malformed utterance metadata and verify stable error code response

## Soak Test
1. Reset or rotate metrics file
2. Run a long session (example 30 minutes):
   `python3 scripts/soak_test.py --duration-s 1800 --command "python3 main.py"`
3. Target pass criteria (adjust per environment):
   - `pipeline_errors == 0`
   - `interaction count >= 20`
   - acceptable `p95` latency under expected network conditions
   - no sustained growth in dropped frames

## Exit Criteria
- All baseline + main scenarios pass
- No unhandled exceptions in console
- Soak run passes thresholds
- Any failures have issue tickets with reproduction notes
