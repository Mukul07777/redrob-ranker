# Redrob Hackathon — Intelligent Candidate Ranker

## How to run
```bash
python rank.py --candidates candidates.jsonl --out submission.csv
```

## Approach
Rule-based scorer with 7 weighted components:
- **Skill match (38%)** — core JD skills weighted by proficiency × endorsements × duration
- **Career quality (22%)** — product company bonus, consulting-firm penalty
- **Years of experience (12%)** — sweet spot 6-8 years per JD
- **Availability (15%)** — recency, response rate, interview completion
- **Location (7%)** — Noida/Pune/Delhi NCR/Hyderabad preferred
- **Notice period (4%)** — sub-30 days ideal
- **GitHub activity (2%)** — open source signal

## Honeypot detection
Flags profiles with impossible YoE vs career history, expert skills with zero endorsements/duration, and suspicious endorsement-to-duration ratios.

## Constraints met
- CPU only ✅
- No network during ranking ✅
- Runs in ~60 seconds on 100K candidates ✅
- stdlib only, no pip installs needed ✅