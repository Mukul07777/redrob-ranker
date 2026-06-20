#!/usr/bin/env python3
"""
Redrob Hackathon — Candidate Ranker
Usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv
Constraints: CPU only, <=16GB RAM, <=5 min, no network during ranking.
"""

import argparse
import csv
import gzip
import json
import math
import re
from datetime import date, datetime
from pathlib import Path

# ── Reference date (competition date) ────────────────────────────────────────
TODAY = date(2026, 6, 19)

# ── Consulting firm blacklist (JD explicitly disqualifies careers entirely here) ──
CONSULTING_BLACKLIST = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "l&t infotech",
    "ltimindtree", "mindtree",  # Mindtree merged into LTIMindtree (IT services)
}

# ── Location scoring ──────────────────────────────────────────────────────────
PREFERRED_CITIES = {
    "noida", "pune", "delhi", "new delhi", "gurugram", "gurgaon",
    "hyderabad", "mumbai", "bangalore", "bengaluru", "faridabad", "ncr",
}

# ── Core required skills (JD section: "Things you absolutely need") ───────────
# Weighted by how central each is to the role
CORE_SKILLS = {
    # Embeddings / retrieval
    "sentence-transformers": 3.0, "sentence transformers": 3.0,
    "embeddings": 3.0, "text embeddings": 3.0, "dense retrieval": 3.0,
    "bge": 2.5, "e5": 2.5, "openai embeddings": 2.0,
    "semantic search": 2.5, "semantic similarity": 2.0,
    # Vector databases / hybrid search
    "pinecone": 2.5, "weaviate": 2.5, "qdrant": 2.5, "milvus": 2.5,
    "faiss": 2.5, "opensearch": 2.0, "elasticsearch": 2.0,
    "vector database": 2.5, "vector db": 2.5, "vector search": 2.5,
    "hybrid search": 2.5, "bm25": 2.0, "ann": 2.0,
    # Ranking / retrieval systems
    "information retrieval": 3.0, "ranking": 2.5, "reranking": 2.5,
    "re-ranking": 2.5, "learning to rank": 2.5, "ltr": 2.5,
    "recommendation system": 2.0, "recommender": 2.0,
    "search": 2.0, "retrieval": 2.5, "rag": 2.5,
    "retrieval augmented generation": 2.5,
    # Evaluation frameworks
    "ndcg": 2.5, "mrr": 2.5, "map": 2.0, "a/b testing": 2.0,
    "ab testing": 2.0, "offline evaluation": 2.0, "online evaluation": 2.0,
    "eval framework": 2.0, "evaluation framework": 2.0,
    # LLMs (nice to have but present)
    "llm": 2.0, "large language model": 2.0, "fine-tuning": 1.5,
    "fine tuning": 1.5, "lora": 1.5, "qlora": 1.5, "peft": 1.5,
    "fine-tuning llms": 1.5, "nlp": 2.0,
    # Python
    "python": 2.0,
    # ML production
    "mlops": 1.5, "model deployment": 1.5, "production ml": 2.0,
    "inference optimization": 1.5, "xgboost": 1.5,
}

# ── Negative skill signals (CV/speech/robotics-only → JD says wrong domain) ──
NEGATIVE_SKILLS = {
    "computer vision", "cv", "image classification", "object detection",
    "speech recognition", "asr", "tts", "text to speech",
    "robotics", "ros", "slam", "gans", "generative adversarial",
    "photoshop", "figma", "illustrator", "brand design",
}

# Titles that are clearly non-ML (used to penalize title mismatch)
NON_ML_TITLES = {
    "marketing manager", "operations manager", "customer support",
    "hr manager", "sales manager", "content writer", "business analyst",
    "brand manager", "graphic designer", "project manager",
    "account manager", "product manager",  # PM is borderline but not AI eng
    "mechanical engineer", "civil engineer",
}

ML_TITLES = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "data scientist", "research scientist", "applied scientist",
    "nlp engineer", "search engineer", "ranking engineer",
    "backend engineer",  # borderline — don't penalize
    "software engineer",  # borderline
    "senior engineer", "staff engineer",
    "data engineer",  # adjacent, not penalized
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def days_since(d):
    if d is None:
        return 9999
    return (TODAY - d).days


def clamp(val, lo=0.0, hi=1.0):
    return max(lo, min(hi, val))


def normalize(val, lo, hi):
    """Linear normalize val into [0,1] given expected [lo,hi]."""
    if hi == lo:
        return 0.5
    return clamp((val - lo) / (hi - lo))


def company_is_consulting(name: str) -> bool:
    name_l = name.lower().strip()
    for firm in CONSULTING_BLACKLIST:
        if firm in name_l:
            return True
    return False


def skill_name_lower(name: str) -> str:
    return name.lower().strip()


# ─────────────────────────────────────────────────────────────────────────────
# Honeypot detection
# ─────────────────────────────────────────────────────────────────────────────

def is_honeypot(candidate: dict) -> bool:
    """
    Detect subtly impossible profiles.
    Returns True if we're confident this is a honeypot.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # 1. YoE inconsistency: claimed YoE >> sum of career durations
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    claimed_yoe = profile.get("years_of_experience", 0)
    if claimed_yoe > 0 and total_career_months > 0:
        implied_yoe = total_career_months / 12.0
        # If claimed is >3x implied (and claimed is substantial), suspicious
        if claimed_yoe > 5 and implied_yoe < claimed_yoe / 3:
            return True

    # 2. Expert in many skills but zero endorsements AND zero duration for all
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and s.get("endorsements", 0) == 0
        and s.get("duration_months", 0) == 0
    )
    if expert_zero >= 4:
        return True

    # 3. Current company start date implies less time than claimed YoE by huge margin
    for job in career:
        if job.get("is_current"):
            start = parse_date(job.get("start_date"))
            if start:
                months_at_current = (TODAY - start).days / 30.4
                # If they claim 10+ years but started current job <6 months ago
                # and have only one job — impossible
                if claimed_yoe >= 10 and months_at_current < 6 and len(career) == 1:
                    return True

    # 4. Skills with future-dated implied tenure (endorsements >> duration)
    # e.g. 80 endorsements on a skill with 1 month duration — suspicious
    suspicious_skills = sum(
        1 for s in skills
        if s.get("endorsements", 0) > 50 and s.get("duration_months", 0) <= 2
    )
    if suspicious_skills >= 3:
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Scoring components
# ─────────────────────────────────────────────────────────────────────────────

def score_skills(candidate: dict) -> float:
    """
    Core skill match. Weighted by proficiency × endorsement trust × duration.
    Not a keyword counter — we weight by evidence quality.
    Returns 0-1.
    """
    skills = candidate.get("skills", [])
    summary = (candidate.get("profile", {}).get("summary", "") or "").lower()
    headline = (candidate.get("profile", {}).get("headline", "") or "").lower()

    # Proficiency multiplier
    prof_mult = {"beginner": 0.3, "intermediate": 0.6, "advanced": 1.0, "expert": 1.0}

    raw_score = 0.0
    max_possible = 0.0
    negative_score = 0.0

    for skill in skills:
        sname = skill_name_lower(skill.get("name", ""))
        prof = prof_mult.get(skill.get("proficiency", "beginner"), 0.3)
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)

        # Endorsement trust: log scale, caps at ~50 endorsements
        endorse_trust = math.log1p(endorsements) / math.log1p(50)
        endorse_trust = clamp(endorse_trust)

        # Duration trust: caps at 48 months
        duration_trust = clamp(duration / 48.0)

        # Combined evidence quality
        evidence = 0.5 * endorse_trust + 0.5 * duration_trust

        # If zero endorsements AND zero duration: low trust even if "advanced"
        if endorsements == 0 and duration == 0:
            evidence = 0.05

        # Is this a core JD skill?
        weight = CORE_SKILLS.get(sname, 0.0)
        if weight == 0.0:
            # Partial matches (e.g. "elasticsearch" substring)
            for core_kw, w in CORE_SKILLS.items():
                if core_kw in sname or sname in core_kw:
                    weight = w * 0.8
                    break

        if weight > 0:
            raw_score += weight * prof * (0.4 + 0.6 * evidence)
            max_possible += weight

        # Negative: CV/speech/robotics-only skills
        if sname in NEGATIVE_SKILLS:
            negative_score += 0.15 * prof

    # Also check summary/headline for core concept mentions (catches Tier 5 candidates)
    text = summary + " " + headline
    text_bonus = 0.0
    for kw, w in CORE_SKILLS.items():
        if kw in text:
            # Small bonus for describing the work even if not listed as skill
            text_bonus += w * 0.15
    text_bonus = min(text_bonus, 2.0)  # cap

    raw_score += text_bonus

    # Normalize against a reasonable max (top candidate might hit ~15-20 raw)
    normalized = clamp(raw_score / 18.0)

    # Apply negative penalty (heavy CV/speech profiles)
    skill_names = {skill_name_lower(s.get("name", "")) for s in skills}
    negative_ratio = len(skill_names & NEGATIVE_SKILLS) / max(len(skill_names), 1)
    if negative_ratio > 0.5:
        # Majority of skills are in wrong domain
        normalized *= 0.3

    normalized = clamp(normalized - negative_score * 0.1)
    return normalized


def score_career(candidate: dict) -> float:
    """
    Career quality: product company experience, title relevance, consulting penalty.
    Returns 0-1.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    title = (profile.get("current_title", "") or "").lower()
    company = (profile.get("current_company", "") or "").lower()

    score = 0.5  # baseline

    # Title relevance
    if any(t in title for t in ["ml", "machine learning", "ai ", "nlp",
                                  "search", "ranking", "retrieval", "data scientist",
                                  "applied scientist", "research scientist"]):
        score += 0.3
    elif any(t in title for t in NON_ML_TITLES):
        score -= 0.25

    # Career history quality
    product_company_months = 0
    consulting_only = True
    total_months = 0

    for job in career:
        co = (job.get("company", "") or "").lower()
        industry = (job.get("industry", "") or "").lower()
        duration = job.get("duration_months", 0)
        total_months += duration

        is_consulting = company_is_consulting(co)
        if not is_consulting:
            consulting_only = False
            # Proxy for product company: not IT services / consulting
            if "it services" not in industry and "consulting" not in industry:
                product_company_months += duration

        # Check job description for ML/retrieval work
        desc = (job.get("description", "") or "").lower()
        ml_signals = ["embedding", "retrieval", "ranking", "vector", "search",
                      "recommendation", "nlp", "llm", "fine-tun", "rag",
                      "a/b test", "ndcg", "mrr", "machine learning"]
        for sig in ml_signals:
            if sig in desc:
                score += 0.04
                break  # one bonus per job

    # Consulting-only career: heavy penalty (JD explicitly says this)
    if consulting_only and len(career) > 0:
        score -= 0.35

    # Product company experience bonus
    if product_company_months > 24:
        score += 0.15
    elif product_company_months > 12:
        score += 0.07

    return clamp(score)


def score_experience_years(candidate: dict) -> float:
    """
    YoE fit: sweet spot 5-9 years, ideal 6-8.
    Returns 0-1.
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0

    if 6 <= yoe <= 8:
        return 1.0
    elif 5 <= yoe < 6:
        return 0.85
    elif 8 < yoe <= 9:
        return 0.85
    elif 4 <= yoe < 5:
        return 0.65
    elif 9 < yoe <= 12:
        return 0.70
    elif yoe > 12:
        return 0.55  # overqualified / may be title-chaser
    elif 3 <= yoe < 4:
        return 0.45
    else:
        return 0.2


def score_availability(candidate: dict) -> float:
    """
    Behavioral availability: recency, response rate, engagement.
    A great-on-paper candidate who's unreachable is worthless to a recruiter.
    Returns 0-1 (used as multiplier).
    """
    sig = candidate.get("redrob_signals", {})

    # Recency of last login
    last_active = parse_date(sig.get("last_active_date"))
    days_inactive = days_since(last_active)

    if days_inactive <= 7:
        recency = 1.0
    elif days_inactive <= 30:
        recency = 0.9
    elif days_inactive <= 60:
        recency = 0.75
    elif days_inactive <= 90:
        recency = 0.55
    elif days_inactive <= 180:
        recency = 0.35
    else:
        recency = 0.15  # >6 months inactive: probably not available

    # Open to work
    otw = 1.1 if sig.get("open_to_work_flag") else 0.85

    # Response rate
    rr = sig.get("recruiter_response_rate", 0.5) or 0.5
    # Normalize: 0.8+ is great, 0.2 is bad
    rr_score = normalize(rr, 0.1, 0.9)

    # Interview completion rate
    icr = sig.get("interview_completion_rate", 0.5) or 0.5
    icr_score = normalize(icr, 0.3, 1.0)

    availability = recency * otw * (0.6 * rr_score + 0.4 * icr_score)
    return clamp(availability)


def score_location(candidate: dict) -> float:
    """
    Location fit. Prefers Noida/Pune/Delhi NCR/Hyderabad.
    Returns 0.6-1.0.
    """
    sig = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    location = (profile.get("location", "") or "").lower()
    country = (profile.get("country", "") or "").lower()
    relocate = sig.get("willing_to_relocate", False)

    # Outside India: case-by-case, no visa sponsorship
    if country not in ("india", "in", ""):
        return 0.65 if relocate else 0.55

    # Check preferred cities
    for city in PREFERRED_CITIES:
        if city in location:
            return 1.0

    # India but not preferred city
    if relocate:
        return 0.85
    else:
        return 0.70


def score_notice_period(candidate: dict) -> float:
    """
    Notice period: <30 days ideal, 30+ penalized, 90+ heavily penalized.
    Returns 0.5-1.0.
    """
    sig = candidate.get("redrob_signals", {})
    notice = sig.get("notice_period_days", 60) or 60

    if notice <= 0:
        return 1.0
    elif notice <= 30:
        return 1.0
    elif notice <= 60:
        return 0.85
    elif notice <= 90:
        return 0.72
    elif notice <= 120:
        return 0.60
    else:
        return 0.50


def score_github_activity(candidate: dict) -> float:
    """GitHub activity as a proxy for open-source / active coding signal."""
    sig = candidate.get("redrob_signals", {})
    gh = sig.get("github_activity_score", -1)
    if gh == -1 or gh is None:
        return 0.4  # no github — neutral-negative
    return 0.4 + 0.6 * clamp(gh / 80.0)


# ─────────────────────────────────────────────────────────────────────────────
# Final scoring
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "skill":        0.38,
    "career":       0.22,
    "yoe":          0.12,
    "availability": 0.15,
    "location":     0.07,
    "notice":       0.04,
    "github":       0.02,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1"


def score_candidate(candidate: dict) -> float:
    sk = score_skills(candidate)
    ca = score_career(candidate)
    yoe = score_experience_years(candidate)
    av = score_availability(candidate)
    loc = score_location(candidate)
    not_ = score_notice_period(candidate)
    gh = score_github_activity(candidate)

    composite = (
        WEIGHTS["skill"]        * sk +
        WEIGHTS["career"]       * ca +
        WEIGHTS["yoe"]          * yoe +
        WEIGHTS["availability"] * av +
        WEIGHTS["location"]     * loc +
        WEIGHTS["notice"]       * not_ +
        WEIGHTS["github"]       * gh
    )
    return round(clamp(composite), 6)


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning generation (factual, specific, honest)
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning(candidate: dict, rank: int) -> str:
    profile = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    notice = sig.get("notice_period_days", "?")
    rr = sig.get("recruiter_response_rate", 0)
    last_active = parse_date(sig.get("last_active_date"))
    days_ago = days_since(last_active)
    relocate = sig.get("willing_to_relocate", False)

    # Top matching skills for this candidate
    top_skills = []
    for s in skills:
        sname = s.get("name", "")
        if skill_name_lower(sname) in CORE_SKILLS:
            top_skills.append(sname)
    top_skills = top_skills[:3]

    # Concerns
    concerns = []
    if days_ago > 180:
        concerns.append(f"inactive {days_ago // 30}+ months")
    if notice and notice > 60:
        concerns.append(f"{notice}-day notice")
    if rr < 0.3:
        concerns.append(f"low recruiter response rate ({rr:.0%})")
    if company_is_consulting(company):
        concerns.append("consulting-firm background")

    skill_str = ", ".join(top_skills) if top_skills else "adjacent skills"
    concern_str = ("; concern: " + ", ".join(concerns)) if concerns else ""

    reasoning = (
        f"{title} at {company} | {yoe:.1f} yrs | {location}"
        f"{' (willing to relocate)' if relocate else ''}"
        f" | skills: {skill_str}"
        f" | response rate {rr:.0%}, last active {days_ago}d ago"
        f"{concern_str}."
    )
    return reasoning[:500]  # keep it within reasonable bounds


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(path: str):
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    mode = "rt"
    candidates = []
    with opener(p, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates.")

    print("Scoring...")
    results = []
    honeypots_flagged = 0

    for c in candidates:
        cid = c.get("candidate_id", "")
        if not cid:
            continue

        if is_honeypot(c):
            score = 0.001  # push to bottom, don't discard (they need a rank if in top 100)
            honeypots_flagged += 1
        else:
            score = score_candidate(c)

        results.append((cid, score, c))

    print(f"Flagged {honeypots_flagged} honeypots.")

    # Sort by score descending; tie-break by candidate_id ascending
    results.sort(key=lambda x: (-x[1], x[0]))

    top100 = results[:100]

    print("Writing output...")
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score, candidate) in enumerate(top100, start=1):
            reasoning = generate_reasoning(candidate, rank)
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])

    print(f"Done. Written to {out_path}")
    print(f"Top 5 candidates:")
    for i, (cid, score, c) in enumerate(top100[:5], 1):
        title = c.get("profile", {}).get("current_title", "?")
        yoe = c.get("profile", {}).get("years_of_experience", "?")
        print(f"  {i}. {cid} | {title} | {yoe} yrs | score={score:.4f}")


if __name__ == "__main__":
    main()
