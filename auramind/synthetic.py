"""Synthetic dialogue generator, clinical safety guardrails, quality judge, and semantic deduplicator."""

from typing import List, Dict, Tuple, Optional, Set
import re
import math
import random
import hashlib
from collections import Counter, defaultdict

# Domain emotions and scenario components
DOMAINS = {
    "ACADEMIC": ["anxious", "overwhelmed", "discouraged", "ashamed", "exhausted"],
    "CAREER": ["undervalued", "uncertain", "overwhelmed", "self-doubting", "burned out"],
    "RELATIONSHIPS": ["lonely", "misunderstood", "disconnected", "hurt", "uncertain"],
    "FAMILY": ["guilty", "unheard", "conflicted", "burdened", "frustrated"],
    "LIFE_TRANSITIONS": ["disoriented", "apprehensive", "vulnerable", "uncertain", "unmoored"],
    "PERSONAL_GROWTH": ["self-critical", "stagnant", "impatient", "confused", "discouraged"],
}

ACADEMIC_ENTITIES = [
    "my exam",
    "my final project",
    "my coursework",
    "my upcoming presentation",
]

CAREER_ENTITIES = [
    "my new job",
    "my team",
    "my career transition",
    "my workload",
]

GENERAL_ENTITIES = [
    "my relationship",
    "my family situation",
    "this life transition",
    "my personal goals",
]

OPENERS = [
    "I have been feeling {emotion} about {entity} {timeframe}.",
    "Lately I keep thinking about {entity}, and I feel really {emotion}.",
    "I am struggling with {entity} {timeframe}, and it has left me feeling {emotion}.",
    "Every time I think about {entity}, I get this strong feeling of being {emotion}.",
]

VALIDATIONS = [
    "It sounds like {entity} has been carrying a lot of emotional weight for you. It makes sense that you would feel {emotion}.",
    "I hear how much pressure you have been carrying around {entity}. Thank you for putting that experience into words.",
    "That sounds genuinely difficult. Having so much uncertainty around {entity} can leave a person feeling emotionally worn down.",
    "I can hear how important {entity} is to you, and how exhausting it has been to sit with these feelings.",
]

FOLLOWUPS = [
    "What part of this feels hardest to sit with right now?",
    "Which part of the situation is taking up the most space in your mind?",
    "What would feeling a little more supported look like for you today?",
    "Have you noticed a particular moment when these feelings become strongest?",
]

USER_TURNS = [
    "I keep comparing myself with other people, and then I feel like I am falling behind.",
    "I know I cannot control everything, but my mind keeps jumping to the worst possible outcome.",
    "I have been trying to stay productive, but sometimes I just need permission to slow down.",
    "I think I have been much harder on myself than I would ever be on someone I care about.",
]

RESPONSES = [
    "That comparison sounds exhausting. You do not have to solve the whole situation at once; it may help to notice what you are feeling without treating the feeling as a verdict about your worth.",
    "It sounds like your mind is trying to protect you by preparing for every possible outcome. We can make room for that fear without assuming the worst-case scenario will happen.",
    "It is understandable to want a little room to breathe. Taking things one manageable step at a time can be kinder than demanding certainty from yourself.",
    "That self-criticism sounds heavy. If it helps, you might ask what you would say to someone you cared about who was facing exactly the same situation.",
]

CLOSINGS = [
    "You do not need to have everything figured out in this moment.",
    "For now, it may be enough to give yourself a little patience and notice what you need.",
    "We can stay with what is happening right now rather than rushing toward a perfect answer.",
    "Your feelings are worth taking seriously, even when you are still figuring out what they mean.",
]

TIMEFRAMES = [
    "tonight",
    "this week",
    "before my next deadline",
    "lately",
    "over the last few days",
]


# Safety guardrail patterns (blocking unauthorized medical/clinical diagnosing)
UNSAFE_PATTERNS = [
    r"\byou (have|are suffering from|definitely have)\b",
    r"\bi diagnose you\b",
    r"\byou are diagnosed with\b",
    r"\byour symptoms prove\b",
    r"\byour heartbeat is definitely caused by\b",
    r"\byour blood pressure is definitely\b",
    r"\bi am a (doctor|psychiatrist|psychologist)\b",
    r"\bas your doctor\b",
    r"\bthis is definitely a psychiatric\b",
]

UNSAFE_REGEX = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]


def safety_filter(text: str) -> Tuple[bool, Optional[str]]:
    """Checks if text contains unauthorized clinical/medical diagnosing statements."""
    for pattern in UNSAFE_REGEX:
        if pattern.search(text):
            return False, pattern.pattern
    return True, None


def validate_structure(text: str) -> bool:
    """Verifies that dialogue contains balanced user/counselor turns and an EOS token."""
    required = ["<|user|>", "<|counselor|>", "<|eos|>"]
    if not all(token in text for token in required):
        return False
    if text.count("<|user|>") < 1 or text.count("<|counselor|>") < 1:
        return False
    return True


# Transparent 5-Dimension Quality Judge
EMPATHY_MARKERS = [
    "sounds", "hear", "understandable", "difficult", "hard",
    "pressure", "feel", "carrying", "exhausting", "thank you",
]

RELEVANCE_MARKERS = [
    "situation", "right now", "today", "this", "part", "what",
]

DIRECTIVE_PATTERNS = [
    r"\byou must\b",
    r"\byou need to\b",
    r"\bjust do\b",
    r"\byou should definitely\b",
    r"\bthe answer is\b",
]

MEDICAL_PATTERNS = [
    r"\bdiagnos",
    r"\bdisorder\b",
    r"\bmedication\b",
    r"\bprescription\b",
    r"\bpsychiatric\b",
    r"\bmedical condition\b",
]


def extract_roles(text: str) -> Tuple[List[str], List[str]]:
    pieces = re.split(r"(<\|user\|>|<\|counselor\|>)", text)
    current = None
    users = []
    counselors = []
    for piece in pieces:
        if piece == "<|user|>":
            current = "user"
        elif piece == "<|counselor|>":
            current = "counselor"
        elif current == "user" and piece.strip():
            users.append(piece.strip())
        elif current == "counselor" and piece.strip():
            counselors.append(piece.strip())
    return users, counselors


def count_hits(text: str, markers: List[str]) -> int:
    lower = text.lower()
    return sum(marker in lower for marker in markers)


def score_dialogue(text: str) -> Dict[str, float]:
    users, counselors = extract_roles(text)
    counselor_text = " ".join(counselors)
    all_text = text.lower()

    # 1. Empathy dimension (1.0 to 5.0)
    empathy_hits = count_hits(counselor_text, EMPATHY_MARKERS)
    empathy = min(5.0, 3.0 + 0.4 * min(empathy_hits, 5))

    # 2. Relevance dimension (1.0 to 5.0)
    rel_hits = count_hits(counselor_text, RELEVANCE_MARKERS)
    relevance = min(5.0, 3.5 + 0.3 * min(rel_hits, 5))

    # 3. Non-directive stance (penalize bossiness)
    directive_hits = sum(len(re.findall(p, all_text)) for p in DIRECTIVE_PATTERNS)
    non_directive = max(1.0, 5.0 - 1.5 * directive_hits)

    # 4. Clinical boundary / safety score
    med_hits = sum(len(re.findall(p, all_text)) for p in MEDICAL_PATTERNS)
    safety_score = max(1.0, 5.0 - 2.0 * med_hits)

    # 5. Length appropriateness
    words = len(text.split())
    if 40 <= words <= 280:
        length_score = 5.0
    elif 20 <= words < 40 or 280 < words <= 350:
        length_score = 4.0
    else:
        length_score = 2.0

    total = round(empathy + relevance + non_directive + safety_score + length_score, 2)
    return {
        "empathy": empathy,
        "relevance": relevance,
        "non_directive": non_directive,
        "safety": safety_score,
        "length": length_score,
        "total": total,
    }


class SemanticDeduplicator:
    """Exact SHA-256 and feature-based cosine deduplicator."""

    def __init__(self, threshold: float = 0.88):
        self.threshold = threshold
        self.exact_hashes: Set[str] = set()
        self.vectors: List[Tuple[Counter, float]] = []
        self.term_index: Dict[str, List[int]] = defaultdict(list)
        self.exact_duplicates = 0
        self.semantic_duplicates = 0

    @staticmethod
    def features(text: str) -> Tuple[Counter, float]:
        lower = text.lower()
        words = re.findall(r"[a-z0-9']+", lower)
        word_features = [f"w:{w}" for w in words if len(w) > 2]
        char_features = ["c:" + lower[i : i + 4] for i in range(max(0, len(lower) - 3))]
        counts = Counter(word_features + char_features)
        norm = math.sqrt(sum(v * v for v in counts.values()))
        return counts, norm

    def accept(self, text: str) -> bool:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in self.exact_hashes:
            self.exact_duplicates += 1
            return False

        vector, norm = self.features(text)
        if norm == 0:
            return False

        candidate_ids = set()
        for term in vector:
            ids = self.term_index.get(term, [])
            candidate_ids.update(ids[-8:])
            if len(candidate_ids) >= 40:
                break

        for cand_id in candidate_ids:
            old_vec, old_norm = self.vectors[cand_id]
            shared = set(vector) & set(old_vec)
            if not shared:
                continue
            dot = sum(vector[k] * old_vec[k] for k in shared)
            similarity = dot / (norm * old_norm + 1e-9)
            if similarity >= self.threshold:
                self.semantic_duplicates += 1
                return False

        idx = len(self.vectors)
        self.vectors.append((vector, norm))
        self.exact_hashes.add(digest)
        for term in vector:
            self.term_index[term].append(idx)
        return True


def build_synthetic_dialogue(index: int) -> str:
    """Generates a structured synthetic multi-turn empathetic dialogue."""
    domain = random.choice(list(DOMAINS))
    emotion = random.choice(DOMAINS[domain])

    if domain == "ACADEMIC":
        entity = random.choice(ACADEMIC_ENTITIES)
    elif domain == "CAREER":
        entity = random.choice(CAREER_ENTITIES)
    else:
        entity = random.choice(GENERAL_ENTITIES)

    timeframe = random.choice(TIMEFRAMES)
    difficulty = ((index - 1) % 10) + 1

    opening = random.choice(OPENERS).format(emotion=emotion, entity=entity, timeframe=timeframe)
    validation = random.choice(VALIDATIONS).format(emotion=emotion, entity=entity)
    followup = random.choice(FOLLOWUPS)
    user2 = random.choice(USER_TURNS)
    response2 = random.choice(RESPONSES)
    closing = random.choice(CLOSINGS)

    if difficulty <= 2:
        turns = [
            f"<|user|> {opening}",
            f"<|counselor|> {validation} {followup}",
        ]
    elif difficulty <= 5:
        turns = [
            f"<|user|> {opening}",
            f"<|counselor|> {validation}",
            f"<|user|> {user2}",
            f"<|counselor|> {response2} {followup}",
        ]
    else:
        turns = [
            f"<|user|> {opening}",
            f"<|counselor|> {validation}",
            f"<|user|> {user2}",
            f"<|counselor|> {response2}",
            f"<|user|> I think I need to stop expecting myself to resolve everything immediately.",
            f"<|counselor|> {closing}",
        ]

    return "\n".join(turns) + " <|eos|>"
