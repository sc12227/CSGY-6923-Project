import re

TOKEN_PATTERN = re.compile(
    r"""
    \[[^\]]+\]              |  # chords like [CEG]
    \([0-9]+[A-Za-z]+       |  # decorations like (3ABC
    [A-Ga-g][,']*           |  # note letters with octave markers
    [0-9/]*                 |  # durations after notes, empty allowed
    z[0-9/]*                |  # rests: z, z2, z/2
    \|+                     |  # bar lines | ||
    [A-Za-z]+:[^\s]+        |  # headers K:C, M:4/4, Q:1/4=120
    [0-9/]+                 |  # pure numbers (durations)
    [^A-Za-z0-9\s]          |  # any single leftover symbol
    \S+                        # any other non-space token
""",
    re.VERBOSE,
)

def tokenize_abc_line(line: str):
    if line.startswith("%"):
        return []

    tokens = TOKEN_PATTERN.findall(line)

    tokens = [t.strip() for t in tokens if t.strip()]

    return tokens


def tokenize_abc_text(text: str):
    """Tokenize a full ABC file (string)."""
    all_tokens = []

    for line in text.splitlines():
        t = tokenize_abc_line(line)
        if t:
            all_tokens.extend(t)

    return all_tokens


if __name__ == "__main__":
    sample = """
X:1
T:Title
M:4/4
K:C
C2 E2 G2 | [CEG] (3ABC z2 ||
"""
    print(tokenize_abc_text(sample))
