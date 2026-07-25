"""Convert digit sequences to spoken Kazakh cardinal words.

ASR services differ in how they render numbers: Whisper-family models tend to
emit digits ("5, 6, 7", "40 минут"), while the benchmark references spell numbers
out ("бес алты жеті", "қырық минут"). Left as-is this shows up as WER errors that
are formatting, not recognition. Normalizing hypotheses (and references, a no-op
since they contain no digits) to spelled-out Kazakh words removes that artifact.

Kazakh cardinals are fully regular, so a compact generator covers any integer up
to the billions — more than enough for spontaneous speech.
"""

import re

_ONES = ["", "бір", "екі", "үш", "төрт", "бес", "алты", "жеті", "сегіз", "тоғыз"]
_TENS = ["", "он", "жиырма", "отыз", "қырық", "елу", "алпыс", "жетпіс", "сексен",
         "тоқсан"]

_DIGITS_RE = re.compile(r"\d+")


def _below_1000(n: int) -> list[str]:
    parts: list[str] = []
    h, rem = divmod(n, 100)
    if h:
        # 100 -> "жүз", 200 -> "екі жүз"
        parts.append("жүз" if h == 1 else f"{_ONES[h]} жүз")
    t, o = divmod(rem, 10)
    if t:
        parts.append(_TENS[t])
    if o:
        parts.append(_ONES[o])
    return parts


# (scale value, scale word, whether a leading "бір" is used for exactly one unit)
_SCALES = [
    (1_000_000_000, "миллиард", True),
    (1_000_000, "миллион", True),
    (1_000, "мың", False),  # 1000 is "мың", not "бір мың"
]


def spell_integer_kk(n: int) -> str:
    """Spell a non-negative integer as Kazakh cardinal words."""
    if n == 0:
        return "нөл"
    parts: list[str] = []
    remaining = n
    for value, word, use_one in _SCALES:
        if remaining >= value:
            count, remaining = divmod(remaining, value)
            group = _below_1000(count)
            if count == 1 and not use_one:
                parts.append(word)
            else:
                parts.extend(group + [word])
    parts.extend(_below_1000(remaining))
    return " ".join(parts)


def digits_to_words_kk(text: str) -> str:
    """Replace every run of digits in ``text`` with its Kazakh spelling."""
    return _DIGITS_RE.sub(lambda m: spell_integer_kk(int(m.group())), text)
