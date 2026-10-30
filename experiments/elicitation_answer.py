"""Read an explicitly stated final integer without requiring LaTeX boxing.

This extension was specified after inspecting calibration output only. Original
raw responses and their strict-parser scores remain intact. No task labels are
available to this module.
"""

import re

from elicitation_task import parse_answer as parse_boxed_answer


FINAL = re.compile(r"\b(?:final\s+(?:integer|answer|state|value|result)|answer)\b", re.IGNORECASE)
INTEGER = re.compile(r"(?<![\w.,])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?!\w|\.\d|,\d)")


def parse_answer(text):
    boxed = parse_boxed_answer(text)
    if boxed is not None: return boxed
    tagged = [line for line in text.splitlines() if FINAL.search(line)]
    if not tagged: return None
    suffix = FINAL.split(tagged[-1])[-1]
    numbers = INTEGER.findall(suffix)
    if len(numbers) == 1: return int(numbers[0].replace(",", ""))
    return None
