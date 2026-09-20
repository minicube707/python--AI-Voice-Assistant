"""
nlp/rules/rules.py
-------------
Default NLP engine: No LLM; rules, regex, and fuzzy matching only.
This is the recommended engine for the target configuration.

Typical resource usage: a few milliseconds of CPU time,
< 5 MB of additional RAM (rapidfuzz is written in C++).
"""

from __future__ import annotations
import json
import re

from nlp.engine import NLPEngine
from nlp.intent import INTENTS


class RulesEngine(NLPEngine):

    def detect_intent(self, text: str) -> str:
        """Detects the user's intent based on predefined patterns.

        The input text is being compared against the patterns
        defined in `INTENTS`. The function returns the first matching intent.
        If no pattern matches, the intent is set to ``"unknown"``.

        Args:
            text (str): The input text used to detect the user's intent.

        Returns:
            str: The detected intent, or ``"unknown"`` if no matching pattern
                is found.
        """
     
        for intent, patterns in INTENTS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return intent

        return "unknown"


        


    