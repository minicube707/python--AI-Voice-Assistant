"""
nlp/sentiment/sentiment.py
-------------------------
Alternative NLP engine based on a classification model (zero-shot)
rather than rules/regex. Adheres to EXACTLY the same contract as
RulesEngine (nlp/engine.py): the rest of the application (commands/,
player/) sees no difference.

Important terminology note:
A standard "sentiment analysis" model (positive/negative/neutral) cannot
categorize your intents (play_music, exit, ai_agent...)—it is not
the right tool. What is needed here is a "zero-shot" classification
model (based on NLI): you provide the text plus a list of
arbitrary labels (your intents), and it returns a score for each label
without having been specifically trained on them. This module handles that.

Selected model: cmarkea/distilcamembert-base-nli
- DistilCamemBERT base (~68M parameters, French), MIT license
- ~270 MB in RAM/disk -> significantly heavier than rules.py (~0),
but nowhere near an LLM. CPU-only; viable on an i3-1115G4. 
- Indicative latency: ~100-400 ms per command on CPU (vs <5 ms
for rules). Best reserved for when the regex dictionary
in intent.py becomes difficult to maintain.

Installation:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
(the --index-url flag avoids downloading the CUDA version of torch,
which is unnecessary and bulky on a machine without a dedicated GPU)
"""

from __future__ import annotations
import logging

from transformers import pipeline

from nlp.engine import NLPEngine
from nlp.intent import INTENT_DESCRIPTIONS

logger = logging.getLogger("nlp.sentiment_engine")


class ZeroShotIntentEngine(NLPEngine):

    def __init__(
        self,
        model_name: str = "cmarkea/distilcamembert-base-nli",
        hypothesis_template: str = "Cette phrase correspond à l'intention suivante : {}.",
        catalog_path: str | None = None,
    ):

        super().__init__(catalog_path=catalog_path)
        logger.info("Loading model zero-shot (%s)...", model_name)

        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1,  # explicitly force CPU (no dedicated GPU available)
        )
        # The candidate labels are derived directly from intent.py:
        # a single place to maintain for both engines (rules / zero-shot).
        self.candidate_labels = list(INTENT_DESCRIPTIONS.values())
        self.hypothesis_template = hypothesis_template
        logger.info("Modèle chargé. Labels : %s", self.candidate_labels)


    def detect_intent(self, text: str) -> str:

        result = self.classifier(
            text,
            candidate_labels=self.candidate_labels,
            hypothesis_template=self.hypothesis_template,
        )

        best_intent = result["labels"][0]
        best_score = result["scores"][0]

        if best_score < 0.50:
            return "unknown"

        return best_intent
