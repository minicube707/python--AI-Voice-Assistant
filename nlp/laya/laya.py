"""
nlp/laya/laya.py
----------------
NLP engine based on the Laya model (convaiinnovations/laya), a non-autoregressive
"decision" model: it takes text (state) + a typed question as input and returns
a probability for each option in a single pass (~30-40 ms on GPU, ~200-450 ms
on CPU). It never generates text, so there is nothing to parse.

Adheres to EXACTLY the same contract as RulesEngine / ZeroShotIntentEngine:
`detect_intent(text) -> str` returns the intent key ("play_music", "exit",
...) or "unknown".

Intents and their descriptions are sourced from intent.py (INTENT_DESCRIPTIONS),
meaning there is only one place to maintain them for all three engines.

Installation:
pip install laya
(if loading hangs and TensorFlow is installed: USE_TF=0)

Checkpoints (only one is downloaded, depending on `subfolder`):
- "multilingual"     : mmBERT-base, 322M params, 100+ languages <- default (French)
- None               : ModernBERT-large, English only
- "typed-decisions"  : version fine-tuned on 4 specific workflows
"""

from __future__ import annotations
import logging
import re

import laya

from nlp.engine import NLPEngine
from nlp.intent import INTENT_DESCRIPTIONS

from reply.reply import REPLY

logger = logging.getLogger("nlp.laya_engine")

UNKNOWN_KEY = "unknown"


class LayaEngine(NLPEngine):

    def __init__(
        self,
        repo: str = "convaiinnovations/laya",
        subfolder: str | None = "multilingual",
        instructions: str = "Quelle est l'intention de l'utilisateur ?",
        min_confidence: float = 0.50,
        catalog_path: str | None = None,
    ):
        super().__init__(catalog_path=catalog_path)
        logger.info("Chargement du modèle Laya (%s / %s)...", repo, subfolder)

        # Charge un seul checkpoint (pas de Router : le français est fixé
        # d'avance, inutile de payer la détection de langue ni un 2e modèle).
        if subfolder:
            self.agent = laya.load(repo, subfolder=subfolder)
        else:
            self.agent = laya.load(repo)

        self.min_confidence = min_confidence

        # Une seule question "choice" : chaque option = une intention.
        # Clé = nom de l'intention, valeur = description (depuis intent.py).
        # On ajoute une option "unknown" pour laisser le modèle refuser.
        criteria = dict(INTENT_DESCRIPTIONS)
        criteria[UNKNOWN_KEY] = "aucune des intentions ci-dessus, phrase hors sujet"

        self.questions = {
            "intent": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        }
        logger.info("Modèle chargé. Intentions : %s", list(criteria))


    def detect_intent(self, text: str) -> str:
        """Détecte l'intention de l'utilisateur avec Laya.

        Args:
            text (str): La phrase de l'utilisateur.

        Returns:
            str: La clé d'intention (ex. ``"play_music"``), ou ``"unknown"``
                si le modèle choisit "unknown" ou si sa confiance est
                inférieure à ``min_confidence``.
        """

        #Check before the reply sentence
        for x in REPLY:
            if re.search(rf"\b{x['quote']}\b", text):
                return 'reply'

        try:
            result = self.agent.predict(text, self.questions)
        except Exception:
            logger.exception("Erreur Laya sur : %r", text)
            return UNKNOWN_KEY

        answer = result["answers"]["intent"]
        intent = answer["choice"]
        confidence = self._confidence(answer)

        logger.debug("Laya -> %s (%.2f) pour %r", intent, confidence, text)

        if intent not in INTENT_DESCRIPTIONS:  # inclut "unknown"
            return UNKNOWN_KEY
        if confidence < self.min_confidence:
            return UNKNOWN_KEY

        return intent


    @staticmethod
    def _confidence(answer: dict) -> float:
        """Extrait la confiance de la réponse, quel que soit le format exact.

        Si aucune confiance n'est exposée, renvoie 1.0 (pas de filtrage).
        """
        if "confidence" in answer:
            return float(answer["confidence"])

        probs = answer.get("probabilities") or answer.get("probs")
        if isinstance(probs, dict) and probs:
            return float(max(probs.values()))

        return 1.0