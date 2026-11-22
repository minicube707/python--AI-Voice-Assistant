

from __future__ import annotations
import json
import re

from nlp.engine import NLPEngine
from nlp.neural.classifier import NeuralClassifier

class NeuralEngine(NLPEngine):

    def detect_intent(self, text: str) -> str:
        
        classifier = NeuralClassifier()
        return classifier.predict(text)

