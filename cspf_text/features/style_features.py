from __future__ import annotations

import math
import string
from collections import Counter
from dataclasses import dataclass, field

from ..utils import safe_divide, simple_sentence_split, simple_word_tokenize


FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with",
}


@dataclass
class StyleFeatureExtractor:
    """Stylometric features with NLTK-first, regex-second behavior."""

    use_nltk: bool = True
    _stopwords: set[str] = field(default_factory=lambda: set(FALLBACK_STOPWORDS), init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.use_nltk:
            return
        try:
            import nltk
            from nltk.corpus import stopwords

            resource_paths = {
                "punkt": "tokenizers/punkt",
                "punkt_tab": "tokenizers/punkt_tab",
                "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
                "stopwords": "corpora/stopwords",
            }
            for resource, lookup_path in resource_paths.items():
                try:
                    nltk.data.find(lookup_path)
                except LookupError:
                    try:
                        nltk.download(resource, quiet=True)
                    except Exception:
                        pass
            self._stopwords = set(stopwords.words("english"))
        except Exception:
            self.use_nltk = False

    def _sentence_split(self, text: str) -> list[str]:
        if not self.use_nltk:
            return simple_sentence_split(text)
        try:
            from nltk import sent_tokenize

            return sent_tokenize(text)
        except Exception:
            self.use_nltk = False
            return simple_sentence_split(text)

    def _word_tokenize(self, text: str) -> list[str]:
        if not self.use_nltk:
            return simple_word_tokenize(text)
        try:
            from nltk import word_tokenize

            return word_tokenize(text)
        except Exception:
            self.use_nltk = False
            return simple_word_tokenize(text)

    def _pos_counts(self, tokens: list[str]) -> Counter[str]:
        if not self.use_nltk or not tokens:
            return Counter()
        try:
            from nltk import pos_tag

            return Counter(tag for _, tag in pos_tag(tokens))
        except Exception:
            self.use_nltk = False
            return Counter()

    def transform(self, text: str) -> dict[str, float]:
        sentences = self._sentence_split(text)
        tokens = self._word_tokenize(text)
        alpha_tokens = [token.lower() for token in tokens if any(ch.isalpha() for ch in token)]
        unique_tokens = set(alpha_tokens)
        token_counter = Counter(alpha_tokens)
        pos_counts = self._pos_counts(tokens)

        punctuation_count = sum(1 for ch in text if ch in string.punctuation)
        digit_count = sum(ch.isdigit() for ch in text)
        uppercase_count = sum(ch.isupper() for ch in text)
        char_count = len(text)
        word_count = len(alpha_tokens)
        sentence_count = len(sentences)
        long_word_count = sum(len(token) >= 7 for token in alpha_tokens)
        stopword_count = sum(token in self._stopwords for token in alpha_tokens)
        hapax_count = sum(freq == 1 for freq in token_counter.values())
        avg_sentence_len = safe_divide(word_count, sentence_count)

        noun_ratio = safe_divide(sum(pos_counts[tag] for tag in pos_counts if tag.startswith("NN")), len(tokens))
        verb_ratio = safe_divide(sum(pos_counts[tag] for tag in pos_counts if tag.startswith("VB")), len(tokens))
        adj_ratio = safe_divide(sum(pos_counts[tag] for tag in pos_counts if tag.startswith("JJ")), len(tokens))
        adv_ratio = safe_divide(sum(pos_counts[tag] for tag in pos_counts if tag.startswith("RB")), len(tokens))

        return {
            "style_char_count": float(char_count),
            "style_word_count": float(word_count),
            "style_sentence_count": float(sentence_count),
            "style_avg_sentence_len": avg_sentence_len,
            "style_avg_word_len": safe_divide(sum(len(token) for token in alpha_tokens), word_count),
            "style_lexical_diversity": safe_divide(len(unique_tokens), word_count),
            "style_hapax_ratio": safe_divide(hapax_count, word_count),
            "style_stopword_ratio": safe_divide(stopword_count, word_count),
            "style_long_word_ratio": safe_divide(long_word_count, word_count),
            "style_punctuation_ratio": safe_divide(punctuation_count, max(char_count, 1)),
            "style_digit_ratio": safe_divide(digit_count, max(char_count, 1)),
            "style_uppercase_ratio": safe_divide(uppercase_count, max(char_count, 1)),
            "style_comma_per_sentence": safe_divide(text.count(","), sentence_count),
            "style_exclamation_per_sentence": safe_divide(text.count("!"), sentence_count),
            "style_question_per_sentence": safe_divide(text.count("?"), sentence_count),
            "style_noun_ratio": noun_ratio,
            "style_verb_ratio": verb_ratio,
            "style_adj_ratio": adj_ratio,
            "style_adv_ratio": adv_ratio,
            "style_sentence_len_std": (
                math.sqrt(
                    safe_divide(
                        sum((len(self._word_tokenize(sentence)) - avg_sentence_len) ** 2 for sentence in sentences),
                        sentence_count,
                    )
                )
                if sentence_count
                else 0.0
            ),
        }
