# Text Feature Formulas

This document summarizes the text-side feature design currently implemented in `CSPF-Net`.

## 1. Overall Feature Vector

For a target sentence `s`, left context `c_L`, right context `c_R`, and merged context

```text
c = c_L ⊕ c_R
```

the final sentence-level feature vector is

```text
x = [phi_target(s), phi_left(c_L), phi_right(c_R), phi_context(c), Delta(s, c)]
```

where

```text
phi(t) = [phi_style(t), phi_prob(t), phi_cohesion(t)]
Delta(s, c) = phi(s) - phi(c)
```

So the model uses:

- target sentence features
- left-context features
- right-context features
- merged-context features
- target-context difference features

## 2. Stylometric Features

For a text `t`, let:

- `N_char` = number of characters
- `N_word` = number of alphabetic tokens
- `N_sent` = number of sentences
- `w_i` = the `i`-th word
- `V` = set of unique words
- `N_hapax` = number of words appearing exactly once
- `N_stopword` = number of stopwords
- `N_longword` = number of words with length >= 7
- `N_punct` = number of punctuation characters
- `N_digit` = number of digits
- `N_upper` = number of uppercase letters

The style feature vector is

```text
phi_style(t) = [
  N_char,
  N_word,
  N_sent,
  N_word / N_sent,
  (sum_i |w_i|) / N_word,
  |V| / N_word,
  N_hapax / N_word,
  N_stopword / N_word,
  N_longword / N_word,
  N_punct / N_char,
  N_digit / N_char,
  N_upper / N_char,
  N_comma / N_sent,
  N_exclamation / N_sent,
  N_question / N_sent,
  r_noun,
  r_verb,
  r_adj,
  r_adv,
  sigma_sent_len
]
```

where:

- `r_noun` = noun count ratio
- `r_verb` = verb count ratio
- `r_adj` = adjective count ratio
- `r_adv` = adverb count ratio
- `sigma_sent_len` = standard deviation of sentence lengths

Implemented feature names:

- `style_char_count`
- `style_word_count`
- `style_sentence_count`
- `style_avg_sentence_len`
- `style_avg_word_len`
- `style_lexical_diversity`
- `style_hapax_ratio`
- `style_stopword_ratio`
- `style_long_word_ratio`
- `style_punctuation_ratio`
- `style_digit_ratio`
- `style_uppercase_ratio`
- `style_comma_per_sentence`
- `style_exclamation_per_sentence`
- `style_question_per_sentence`
- `style_noun_ratio`
- `style_verb_ratio`
- `style_adj_ratio`
- `style_adv_ratio`
- `style_sentence_len_std`

## 3. Probabilistic Features

Given a token sequence `x_1, ..., x_n`, GPT-2 is used to compute token-level negative log-likelihood:

```text
NLL_i = -log p(x_i | x_<i)
```

Then the probabilistic feature vector is

```text
phi_prob(t) = [
  mean(NLL),
  exp(mean(NLL)),
  std(NLL),
  max(NLL),
  min(NLL),
  NLL_last - NLL_first
]
```

Interpretation:

- lower average NLL means the text is more predictable to the language model
- perplexity is an exponential transform of the average NLL
- larger NLL variance may indicate unstable token predictability

Implemented feature names:

- `prob_avg_neg_log_likelihood`
- `prob_perplexity`
- `prob_token_nll_std`
- `prob_token_nll_max`
- `prob_token_nll_min`
- `prob_first_last_nll_gap`

## 4. Cohesion Features

The cohesion branch applies repeated random token deletion. Let `t^(k)` be the corrupted text in round `k`.

Semantic similarity is estimated as

```text
sim^(k) = cosine(TFIDF(t), TFIDF(t^(k)))
```

and semantic drift is

```text
d^(k) = 1 - sim^(k)
```

Then the cohesion feature vector is

```text
phi_cohesion(t) = [
  mean(d),
  max(d),
  min(d),
  std(d),
  mean(rho),
  mean(d) / N_token
]
```

where `rho` is the deletion ratio in each corruption round.

Implemented feature names:

- `cohesion_avg_semantic_drift`
- `cohesion_max_semantic_drift`
- `cohesion_min_semantic_drift`
- `cohesion_std_semantic_drift`
- `cohesion_avg_deletion_ratio`
- `cohesion_length_normalized_drift`

## 5. Context-Aware Expansion

Each base feature group is computed on four text scopes:

- target sentence
- left context
- right context
- merged context

The final feature vector therefore contains prefixed variants such as:

- `target_style_*`
- `left_style_*`
- `right_style_*`
- `context_style_*`
- `target_prob_*`
- `left_prob_*`
- `right_prob_*`
- `context_prob_*`
- `target_cohesion_*`
- `left_cohesion_*`
- `right_cohesion_*`
- `context_cohesion_*`

In addition, difference features are added:

```text
delta_feature_j = target_feature_j - context_feature_j
```

This is intended to capture whether the target sentence deviates from its surrounding document context.

## 6. Compact Summary

The current text-side detector can be summarized as:

```text
Sentence/Context Text
-> style features
-> probabilistic GPT-2 features
-> cohesion-under-deletion features
-> context-aware concatenation
-> stacking classifier or MLP
```

This design aims to combine:

- stylometric cues
- language-model predictability
- robustness/cohesion behavior
- sentence-context inconsistency
