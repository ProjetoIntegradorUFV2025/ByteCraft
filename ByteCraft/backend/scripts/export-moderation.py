"""Export the supplied fitted pipeline, without training; Python is a build-time tool only."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import sys

backend = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=backend / 'moderation-source')
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))
from src.nickname_model import load_saved_model, predict_nickname, TrainingConfig
from src.nickname_normalization import normalize_nickname
from src.lexical_filter import load_candidates, load_candidate_reviews
from src.moderation_api import _moderation_response

model = load_saved_model(args.source / 'models/nickname_moderation.joblib')
vectorizer = model.named_steps['tfidf']
classifier = model.named_steps['classifier']
assert vectorizer.analyzer == 'char' and vectorizer.ngram_range == (2, 5)
assert vectorizer.sublinear_tf and vectorizer.norm == 'l2' and not vectorizer.lowercase
assert list(classifier.classes_) == ['inadequado', 'permitido']
assert classifier.coef_.shape == (1, len(vectorizer.vocabulary_))

curated = args.source / 'data/curated'
reviewed = []
samples = set()
for filename, label_key in [('nickname_examples.csv', 'label'),
                            ('manual_evaluation.csv', 'expected_label'),
                            ('bytecraft_nicknames_moderacao.csv', 'expected_label')]:
    with (curated / filename).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            samples.add(row['nickname'])
            if row['review_status'] == 'revisado' and row[label_key] == 'inadequado':
                reviewed.append(normalize_nickname(row['nickname']))
ledger = load_candidate_reviews()
lexical = [{'term': c.normalized_term, 'classification': ledger[c.source_line].classification}
           for c in load_candidates()]
payload = {
    'schema': 1,
    'sourceSha256': hashlib.sha256((args.source / 'models/nickname_moderation.joblib').read_bytes()).hexdigest(),
    'minimumConfidence': TrainingConfig.minimum_confidence,
    'classes': list(classifier.classes_),
    'terms': list(vectorizer.get_feature_names_out()),
    'idf': vectorizer.idf_.tolist(),
    'coefficients': classifier.coef_[0].tolist(),
    'intercept': float(classifier.intercept_[0]),
    'reviewedInappropriate': sorted(set(reviewed)),
    'lexical': lexical,
    # Java uses these mappings to reproduce Python casefold, including sharp-s.
    'casefold': {chr(i): chr(i).casefold() for i in range(sys.maxunicode + 1)
                 if chr(i) != chr(i).casefold()},
}
resource = backend / 'src/main/resources/moderation/model.json'
resource.parent.mkdir(parents=True, exist_ok=True)
resource.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')

# Golden cases are calculated by the original Python inference, not the Java port.
samples.update(['JogadorLegal', 'Ana', 'João', 'RobôAzul', 'a', '123', 'Straße', 'İ',
                'Ａｎａ', '👩‍💻Ana', 'Ana\u00a0Maria', 'e\u0301', 'a\t b', 'p.u.t.a', 'p u t a'])
for candidate in load_candidates():
    term = candidate.term
    samples.update([term.upper(), term + '123', '.'.join(term), ' '.join(term),
                    term.translate(str.maketrans('aoies', '40135')), term * 2])
randomizer = random.Random(42)
for _ in range(150):
    samples.add(''.join(randomizer.choices('abcde0123_@ $!çé🚀', k=randomizer.randint(1, 30))).strip() or 'a')
cases = []
for nickname in sorted(samples):
    if not nickname.strip() or len(nickname.strip()) > 50:
        continue
    value = nickname.strip()
    prediction = predict_nickname(value, model=model)
    response = _moderation_response(prediction)
    cases.append({'nickname': nickname, 'normalized': normalize_nickname(value),
                  'probabilityPermitted': prediction['probabilities']['permitido'],
                  'status': response.status})
fixtures = backend / 'src/test/resources/moderation/python-parity.json'
fixtures.parent.mkdir(parents=True, exist_ok=True)
fixtures.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Exported fitted model: {len(payload["terms"])} features; {len(cases)} original Python parity cases.')
