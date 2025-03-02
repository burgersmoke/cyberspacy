
from spacy.language import Language
from spacy.tokens import Span, Doc

@Language.factory("cyberspacy_malicious_url_classifier")
class MaliciousURLClassifier(object):
    def __init__(self,
                 nlp: Language,
                 name: str = "cyberspacy_malicious_url_classifier",):
        self.nlp = nlp
        self.name = name

    def __call__(self, doc: Doc) -> Doc:
        # let's classify all URL entities
        for token_idx, token in enumerate(doc):
            if token.like_url:
                token._.URL_malicious_classification = False



        return doc