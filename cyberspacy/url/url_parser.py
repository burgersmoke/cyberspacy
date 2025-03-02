
from spacy.language import Language
from spacy.tokens import Span, Doc

@Language.factory("cyberspacy_url_parser")
class URLParser(object):
    def __init__(self,
                 nlp: Language):
        self.nlp = nlp

    def __call__(self, doc: Doc) -> Doc:

        # let's parse all URL entities
