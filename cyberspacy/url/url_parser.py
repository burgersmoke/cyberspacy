
from spacy.language import Language
from spacy.tokens import Span, Doc
import urllib

@Language.factory("cyberspacy_url_parser")
class URLParser(object):
    def __init__(self,
                 nlp: Language,
                 name: str = "cyberspacy_url_parser",):
        self.nlp = nlp
        self.name = name

    def __call__(self, doc: Doc) -> Doc:
        # let's parse all URL entities
        for token_idx, token in enumerate(doc):
            if token.like_url:
                # let's parse this and add extensions we might use for features
                url_fragments = urllib.parse.urlparse(token.text)

                token._.URL_scheme = url_fragments[0]

        return doc
