
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

                token_text = token.text
                # add protocol if it is not there...
                if not token_text.lower().startswith('http'):
                    token_text = 'http://' + token_text

                # TODO: Fix this later, but one of the datasets contains an unmatched bracket
                # like this:
                if 'protected]tageapp' in token_text:
                    print('Special handling of protected]tageapp')
                    print(f'Before: {token_text}')
                    token_text = token_text.replace('protected]tageapp', 'protected_tageapp')
                    print(f'After: {token_text}')

                # let's parse this and add extensions we might use for features
                url_fragments = [''] * 6
                try:
                    url_fragments = urllib.parse.urlparse(token_text)

                    # per this: https://docs.python.org/3/library/urllib.parse.html
                    token._.URL_scheme = url_fragments[0]
                    token._.URL_netloc = url_fragments[1]
                    token._.URL_path = url_fragments[2]
                    token._.URL_params = url_fragments[3]
                    token._.URL_query = url_fragments[4]
                    token._.URL_fragment = url_fragments[5]

                except:
                    print(f'Could not parse this URL:')
                    print(token_text)


        return doc
