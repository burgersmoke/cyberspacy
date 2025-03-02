import re
from typing import Dict, Any
import typing

from spacy.tokens import Token


class URLFeatureExtractor(object):

    def extract_feature_dictionary(self, token: Token) -> Dict[str, Any]:
        feature_dict = {}

        if not token.like_url:
            return feature_dict

        netloc_str = token._.URL_netloc

        netloc_tokens = netloc_str.split('.')


        domain = ''
        subdomain = ''
        domain_extension = ''

        if len(netloc_tokens) == 2:
            domain = netloc_tokens[0]
            domain_extension = netloc_tokens[1]
        elif len(netloc_tokens) >= 3:
            subdomain = netloc_tokens[0]
            domain = netloc_tokens[1]
            domain_extension = netloc_tokens[-1]

        feature_dict['fragment'] = token._.URL_fragment
        feature_dict['subdomain'] = subdomain
        feature_dict['domain'] = domain
        feature_dict['domain_extension'] = domain_extension

        return feature_dict