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

        feature_dict['has_fragment'] = len(token._.URL_fragment) > 0
        feature_dict['has_query'] = len(token._.URL_query) > 0
        feature_dict['has_params'] = len(token._.URL_params) > 0
        feature_dict['subdomain'] = subdomain
        feature_dict['domain'] = domain
        feature_dict['domain_extension'] = domain_extension
        feature_dict['total_netloc_tokens'] = len(netloc_tokens)
        # experimental features
        feature_dict['total_netloc_tokens_over_3'] = len(netloc_tokens) > 3
        feature_dict['total_netloc_tokens_over_4'] = len(netloc_tokens) > 4

        return feature_dict