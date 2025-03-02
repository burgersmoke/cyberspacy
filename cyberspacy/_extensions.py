"""This module will set extension attributes and methods for medspaCy. Examples include custom methods like span._.window()"""
from spacy.tokens import Doc, Span, Token

# from .io.doc_consumer import ALLOWED_DATA_TYPES

ALLOWED_DATA_TYPES = ("ents", "group", "section", "context", "doc")


def set_extensions():
    """Set custom medspaCy extensions for Token, Span, and Doc classes."""
    set_token_extensions()
    set_span_extensions()
    set_doc_extensions()


def set_token_extensions():
    for attr, attr_info in _token_extensions.items():
        try:
            Token.set_extension(attr, **attr_info)
        except ValueError as e:  # If the attribute has already set, this will raise an error
            pass


def set_span_extensions():
    for attr, attr_info in _span_extensions.items():
        try:
            Span.set_extension(attr, **attr_info)
        except ValueError as e:  # If the attribute has already set, this will raise an error
            # print(e)
            pass


def set_doc_extensions():
    for attr, attr_info in _doc_extensions.items():
        try:
            Doc.set_extension(attr, **attr_info)
        except ValueError as e:  # If the attribute has already set, this will raise an error
            pass


def get_extensions():
    """Get a list of extensions for Token, Span, and Doc classes."""
    return {
        "Token": get_token_extensions(),
        "Span": get_span_extensions(),
        "Doc": get_doc_extensions(),
    }


def get_token_extensions():
    return _token_extensions


def get_span_extensions():
    return _span_extensions


def get_doc_extensions():
    return _doc_extensions

_token_extensions = {
    "URL_scheme": {"default": None},
    "URL_netloc": {"default": None},
    "URL_path": {"default": None},
    "URL_params": {"default": None},
    "URL_query": {"default": None},
    "URL_fragment": {"default": None},
    "URL_malicious_classification": {"default": None},
}


_span_extensions = {

}

_doc_extensions = {

}
