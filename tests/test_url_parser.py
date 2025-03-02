import cyberspacy
from cyberspacy.pipelines import PipelineFactory

import pytest



class TestURLParser:

    def test_parse(self):
        factory = PipelineFactory()

        assert factory is not None

        nlp = factory.create_malicious_url_classifier_pipeline()

        assert nlp is not None

        text = 'My project is also on http://www.github.com as well'

        doc = nlp(text)

        url_like_tokens = [x for x in doc if x.like_url]

        assert url_like_tokens

        first_ent = url_like_tokens[0]

        url_scheme = first_ent._.URL_scheme

        assert url_scheme == 'http'

    def test_parse_with_fragment(self):
        factory = PipelineFactory()

        assert factory is not None

        nlp = factory.create_malicious_url_classifier_pipeline()

        assert nlp is not None

        text = 'www.example.com/foo.html#bar'
        doc = nlp(text)
        url_like_tokens = [x for x in doc if x.like_url]

        assert url_like_tokens

        first_url = url_like_tokens[0]
        url_fragment = first_url._.URL_fragment

        assert url_fragment == 'bar'