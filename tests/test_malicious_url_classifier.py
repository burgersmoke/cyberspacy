import cyberspacy
from cyberspacy.pipelines import PipelineFactory

import pytest



class TestMaliciousURLClassifier:

    def test_classify(self):
        factory = PipelineFactory()

        assert factory is not None

        nlp = factory.create_malicious_url_classifier_pipeline()

        assert nlp is not None

        text = 'My project is also on http://www.github.com as well'

        doc = nlp(text)

        url_like_tokens = [x for x in doc if x.like_url]

        first_ent = url_like_tokens[0]

        assert first_ent._.URL_malicious_classification is False