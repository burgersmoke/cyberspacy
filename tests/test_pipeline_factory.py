import cyberspacy
from cyberspacy.pipelines import PipelineFactory

import pytest



class TestPipelineFactory:

    def test_factory_construction(self):
        factory = PipelineFactory()

        assert factory is not None

    def test_malicious_url_pipeline_construction(self):
        factory = PipelineFactory()

        assert factory is not None

        nlp = factory.create_malicious_url_classifier_pipeline()

        assert nlp is not None

        pipe_names = list(nlp.pipe_names)

        assert 'cyberspacy_url_parser' in pipe_names

        assert 'cyberspacy_malicious_url_classifier' in pipe_names
