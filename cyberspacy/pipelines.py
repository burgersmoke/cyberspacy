import spacy
from spacy import Language

import cyberspacy.url.url_parser
import cyberspacy.url.malicious_url_classifier
from cyberspacy._extensions import set_extensions

class PipelineFactory(object):
    def __init__(self):
        self.pipeline_names = ['malicious_url_classifier']

        set_extensions()

    def create_malicious_url_classifier_pipeline(self):
        #start with blank
        nlp = spacy.blank("en")

        # break URLs into parts
        nlp.add_pipe("cyberspacy_url_parser")

        nlp.add_pipe("cyberspacy_malicious_url_classifier")

        return nlp
