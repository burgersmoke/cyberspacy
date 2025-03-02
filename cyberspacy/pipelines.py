import spacy
from spacy import Language

class PipelineFactory(object):
    def __init__(self):
        self.pipeline_names = ['malicious_url_classifier']

    def create_mailicious_url_classifier_pipeline(self):
        #start with blank
        nlp = spacy.blank("en")

        # break URLs into parts
        nlp.add_pipe("cyberspacy_url_parser")

        nlp.add_pipe("cyberspacy_malicious_url_classifier")

        return nlp
