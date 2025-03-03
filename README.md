<p align="center">
    <img src="./images/cyberspacy_logo_v2.png">
</p>

# cyberspacy
cybersecurity text processing toolkit with spacy, with modular components and use cases

# Overview
cyberspacy is a toolkit library for performing text processing tasks for cybersecurity with the popular [spaCy](https://spacy.io) 
framework. The `cyberspacy` package brings together a number of other packages, trained models, and tests each of which implements specific functionality for some of the tasks involved in text processing in cybersecurity.

`cyberspacy` is modularized so that each component can be used independently. All of `cyberspacy` is designed to be used as part of a `spacy` processing pipeline. Each of the following modules (i.e. "pipes" in spacy terms) is available as part of `cyberspacy`:

- `cyberspacy_malicious_url_classifier`: URLs encountered in text and processed by a provided prediction model based on training and validation data from two datasets on Kaggle for predicting malicious URLs


# Usage
## Installation

At the current time, cyberspacy only supports installing its dependencies with `conda` or an equivalent alternative with the provided environment file:

```bash
conda env create -f environment.yml
```

Once you have cyberspacy and its dependencies, you can either start to use each of its modualr pieces directly or you can use the `PipelineFactory` class to select from a number of "out of the box" pipelines.

For example, this will construct a `PipelineFactory` and get a new pipeline for detecting URLS, and whether they may be malicious:

```python
from cyberspacy.pipelines import PipelineFactory

factory = PipelineFactory()

nlp = factory.create_malicious_url_classifier_pipeline()

text = 'My project is also on http://www.github.com as well'

doc = nlp(text)
```

# Demo 

To give a brief sense of some of the initial capabilities and components of cyberspacy, this notebook gives an [example](notebooks/cyberspacy_demo.ipynb)

# Performance metrics
## cyberspacy_malicious_url_classifier performance results

The current model is a very simple model demonstrating features created in two hours of coding.  More features are soon to be added.

In addition, another work in progress is a LLM (i.e. Transformer) based model to benefit from transfer learning and word-piece tokens.

Train set performance:
```
              precision    recall  f1-score   support

           0       0.84      0.96      0.90    604771
           1       0.82      0.52      0.64    221254

    accuracy                           0.84    826025
   macro avg       0.83      0.74      0.77    826025
weighted avg       0.84      0.84      0.83    826025
```


Validation set performance:
```
              precision    recall  f1-score   support

           0       0.85      0.96      0.90    120955
           1       0.82      0.52      0.64     44251

    accuracy                           0.84    165206
   macro avg       0.83      0.74      0.77    165206
weighted avg       0.84      0.84      0.83    165206
```