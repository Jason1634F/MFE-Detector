#  Datasets README

## 1. Collected and Generated Dataset Sources
The `Collected and Generated` datasets are derived from a set of benchmark datasets known as HWT Data. The following are the details of each source:

- **CMV**: Opinion statements and restatements from the `/r/ChangeMyView` (CMV) sub - community on Reddit.
- **Yelp**: Reviews from the Yelp review dataset.
- **XSum**: News articles from the XSum dataset.
- **TLDR**: News articles from the TLDR dataset.
- **ELI5**: Question - and - answer texts from the ELI5 dataset.
- **WP**: Story texts from the Reddit WritingPrompts (WP) dataset.
- **ROC**: Story texts from the ROC Stories Corpora (ROC).
- **HellaSwag**: Commonsense reasoning texts from the HellaSwag dataset.
- **SQuAD**: Knowledge illustrations from the SQuAD dataset.
- **SciGen**: Abstracts of scientific articles from the SciGen dataset.

In this study, we employed multiple language models to generate machine - written texts corresponding to each human - written text in the HWT Data. The process is as follows: We select the top 30 tokens from each human - written text. These selected tokens are then fed into the following large language models: GPT-turbo-3.5， T5， LLaMA. Each model generates several machine - generated texts based on the input tokens.

## 2. Human ChatGPT Contrast Corpus Dataset
We also referred to the Human ChatGPT Contrast Corpus (HC3) (https://github.com/Hello-SimpleAI/chatgpt-comparison-detection), which already contains both human - written texts (HWT) and machine - generated texts (MGT). This dataset serves as a comparison reference for our study.
