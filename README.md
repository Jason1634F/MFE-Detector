Description of Collected and Generated Datasets:
The Collected and Generated datasets are sourced from a set of benchmark datasets, namely the HWT Data. The composition of HWT Data is as follows:
CMV: Opinion statements and restatements from the /r/ChangeMyView (CMV) subcommunity on Reddit.
Yelp: Reviews from the Yelp review dataset.
XSum: News articles from the XSum dataset.
TLDR: News articles from the TLDR dataset.
ELI5: Question - and - answer texts from the ELI5 dataset.
WP: Story texts from the Reddit WritingPrompts (WP) dataset.
ROC: Story texts from the ROC Stories Corpora (ROC).
HellaSwag: Commonsense reasoning texts from the HellaSwag dataset.
SQuAD: Knowledge illustrations from the SQuAD dataset.
SciGen: Abstracts of scientific articles from the SciGen dataset.
In this study, multiple language models such as GPT-turbo-3.5, T5, and LLaMA are utilized to construct several corresponding machine - generated texts for each human - written text in the HWT Data. The specific approach is to select the top 30 tokens of the human - written text and input them into the above - mentioned large language models to generate several machine - generated texts.
We also referred to a comparison dataset, the Human ChatGPT Contrast Corpus (HC3) (https://github.com/Hello-SimpleAI/chatgpt-comparison-detection), which already contains HWT and MGT.
