#  MFE-Detector README

## Datasets
### 1. Collected and Generated Dataset Sources
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

### 2. Human ChatGPT Contrast Corpus Dataset
We also referred to the Human ChatGPT Contrast Corpus (HC3) (https://github.com/Hello-SimpleAI/chatgpt-comparison-detection), which already contains both human - written texts (HWT) and machine - generated texts (MGT). This dataset serves as a comparison reference for our study.

## Code File Descriptions
- **Build_data**: Performs pre - processing operations on data, standardizing data format and cleaning invalid data to lay the foundation for subsequent analysis.
- **Compute_features**: Focuses on calculating text features, extracting representative attribute information from text to facilitate subsequent model processing.
- **Detect_program**: Implements the MFE - Detector detection program proposed in the paper. It is the code carrier of the core detection function and also a part of the ablation experiment.
- **Func_Tool**: Serves as a utility program, providing various auxiliary functions for easy invocation by other program modules.
- **MLP_classify**: Uses a fully - connected network (multilayer perceptron) to classify text feature data by category, applied in ablation experiments.
- **Baseline_experiment**: Runs multiple baseline methods such as Log - Likelihood, Rank, Log - Rank, Entropy, and Supervised Detector for comparison and evaluation.
- **Perturb_text**: As a tool, it is used for perturbing text, applicable to methods such as DetectGPT, DetectLLM - LRR, and DetectLLM - NPR.
- **DetectGPT_and_LRR_NPR**: Mainly conducts threshold selection for the three algorithms of DetectGPT, DetectLLM - LRR, and DetectLLM - NPR to optimize the detection performance of the algorithms.
