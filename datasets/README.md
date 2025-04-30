# Datasets

## MultiClaim

See more information in [multiclaim/README.md](multiclaim/README.md).

## AFP-Sum dataset

Download data from [Zenodo](https://zenodo.org/records/15267292), especially the following:

- `afp-sum.csv` - the entire dataset scrapped from the AFP organization
- `sample2.csv` - 2 fact-checking articles per languages for prompt engineering experiments
- `sample100.csv` - 100 fact-checking articles per language for summarization evaluation

## MultiClaim subset

We selected 1000 posts for the final experiments with the entire pipeline. For these posts, we manually checked the veracity based on the corresponding fact-checking articles.

Files:
- `sampled_posts.csv` - 1000 posts with their ratings and language

## Paper citing

If you use the code or information from this repository, please cite our paper, which is available on arXiv.

```bibtex
@misc{vykopal2025generativeaidrivenclaimretrievalcapable,
      title={A Generative-AI-Driven Claim Retrieval System Capable of Detecting and Retrieving Claims from Social Media Platforms in Multiple Languages}, 
      author={Ivan Vykopal and Martin Hyben and Robert Moro and Michal Gregor and Jakub Simko},
      year={2025},
      eprint={2504.20668},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.20668}, 
}
```