# Datasets

## MultiClaim

See more information in [multiclaim/README.md](multiclaim/README.md).

## AFP-Sum dataset

Download data from [Zenodo](), especially the following:

- `afp-sum.csv` - the entire dataset scrapped from the AFP organization
- `sample2.csv` - 2 fact-checking articles per languages for prompt engineering experiments
- `sample100.csv` - 100 fact-checking articles per language for summarization evaluation

## MultiClaim subset

We selected 1000 posts for the final experiments with the entire pipeline. For these posts, we manually checked the veracity based on the corresponding fact-checking articles.

Files:
- `sampled_posts.csv` - 1000 posts with their ratings and language
