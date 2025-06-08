# Project Name

This project is to conduct reddite opinion monitoring by analyzing sentiment alignment and discrepancy between post
and the corresponding comments.

## Folder Structure

- **data/**: Contains all data files used in the project.
- **src/**: Contains all source code files.
- **visuals/**: Contains all visualization outputs.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/lxylyy/reddit_opinion_monitoring.git
   cd your-repo
   ```

2. Run fetch_reddit_data.py
   ```bash
   python fetch_reddit_data.py
   ```

3. Run sentiment_analysis.py, then eda.py
   ```bash
   python sentiment_analysis.py
   python eda.py
   ```


## Dependencies

The project requires the following Python packages:
- python=3.9
- numpy
- pandas
- praw
- nltk
- spacy=3.5.0
- transformers
- tensorflow
- boto3
- pip
- pytorch
- torchvision
- pip:
  - contractions
  - emoji

You can install all dependencies using:
```bash
conda env create -f src/environment.yml
conda activate reddit_env
```

## Usage

Describe how to use your project here.

## License

MIT 