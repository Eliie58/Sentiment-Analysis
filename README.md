# Sentiment Analysis

Sentiment analysis using ML

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

---

## Functional Part

### Use Case

Our project is a Sentiment Analysis Machine Learning project, built using nltk and RandomForestClassifier. The project aims to create an ML model that takes as input a sentence, and tries to predict the sentiment in the sentence, positive or negative.

### Goal or problem to solve:
The goal of our Sentiment Analysis Machine Learning project is to develop an ML model that can accurately predict the sentiment of a sentence as positive or negative. The model can help solve several problems for businesses or organizations, such as:
- Our model can assist firms in analyzing customer feedback, social media comments, or product evaluations to ascertain how their clients feel about their goods, services, or brand.
- In order to better fulfill consumer requirements and preferences, organizations can discover areas where their products or services need to be improved by evaluating client sentiment.
- Enhancing marketing initiatives: Our model can assist companies in better understanding how consumers feel about their advertising and promotional initiatives, allowing them to better target their marketing campaigns.
- Analyzing consumer opinion on social media, review websites, and other sources, our model can assist firms in monitoring their brand reputation.

### Who are your customer/users who will benefit/use our project, model, model output:
Organizations interested in examining customer feedback, comments on social media, or product evaluations to learn more about customer sentiment may be possible clients for our Sentiment Analysis Machine Learning project. These businesses may be involved in a range of sectors, including retail, hospitality, healthcare, and finance.
For example, an e-commerce company might use your sentiment analysis model to analyze customer reviews of their products and identify which products are popular and why. A hospitality company might use it to analyze feedback from guests to identify areas for improvement in their service.

### Datasets that we will use for our use case and their source: 
https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp

### Project management framework we prefer to use:
The choice of a project management framework will be influenced by a number of variables, including the scope and complexity of the project, the resources at hand, and the experience and expertise of the project team. However, one project management structure that can be helpful for a Sentiment Analysis Machine Learning project is the Agile technique.
Agile is a project management methodology that places a focus on continuous delivery, iterative and incremental development, and customer collaboration. It is renowned for its flexibility and adaptability and is highly suited for projects with intricate requirements that change quickly, like machine learning programs.
In an Agile project, the project team works in brief iterations, usually lasting two to four weeks, to incrementally provide usable software. Based on input from the stakeholders and the project's development, the project's requirements and scope are continuously reviewed and modified. To prioritize and complete tasks, the project team members collaborate and self-organize.
Scrum, Kanban, and Extreme Programming are three common Agile frameworks (XP). Scrum is a well-known Agile methodology with well defined roles, ceremonies, and artifacts that employs sprints to manage the project. Another Agile methodology, Kanban, places a strong emphasis on visual management and continuous flow, making it a good fit for projects involving machine learning.
In conclusion, Agile can be a good project management framework for your Sentiment Analysis Machine Learning project, and Scrum or Kanban can be prospective Agile frameworks to examine.

### Team organization and roles:
Project Manager: responsible for overseeing the project and ensuring that it is delivered on time.
Data Scientist: responsible for developing the machine learning model and selecting the appropriate algorithms and techniques to achieve the project goals.
Data Engineer: responsible for managing the data pipeline and ensuring that the data is cleaned, preprocessed, and ready for analysis.
Quality Assurance (QA) Analyst: responsible for testing the model and ensuring that it meets the required quality standards.
Business Analyst: responsible for defining the business requirements.








## Techinal Part

### Requiremenets

The project requirements in the [requirements](requirements.txt) file.

### Setup

To create the environment to execute the project, follow these steps:

- Create new conda environment:
  ```
  make create_environment
  ```
- Activate the environment
  ```
  source activate sentiment_analysis
  ```
- Install the requirements
  ```
  make requirements
  ```

### Train the model

To train the model, you can use the train_predict.py file as an example, or run `make train_predict`.

Two pickle files will be created in the root of the folder:

- classifier.pkl: The trained RandomForestClassifier.
- feature_engineer.pkl: The feature engineering class to be used for before inference.

#### Data

The data used for training the model is a txt file placed under [this folder](data/raw/), called train.csv with the following format:

```
i didnt feel humiliated;sadness
```

Each line is one sample, first column is the text, and second is the category. The data is `;` separated.

#### Documentation

To generate the documentation for the project, run:

```
cd docs
make html
```

The documentation are generated under [here](docs/build/html/index.html)

#### ML Schduling

In order to schedule the retraining of the module, our proposed solution is using airflow to create a dag that every set period of time (1 day, 1 wekk, ...) fetchs the latest version of the data, and overrides the train.txt file under [here](data/raw/). Next step in the dag is executing `make train_predict` to create a new trained model on the latest training data.
