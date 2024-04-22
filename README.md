# Home Credit - Credit Risk Model Stability

This repository contains code and resources for predicting loan default risk while accounting for model stability over time. The project utilizes data from a Kaggle competition hosted by Home Credit, focusing on developing predictive models that are both accurate and demonstrate consistent performance across different time periods.

## Project Overview

The challenge of predicting loan default risk for individuals with limited credit history poses a significant barrier to financial inclusion. This project aims to address this problem by exploring various data preprocessing techniques, model architectures, and hyperparameter tuning strategies to develop reliable credit risk models.

The experimental design follows a three-phase approach:

1. **Phase 1**: Establish baseline models using complete datasets with few missing values.
2. **Phase 2**: Expand the feature set by incorporating additional data sources and perform feature engineering.
3. **Phase 3**: Refine model performance through hyperparameter tuning, balancing accuracy (AUC) and temporal stability.

## Dataset

The project utilizes a dataset provided by Home Credit, containing information on 1,526,659 case IDs ranging from loans taken between 2018-2020. The data includes various features related to individuals' credit histories, previous loan applications, demographic information, and other relevant factors.

## Results

The final phase culminated in a tuned LightGBM model emerging as the top performer, achieving an impressive AUC of 0.877 and a stability score of 0.743, significantly outperforming the Logistic Regression and other baseline model. The stability score on the test set was 0.56.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/home-credit-risk-modeling.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Download the dataset from the Kaggle competition: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) - The data was quite large (even for git lfs) so not pushed in this repository.
4. Place the dataset files in the `data/` directory.
5. Run the Jupyter notebooks or Python scripts to preprocess the data, train the models, and evaluate their performance. Feel free to use other 

## Contributing

Contributions to this project are welcome! If you have any suggestions, bug reports, or improvements, please open an issue or submit a pull request. Or contact me on farazjawedd@gmail.com.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Home Credit](https://www.homecredit.net/) for providing the dataset and hosting the Kaggle competition.
- [Kaggle](https://www.kaggle.com/) for hosting the competition and providing a platform for data science challenges. We also took help from discussion forums in the competition for a few helper functions and data processing tasks.

## Team Members

- Meixiang Du
- Faraz Jawed
- Divya Sharma
- Adler Viton
