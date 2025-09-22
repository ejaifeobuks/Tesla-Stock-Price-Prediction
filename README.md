# Tesla Stock Price Prediction
This project was a personal project aimed at gaining some experience working with Datasets, Pandas and some common ML Libraries. The project predicts Tesla stock prices using historical market data and various regression algorithms.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ejaifeobuks/Tesla-Stock-Price-Prediction/blob/main/Tesla_Stock_Market_Price_Prediction.ipynb)

## Overview

This project analyzes Tesla's historical stock data from 2014 to 2023 and implements multiple machine learning models to predict the next day's closing price. The analysis includes data visualization, correlation analysis, and performance comparison of different regression algorithms.

## Dataset

The project uses Tesla stock market data (`tsla_2014_2023.csv`) containing the following features:

- **Date**: Trading date
- **Open**: Opening price
- **High**: Highest price during the day
- **Low**: Lowest price during the day
- **Close**: Closing price
- **Volume**: Number of shares traded
- **Next Day Close**: Target variable for prediction

## Features

- 📊 **Data Analysis**: Comprehensive exploration of Tesla stock data
- 📈 **Visualization**: Time series plots and correlation heatmaps
- 🤖 **Multiple Models**: Implementation of various regression algorithms
- 📉 **Performance Comparison**: Evaluation using MAE and R² scores
- 🔄 **Data Preprocessing**: Feature scaling and train-test splitting

## Machine Learning Models

The project implements and compares the following regression models:

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **Support Vector Regression (SVR)**

## Key Technologies

- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **NumPy**: Numerical computing

## Usage

1. **Jupyter Notebook**: Open `Tesla_Stock_Market_Price_Prediction.ipynb` in Jupyter Notebook or JupyterLab
2. **Google Colab**: Click the "Open in Colab" badge above to run in Google Colab
3. **Local Environment**: Run the notebook cells sequentially to reproduce the analysis

## Project Structure

```
Tesla-Stock-Price-Prediction/
│
├── Tesla_Stock_Market_Price_Prediction.ipynb    # Main analysis notebook
├── README.md                                    # Project documentation
└── tsla_2014_2023.csv                          # Dataset (not included in repo)
```

## Analysis Workflow

1. **Data Loading**: Import Tesla stock data from CSV file
2. **Data Exploration**: Examine data structure, check for missing values
3. **Data Preprocessing**: Convert date column, set as index, handle missing data
4. **Visualization**: Create time series plots and correlation heatmaps
5. **Feature Engineering**: Prepare features and target variable
6. **Model Training**: Train multiple regression models
7. **Evaluation**: Compare model performance using MAE metrics

## Results

The project evaluates model performance using Mean Absolute Error (MAE) and R² scores. Each model's predictions are compared to determine the best performing algorithm for Tesla stock price prediction.

## Key Insights

- Correlation analysis reveals relationships between different stock price features
- Time series visualization shows Tesla's stock price trends over the 2014-2023 period
- Multiple regression models provide different prediction accuracies
- Feature scaling improves model performance

## Future Enhancements

- [ ] Add more advanced models (LSTM, ARIMA)
- [ ] Include technical indicators as features
- [ ] Implement cross-validation
- [ ] Add model deployment capabilities
- [ ] Include sentiment analysis from news data

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**ejaifeobuks**

- GitHub: [@ejaifeobuks](https://github.com/ejaifeobuks)

## Acknowledgments

- Tesla, Inc. for providing publicly available stock data
- The scikit-learn community for excellent machine learning tools
- The Python data science ecosystem

---

⭐ If you found this project helpful, please consider giving it a star!
