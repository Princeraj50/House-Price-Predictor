![AdobeStock_464438353](https://github.com/user-attachments/assets/be58ae35-7fa9-410b-b7b5-73fbe1c19e85)
# Boston House Price Prediction

**[Deployed URL](https://house-price-predictor-152k.onrender.com/predict)**

This repository contains an end-to-end machine learning project on predicting house prices in Boston. The project covers steps from Exploratory Data Analysis (EDA), data cleaning, model building, saving the model as a pickle file, creating a front-end using HTML, and deploying it on the web using Render.

## About Dataset

### Context
The dataset is used to explore more on regression algorithms.

### Content
Each record in the database describes a Boston suburb or town. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. The attributes are defined as follows (taken from the UCI Machine Learning Repository):

1. **CRIM**: per capita crime rate by town
2. **ZN**: proportion of residential land zoned for lots over 25,000 sq.ft.
3. **INDUS**: proportion of non-retail business acres per town
4. **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
5. **NOX**: nitric oxides concentration (parts per 10 million)
6. **RM**: average number of rooms per dwelling
7. **AGE**: proportion of owner-occupied units built prior to 1940
8. **DIS**: weighted distances to five Boston employment centers
9. **RAD**: index of accessibility to radial highways
10. **TAX**: full-value property-tax rate per $10,000
11. **PTRATIO**: pupil-teacher ratio by town
12. **B**: 1000(Bk − 0.63)² where Bk is the proportion of blacks by town
13. **LSTAT**: percentage of lower status of the population
14. **MEDV**: median value of owner-occupied homes in $1000s

For more information, visit the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing).

## Project Structure

- `data/`: Contains the dataset.
- `notebooks/`: Jupyter notebooks for EDA, data cleaning, and model building.
- `models/`: Contains the saved model pickle file.
- `templates/`: HTML templates for the front-end.
- `static/`: Static files for styling.
- `app.py`: Flask application for serving the predictions.
- `wsgi.py`: Entry point for Gunicorn.
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Princeraj50/House-Price-Predictor.git
    cd house-price-predictor
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Flask application:
    ```bash
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Enter the input values for prediction and click on the "Predict" button.

## Deployment

The model is deployed on Render. You can access it [here](https://house-price-predictor-152k.onrender.com/predict).

## License

This project is licensed under the MIT License.

## Contact

For any questions, please contact [choclateyraj50@gmail.com].