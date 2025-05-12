import os
import pandas as pd
import joblib
from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

DATA_FOLDER = "data"
SEQUENCE_LENGTH = 7
COUNTRIES = ["Africa", "Andorra", "Argentina", "Aruba", "Asia"]

lstm_model = load_model("monkeypox_lstm_model.h5")
gru_model = load_model("monkeypox_gru_model.h5")
xgb_model = joblib.load("monkeypox_xgboost.pkl")
svm_model = joblib.load("monkeypox_svm.pkl")

def predict_for_country(country):
    file_path = os.path.join(DATA_FOLDER, f"{country}.xlsx")
    df = pd.read_excel(file_path)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df = df.sort_values(by='date')
    df.fillna(0, inplace=True)

    scaler = MinMaxScaler()
    df['new_cases_scaled'] = scaler.fit_transform(df[['new_cases']])
    recent_data = df['new_cases_scaled'].iloc[-SEQUENCE_LENGTH:].values

    if len(recent_data) < SEQUENCE_LENGTH:
        raise ValueError("Not enough data (need at least 7 rows).")

    input_lstm_gru = recent_data.reshape(1, SEQUENCE_LENGTH, 1)
    input_flat = recent_data.reshape(1, -1)

    predictions = {
        "LSTM": lstm_model.predict(input_lstm_gru)[0][0],
        "GRU": gru_model.predict(input_lstm_gru)[0][0],
        "XGBoost": xgb_model.predict_proba(input_flat)[0][1],
        "SVM": svm_model.predict_proba(input_flat)[0][1]
    }

    trends = {
        model: "Increase Expected" if prob > 0.5 else "Decrease or Stable"
        for model, prob in predictions.items()
    }

    return predictions, trends, df[['date', 'new_cases']].tail(7)

def plot_all_predictions(predictions, country):
    plt.figure(figsize=(8, 5))
    plt.bar(predictions.keys(), predictions.values(), color='skyblue')
    plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
    plt.ylim(0, 1)
    plt.title(f"Model Predictions for {country}")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/model_predictions.png")
    plt.close()

def plot_recent_cases(recent_df, country):
    plt.figure(figsize=(8, 4))
    plt.plot(recent_df['date'], recent_df['new_cases'], marker='o', color='orange')
    plt.title(f"Recent Cases (7 days) - {country}")
    plt.xlabel("Date")
    plt.ylabel("New Cases")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("static/recent_cases.png")
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = {}
    trends = {}
    selected_country = None
    show_plot = False

    if request.method == 'POST':
        selected_country = request.form.get('country')
        try:
            predictions, trends, recent_df = predict_for_country(selected_country)
            plot_all_predictions(predictions, selected_country)
            plot_recent_cases(recent_df, selected_country)
            show_plot = True
        except Exception as e:
            trends = {"Error": str(e)}

    return render_template("index.html",
                           countries=COUNTRIES,
                           selected_country=selected_country,
                           predictions=predictions,
                           trends=trends,
                           show_plot=show_plot)

if __name__ == '__main__':
    app.run(debug=True)
