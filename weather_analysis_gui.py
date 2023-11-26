import tkinter as tk
from tkinter import ttk
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

spark = SparkSession.builder.appName("WeatherAnalysis").getOrCreate()

data_path = "weather_analysis.csv"git 
df = spark.read.csv(data_path, header=True, inferSchema=True)

df.printSchema()

window = tk.Tk()
window.title("Weather Condition Analysis")

canvas_list = []

def analyze_weather_conditions():
    weather_data = df.groupBy("Weather Condition").agg({"Temperature": "mean", "Humidity": "mean", 
                                                        "Wind Speed": "mean", "Precipitation": "sum"}).toPandas()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Weather Condition Analysis")

    axs[0, 0].bar(weather_data['Weather Condition'], weather_data['avg(Temperature)'])
    axs[0, 0].set_title("Average Temperature (°C)")

    axs[0, 1].bar(weather_data['Weather Condition'], weather_data['avg(Humidity)'])
    axs[0, 1].set_title("Average Humidity (%)")

    axs[1, 0].bar(weather_data['Weather Condition'], weather_data['avg(Wind Speed)'])
    axs[1, 0].set_title("Average Wind Speed (km/hr)")

    axs[1, 1].bar(weather_data['Weather Condition'], weather_data['sum(Precipitation)'])
    axs[1, 1].set_title("Total Precipitation (mm)")

    new_window = tk.Toplevel(window)
    new_window.title("Weather Condition Analysis Plots")

    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    canvas_list.append(canvas)

def analyze_air_pressure():
    air_pressure_data = df.select("Air Pressure", "Signal Strength").toPandas()

    fig, ax = plt.subplots()
    ax.scatter(air_pressure_data['Air Pressure'], air_pressure_data['Signal Strength'])
    ax.set_xlabel("Air Pressure (hPa)")
    ax.set_ylabel("Signal Strength (dBm)")
    ax.set_title("Air Pressure vs. Signal Strength")

    new_window = tk.Toplevel(window)
    new_window.title("Air Pressure Analysis Plot")

    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    canvas_list.append(canvas)

def predict_rf_signal():
    clear_plots()

    user_temperature = float(temperature_entry.get())
    user_humidity = float(humidity_entry.get())
    user_wind_speed = float(wind_speed_entry.get())

    feature_cols = ["Temperature", "Humidity", "Wind Speed"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    lr = LinearRegression(featuresCol="features", labelCol="Signal Strength", regParam=0.01) 
    pipeline = Pipeline(stages=[assembler, lr])

    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    #model = pipeline.fit(df)
    model = pipeline.fit(train_data)

    user_df = spark.createDataFrame([(user_temperature, user_humidity, user_wind_speed)], feature_cols)

    prediction = model.transform(user_df).select("prediction").collect()[0][0]

    evaluator = RegressionEvaluator(labelCol="Signal Strength", predictionCol="prediction", metricName="rmse")
    test_predictions = model.transform(test_data)
    rmse = evaluator.evaluate(test_predictions)
    print(f"Root Mean Squared Error (RMSE) on the test set: {rmse:.2f}")

    prediction_label.config(text=f"Predicted Signal Strength: {prediction:.2f} dBm")

def analyze_feature_correlation():
    clear_plots()

    feature_cols = ["Temperature", "Humidity", "Wind Speed", "Precipitation", "Air Pressure", "WiFi Strength"]

    correlation_data = df.select(feature_cols + ["Signal Strength"]).toPandas()

    correlation_matrix = correlation_data.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(correlation_matrix, cmap="coolwarm")
    fig.colorbar(cax)

    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title("Feature Correlation with Signal Strength")

    new_window = tk.Toplevel(window)
    new_window.title("Feature Correlation Analysis Plot")

    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    canvas_list.append(canvas)

def clear_plots():
    for canvas in canvas_list:
        canvas.get_tk_widget().destroy()
    canvas_list.clear()

def clear_gui():
    temperature_entry.delete(0, tk.END)
    humidity_entry.delete(0, tk.END)
    wind_speed_entry.delete(0, tk.END)

    prediction_label.config(text="")

    clear_plots()

analyze_weather_button = ttk.Button(window, text="Analyze Weather Conditions", command=analyze_weather_conditions)
analyze_weather_button.grid(row=1, column=0, pady=10)

analyze_air_pressure_button = ttk.Button(window, text="Analyze Air Pressure", command=analyze_air_pressure)
analyze_air_pressure_button.grid(row=1, column=1, pady=10)

correlation_button = ttk.Button(window, text="Feature Correlation Analysis", command=analyze_feature_correlation)
correlation_button.grid(row=1, column=2, pady=5)

temperature_label = ttk.Label(window, text="Enter Temperature (°C):")
temperature_label.grid(row=2, column=0, pady=5)
temperature_entry = ttk.Entry(window)
temperature_entry.grid(row=2, column=1, pady=5)

humidity_label = ttk.Label(window, text="Enter Humidity (%):")
humidity_label.grid(row=3, column=0, pady=5)
humidity_entry = ttk.Entry(window)
humidity_entry.grid(row=3, column=1, pady=5)

wind_speed_label = ttk.Label(window, text="Enter Wind Speed (km/hr):")
wind_speed_label.grid(row=4, column=0, pady=5)
wind_speed_entry = ttk.Entry(window)
wind_speed_entry.grid(row=4, column=1, pady=5)

predict_button = ttk.Button(window, text="Predict RF Signal", command=predict_rf_signal)
predict_button.grid(row=5, column=0, pady=5)

clear_button = ttk.Button(window, text="Clear", command=clear_gui)
clear_button.grid(row=0, column=0, pady=10)


prediction_label = ttk.Label(window, text="")
prediction_label.grid(row=7, column=0, columnspan=10, pady=10)

window.mainloop()

spark.stop()