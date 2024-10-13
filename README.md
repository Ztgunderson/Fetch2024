# Forecasting App

## Overview
This project is a forecasting web application that can be ran with Docker.

## Prerequisites
- **Docker**: Ensure Docker is installed on your machine. You can download it from [Docker's official website](https://www.docker.com/get-started).

## Installation and Running the Application

1. **Clone the Image:**
   ```bash
   docker pull ztgunderson/forecasting_app:latest
   ```
2. **Run the Image:**
   ```bash
   docker run -d -p 5000:5000 ztgunderson/forecasting_app:latest
   ```

3. **View Website**
   Go on your favorite browser and type http://127.0.0.1:5000/ to view the predictions

4. **Why did I jsut use a linear model?**
The `Fetch2024Investigations.ipynb` shows my thought process analyzing the data.
What I found was:

- By removing the linear trends, the signal that remains appears either meaningful, showing cyclic shopping trends, or white noise.
- I used ADFuller to estimate stationarity to decide if this trend could be meaningful.
- I came to the conclusion that the zero mean of the signal implies it's white noise.

The conclusions of the investigation showed that creating a more complex model like LSTM or an Autoregressive model would overfit the model to the noisy receipt data. Given the time constraints and the magnitude that we should generalize the data for (predicting a year from only having a year of data), using neural networks would likely not generalize well. Occam's razor methodology would suggest the simpler model can predict these trends better for business estimations each month.