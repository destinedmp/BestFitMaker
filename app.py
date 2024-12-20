import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

state = st.session_state

st.title("NE 111: Final Project")


def addRow():
    state.data = np.vstack([state.data, ["", ""]])


def rSquared(yT, yP):
    return 1 - np.sum((yT - yP) ** 2) / np.sum((yT - np.mean(yT)) ** 2)

def errors(yT, yP):
    return np.mean(np.abs(yT - yP)), np.sqrt(np.mean((yT - yP) ** 2))

# Fit functions
def linear(x, y):
    c = np.polyfit(x, y, 1)
    yP = c[0] * x + c[1]
    r2 = rSquared(y, yP)
    mae, rmse = errors(y, yP)
    xL = np.linspace(min(x), max(x), 500)
    return {
        "equation": f"y = {c[0]:.2f}x + {c[1]:.2f}",
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "xL": xL,
        "yL": c[0] * xL + c[1]
    }

def poly(x, y, degree):
    c = np.polyfit(x, y, degree)
    polyEq = np.poly1d(c)
    yP = polyEq(x)
    r2 = rSquared(y, yP)
    mae, rmse = errors(y, yP)
    xL = np.linspace(min(x), max(x), 500)
    equation = "y = " + " + ".join(
        [f"{c[i]:.2f}x^{len(c) - i - 1}" for i in range(len(c))]
    ).replace("x^1", "x").replace("x^0", "")
    return {
        "equation": equation,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "xL": xL,
        "yL": polyEq(xL)
    }

def exp(x, y):
    def model(x, a, b): return a * np.exp(b * x)
    par, _ = curve_fit(model, x, y)
    yP = model(x, *par)
    r2 = rSquared(y, yP)
    mae, rmse = errors(y, yP)
    xL = np.linspace(min(x), max(x), 500)
    return {
        "equation": f"y = {par[0]:.2f}e^({par[1]:.2f}x)",
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "xL": xL,
        "yL": model(xL, *par)
    }

def log(x, y):
    if np.any(x <= 0):
        raise ValueError("Logarithmic fit requires all x values to be greater than 0.")
    def model(x, a, b): return a + b * np.log(x)
    par, _ = curve_fit(model, x, y)
    yP = model(x, *par)
    r2 = rSquared(y, yP)
    mae, rmse = errors(y, yP)
    xL = np.linspace(min(x), max(x), 500)
    return {
        "equation": f"y = {par[0]:.2f} + {par[1]:.2f}ln(x)",
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "xL": xL,
        "yL": model(xL, *par)
    }

def power(x, y):
    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("Power Law fit requires all x and y values to be greater than or equal to 0.")
    def model(x, a, b): return a * x**b
    par, _ = curve_fit(model, x, y)
    yP = model(x, *par)
    r2 = rSquared(y, yP)
    mae, rmse = errors(y, yP)
    xL = np.linspace(min(x), max(x), 500)
    return {
        "equation": f"y = {par[0]:.2f}x^{par[1]:.2f}",
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "xL": xL,
        "yL": model(xL, *par)
    }


def plot_fit(x, y, xL, yL, title):
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color="blue", label="Data Points")
    plt.plot(xL, yL, color="red", label=title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True, color="#B0B0B0", linestyle='-', linewidth=0.5)
    plt.xticks(np.linspace(min(x), max(x), 5))
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)


def fitMaker(x, y):
    fit = st.selectbox("Choose a type of fit:", ["", "Linear Regression", "Polynomial Fit", "Exponential Fit", "Logarithmic Fit", "Power Law Fit"])
    degree = None
    if fit == "Polynomial Fit":
        degree = st.number_input("Enter the polynomial degree:", min_value=2, max_value=10, value=2)

    if fit and len(x) > 0 and len(y) > 0:
        try:
            if fit == "Linear Regression":
                result = linear(x, y)
            elif fit == "Polynomial Fit":
                result = poly(x, y, degree)
            elif fit == "Exponential Fit":
                result = exp(x, y)
            elif fit == "Logarithmic Fit":
                result = log(x, y)
            elif fit == "Power Law Fit":
                result = power(x, y)
            st.write(f"**Equation:** {result['equation']}")
            st.write(f"**R² Value:** {result['r2']:.2f}")
            st.write(f"**Mean Average Error:** {result['mae']:.2f}")
            st.write(f"**Root Mean Squared Error:** {result['rmse']:.2f}")
            plot_fit(x, y, result["xL"], result["yL"], fit)
        except Exception as e:
            st.error(f"Error while performing the fit: {e}")
    else:
        st.error("Please select a line of best fit.")

if "data" not in state:
    state.data = np.array([["", ""]])

page = st.sidebar.radio("Select a Page", ("Data Entry", "Upload CSV File"))

if page == "Data Entry":
    st.write("Enter your x and y data (minimum 3 entries). Floats and integers only!")

    inputs = []
    error = False
    for i in range(state.data.shape[0]):
        cols = st.columns(2)
        x = cols[0].text_input(f"X[{i+1}]", value=state.data[i, 0], key=f"x_{i}")
        y = cols[1].text_input(f"Y[{i+1}]", value=state.data[i, 1], key=f"y_{i}")
        try:
            if x.strip(): float(x)
            if y.strip(): float(y)
        except ValueError:
            error = True
            st.error(f"Invalid input in row {i+1}. Please enter only numbers.")
        inputs.append([x, y])

    state.data = np.array(inputs)

    if not error and (state.data[-1, 0].strip() or state.data[-1, 1].strip()):
        addRow()

    x = np.array([float(i) for i in state.data[:, 0] if i.strip()])
    y = np.array([float(i) for i in state.data[:, 1] if i.strip()])
    if len(x) > 2 and len(y) > 2:
        fitMaker(x, y)

elif page == "Upload CSV File":
    st.write("Upload your CSV file with X and Y data. No header, just two columns (X, Y).")

    file = st.file_uploader("Choose a CSV file", type="csv")
    if file:
        try:
            df = pd.read_csv(file, header=None)
            if df.shape[1] == 2:
                x, y = df[0].values, df[1].values
                fitMaker(x, y)
            else:
                st.error("CSV file must have exactly two columns.")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

