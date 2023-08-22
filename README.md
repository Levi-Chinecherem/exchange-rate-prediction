
# Exchange Rate Prediction System (Case Study Of Naira To Dollar)

This repository contains a Python script that demonstrates the use of a neural network to predict exchange rates using economic indicators. It generates random data for exchange rates and economic indicators, trains a neural network model, and provides predictions for exchange rates along with the context of economic indicators. This exchange rate is based on naira to dollar market values.

## Exchange Rate Prediction System: A Powerful Tool for Economic Analysis and Financial Decisions
Welcome to the Exchange Rate Prediction System, a cutting-edge Python application that empowers you to gain insightful economic perspectives and make informed financial decisions through predictive modeling. This system harnesses the potential of neural networks to forecast exchange rates, enabling you to understand currency fluctuations and their implications for economic trends and investment strategies.

### Unveiling Economic Insights
Data Generation: A Glimpse of Real-World Dynamics
Our system generates meticulously crafted synthetic data for exchange rates and economic indicators, emulating the dynamics of real-world financial markets. This virtual economic landscape allows you to experiment and explore without real-world risks.

### Neural Network Model: Unlocking the Power of Prediction
At the core of our system lies a sophisticated neural network model. Designed with precision, this model leverages two linear layers and ReLU activation functions to decipher intricate relationships between economic indicators and exchange rates.

### Training: Nurturing Intelligence through Iteration
Our neural network undergoes intensive training, simulating thousands of cycles through the data. As each epoch passes, the model evolves, learning and adapting to the nuances of the data. This rigorous training process ensures accurate predictions that mirror real-world trends.

## Empowering Financial Decisions
### Test and Train Loss: Measure of Model Mastery
The system not only provides predictions but also quantifies its performance. By showcasing the test and train loss values, you gain insight into how well the model has absorbed the data. A lower loss signifies a more accurate model, primed for reliable predictions.

### Prediction Results Guide: Understanding the Numbers
Our detailed guide transforms numbers into actionable insights. It deciphers each training epoch, loss value, and prediction result, making it easier to grasp the underlying economic significance.

### Real-Time Decision Support: Navigating Financial Markets
With a strong grasp of future exchange rates, you are empowered to make strategic financial decisions. Whether it's investment planning, risk management, or understanding economic trends, our system equips you with the tools to navigate the complexities of financial markets.

### Harness the Future of Economic Insights
The Exchange Rate Prediction System is your gateway to forecasting exchange rates and unraveling economic trends. It combines cutting-edge technology with financial acumen, bridging the gap between data-driven predictions and actionable economic insights.


## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Levi-Chinecherem/exchange-rate-prediction.git
   cd exchange-rate-prediction
   ```

2. Install the required libraries:

   ```bash
   pip install numpy torch scikit-learn tabulate termcolor
   ```

## Usage

Run the script using the following command:

```bash
python exchange_rate_prediction.py
```

## System Overview

The system's main components are as follows:

- **Data Generation**: Consistent random data is generated for exchange rate values and economic indicators.

- **Neural Network Model**: A neural network model is constructed to predict exchange rates based on economic indicators. The model architecture includes two linear layers and ReLU activation.

- **Training**: The model is trained using the generated data. The training process includes 1000 epochs with loss optimization using the Adam optimizer.

- **Results Display**: After training, the script displays the training and test loss. Additionally, the script presents a guide explaining the key training metrics and the prediction results, including headers like Month-Year, Predicted Naira Value, and Data Used for Prediction.

## Example Output

The output includes the training process updates, guide, and the prediction results in a tabulated format.

For example, the prediction results might look like this:

```
PREDICTIONS:

+------------------+------------------------+-------------------------------------+-------------------------------------+
|   Month-Year    | Predicted Naira Value  | Exchange Rate Naira Values: 0.1234  | Economic Indicators                  |
+------------------+------------------------+-------------------------------------+-------------------------------------+
|   January 2020  |         747.8629       |       765.5084                      | [0.22381762 0.53697443 0.5929399     |
|                 |                        |                                     |  0.58008623]                         |
+------------------+------------------------+-------------------------------------+-------------------------------------+
...
```

## Contributing

Contributions are welcome! If you have ideas, suggestions, or find any issues, feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

