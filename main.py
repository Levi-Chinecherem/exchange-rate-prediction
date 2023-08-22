import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import calendar
from tabulate import tabulate
from termcolor import colored

# Generate consistent random data
np.random.seed(42)
num_samples = 300
num_features = 5

exchange_rate_naira_values = np.random.uniform(460, 800, num_samples)
economic_indicators = np.random.rand(num_samples, num_features - 1)

data = np.column_stack((exchange_rate_naira_values, economic_indicators))

target_values = np.roll(exchange_rate_naira_values, -1)
target_values[-1] = target_values[-2]

X_train, X_test, y_train, y_test = train_test_split(data, target_values, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class ExchangeRatePredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ExchangeRatePredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        y_pred = self.layer2(x)
        return y_pred

input_size = X_train.shape[1]
hidden_size = 64
model = ExchangeRatePredictor(input_size, hidden_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    y_pred_train = model(X_train)
    train_loss = criterion(y_pred_train.squeeze(), y_train)
    print(f'Train Loss: {train_loss.item():.4f}')

    y_pred_test = model(X_test)
    test_loss = criterion(y_pred_test.squeeze(), y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

print(" ")
# Print the guide
guide = colored('''
PREDICTION DATA GUIDE
1. Epoch [100/1000], Loss: 51520.3633:
   This line indicates that the training process has
   completed 100 epochs out of a total of 1000 epochs.
   An epoch is a complete pass through the entire training
   dataset. The value after "Loss" (e.g., 51520.3633) represents
   the current value of the loss function at the end of this
   epoch. The loss function quantifies the difference between
   the predicted values and the actual target values. Lower
   values indicate that the model's predictions are closer
   to the actual values, which is the goal of training.

2. Epoch [200/1000], Loss: 18719.6973:
   This is similar to the previous line. It indicates that the
   model has completed 200 epochs, and the loss value has
   decreased from the previous epoch, indicating that the model
   is improving in its predictions.

3. Train Loss: 18686.8750: This is the final training loss
   value after completing all 1000 epochs. It represents the
   average loss over the entire training dataset after the model
   has been trained. The goal is to minimize this loss value as
   much as possible to ensure accurate predictions.

4. Test Loss: 17535.9004: This is the loss value calculated
   using the model's predictions on the test dataset. The test
   loss provides an indication of how well the model generalizes
   to new, unseen data. If the test loss is close to the
   training loss, it suggests that the model is not overfitting
   (performing well on training data but poorly on test data)
   and is likely making reasonable predictions on new data.

In summary, these lines provide insights into the training
progress of the neural network model. The decreasing loss
values generally indicate that the model is learning and
improving its predictions over the course of training.
However, the model's performance on the test dataset
(as indicated by the test loss) is more critical,
as it shows how well the model is expected to perform on new,
unseen data.

PREDICTION RESULT GUIDE
Month-Year: This is a header column that denotes the month
and year for which the prediction is being made.
It provides a time reference for the predicted values.

Predicted Naira Value: This column displays the predicted
exchange rate value in Naira for the specified month and
year. This value is calculated using the trained neural
network model.

Data Used for Prediction: This column provides additional
context by showing the actual data that was used for making
the prediction. It includes two parts:

Exchange Rate Naira Values: This is the exchange rate value
in Naira that was used as input for the prediction. It's the
first value in the array of data.
Economic Indicators: These are the economic indicators that
were used as additional features for the prediction. These
indicators provide context about the economic conditions for
that particular month and year.

==========================================
This line is a visual separator to distinguish
the title row from the prediction rows. It doesn't
impact the system's functionality; it's purely for
formatting purposes to make the output more readable
and organized.

January 2020: This is an example prediction row.
It represents the output of the model for January 2020.
The subsequent values under "Predicted Naira Value",
"Exchange Rate Naira Values", and "Economic Indicators"
are populated based on the model's predictions and the
actual input data.

In summary, this code segment is responsible for printing
the predictions and relevant data in a well-formatted tabular
manner. It's designed to provide a clear understanding of the
predicted exchange rate values while also showing the input
data used for those predictions. The formatting and
organization of this output help users interpret the
results and their context more easily.\n

''', 'blue', attrs=['bold'])
print("________________________________GUIDE________________________________")
print(guide)

# Generate predictions
with torch.no_grad():
    test_predictions = model(X_test)

# Prepare prediction data for tabulate
prediction_table = []
for i in range(len(X_test)):
    naira_value = test_predictions[i].item()
    year = 2020 + i // 12  # Assuming data is collected monthly starting from 2020
    month = i % 12 + 1  # Adjusting month to 1-based index
    month_name = calendar.month_name[month]
    
    prediction_data = {
        "Month-Year": f"{month_name} {year}",
        "Predicted Naira Value": f"{naira_value:.4f}",
        "Exchange Rate Naira Values": f"{X_test[i][0].item():.4f}",
        "Economic Indicators": X_test[i][1:].numpy()
    }
    
    prediction_table.append(prediction_data)

# Print the test predictions using tabulate
print("\nPREDICTIONS:\n")
print(tabulate(prediction_table, headers="keys", tablefmt="pretty"))
