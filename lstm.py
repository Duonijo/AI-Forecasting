import gc
import warnings
from time import time
import plotly.express as px

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler


np.random.seed(2019)
N = 365 * 6  # 6 years data
rng = pd.date_range('2018-01-01', freq='D', periods=N)
df = pd.DataFrame(np.random.randint(20, size=(N, 4)),
                  columns=['ts', 'tavg', 'prcp', "wind"],
                  index=rng)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

warnings.filterwarnings('ignore')


class LSTMConfiguration:
    def __init__(self, name, epochs, lr, sequence_length, n_layers, hidden_size, horizon, bidirectional):
        self.name = name
        self.epochs = epochs
        self.lr = lr
        self.sequence_length = sequence_length
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.bidirectional = bidirectional


class LSTMConfigLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.configurations = []

    def load_config(self):
        with open(self.file_path, 'r') as file:
            config = yaml.safe_load(file)
            for cfg in config['configurations']:
                name = cfg['name']
                epochs = cfg['epochs']
                lr = float(cfg['lr'])
                sequence_length = cfg['sequence_length']
                model = cfg['model']
                horizon = cfg['horizon']
                n_layers = model['n_layers']
                bidirectional = model['bidirectional']
                hidden_size = model['hidden_size']
                lstm_config = LSTMConfiguration(name, epochs, lr, sequence_length, n_layers, hidden_size, horizon,
                                                bidirectional)
                self.configurations.append(lstm_config)

    def print_configurations(self):
        for config in self.configurations:
            print(f"Name: {config.name}")
            print(f"Epochs: {config.epochs}")
            print(f"Learning Rate: {config.lr}")
            print(f"Sequence Length: {config.sequence_length}")
            print(f"Horizon Size: {config.horizon}")
            print("Model:")
            print(f"  Number of Layers: {config.n_layers}")
            print(f"  Hidden Size: {config.hidden_size}")
            print(f"  Bidirectional: {config.bidirectional}")
            print()


class LSTM(nn.Module):
    def __init__(self, n_features, num_classes, hidden_dim, n_layers, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=bidirectional)

        self.l1 = nn.Linear(hidden_dim, 128)
        self.l2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.num_layers = n_layers
        self.hidden_size = hidden_dim

    def forward(self, X_batch, h_0=None, c_0=None):
        num_directions = 2 if self.lstm.bidirectional else 1

        if h_0 is None and c_0 is None:
            h_0 = torch.zeros(self.num_layers * num_directions, X_batch.size(0), self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_layers * num_directions, X_batch.size(0), self.hidden_size).to(device)

        output, (h_n, c_n) = self.lstm(X_batch, (h_0, c_0))

        # Use the output from the last time step as the representation
        h_n_last_layer = h_n[-1].view(-1, self.hidden_size)
        out = self.relu(h_n_last_layer)
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)

        return out, h_n, c_n


def update_loop(model, test_tensors, criterion, optimiser, num_epochs=20, patience=3):
    X_test, y_test = test_tensors
    h_prev, c_prev = None, None
    predictions = []

    start_time = time()

    size = len(X_test)

    for i in range(size):
        optimiser.param_groups[0]['lr'] = 1e-1

        best_test_loss = float('inf')
        no_improvement_counter = 0
        epoch_idx = 0

        # scheduler = lr_scheduler.MultiStepLR(optimiser, milestones=[20], gamma=0.1)

        model.eval()
        # Select a batch of data for the prediction interval
        with torch.no_grad():
            X_batch = X_test[i].unsqueeze(0).to(device)
            y_batch = y_test[i].unsqueeze(0).to(device)[0][0]

            # Forward pass with the previous hidden state and cell state
            output, _, _ = model(X_batch, None, None)

            # Save predictions
            predictions.append(output.cpu().detach().numpy()[0])

        for epoch in range(num_epochs):
            model.train()

            # Only update the hidden state and cell state once at the beginning of the prediction interval
            if epoch == 0:
                output, _, _ = model(X_batch, None, None)
            else:
                output, h_prev, c_prev = model(X_batch, None, None)
            #
            # h_prev = h_prev.detach()
            # c_prev = c_prev.detach()

            # Compute the loss and perform backpropagation
            optimiser.zero_grad()
            output_first_day_pred = output[0][0]
            loss = criterion(output_first_day_pred, y_batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), 2)
            optimiser.step()
            # scheduler.step()  # Update learning rate

            if epoch % 10 == 0:
                print("Online Training -- Epoch: %d, test loss: %1.5f, learning rate: %e" % (
                    epoch, loss.item(), optimiser.param_groups[0]['lr']))
            if epoch > 10:
                # Early stopping check
                if loss < best_test_loss:
                    best_test_loss = loss
                    no_improvement_counter = 0
                else:

                    no_improvement_counter += 1

                if no_improvement_counter >= patience:
                    print(
                        f"Online Training -- Early stopping at epoch {epoch} as there is no improvement for 5 epochs.")
                    break
            epoch_idx += 1

    # Switch back to training mode if you plan to continue training in subsequent iterations
    model.eval()

    training_time = time() - start_time
    print(f'Online learning completed in {training_time:.2f} seconds')
    return model, optimiser, predictions


def training_loop(n_epochs, model, optimiser, loss_fn, X_train, y_train, X_test, y_test, patience=100, min_epochs=1000):
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    best_test_loss = float('inf')
    no_improvement_counter = 0

    epoch_idx = 0
    scheduler = lr_scheduler.MultiStepLR(optimiser, milestones=[250, 550, 750], gamma=0.1)

    for epoch in range(n_epochs):
        model.train()
        outputs, h_prev, c_prev = model.forward(X_train)

        loss = loss_fn(outputs, y_train)
        # L2 regularization
        l2_regularization = 0.0
        for param in model.parameters():
            l2_regularization += torch.norm(param, p=2)
        loss += 1e-5 * l2_regularization

        optimiser.zero_grad()
        loss.backward()
        # clip_grad_norm_(model.parameters(), 1)
        optimiser.step()
        scheduler.step()  # Update learning rate

        model.eval()
        test_preds, _, _ = model(X_test)
        test_loss = loss_fn(test_preds, y_test)

        if epoch % 100 == 0:
            print("Epoch: %d, train loss: %1.5f, test loss: %1.5f, learning rate: %e" % (
                epoch, loss.item(), test_loss.item(), optimiser.param_groups[0]['lr']))
        if epoch > min_epochs:
            # Early stopping check
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= patience:
                print(f"Early stopping at epoch {epoch} as there is no improvement for {patience} epochs.")
                break
        epoch_idx += 1

    return model, optimiser, epoch_idx


# split a multivariate sequence past, future samples (X and y)
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list()  # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix - 1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)


def mean_absolute_percentage_error(y_true, y_hat):
    return np.mean(np.abs((y_true - y_hat) / y_true)) * 100


def online_training(df_t, target="ts", filepath=None):
    # Create an instance of the class
    if filepath is None:
        raise ValueError("Missing filepath to open YAML config")
    config_loader = LSTMConfigLoader(filepath)

    # Load and parse the configuration
    config_loader.load_config()

    print("df len: ", len(df))

    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    scores = []

    for config in config_loader.configurations:
        n_features, train_test_cutoff, train_tensors, val_tensors, test_tensors, scalers = preprocessing(config,
                                                                                                         df_t,
                                                                                                         min_max_scaler,
                                                                                                         standard_scaler,
                                                                                                         target)
        min_max_scaler, standard_scaler = scalers
        lstm_regressor = LSTM(n_features,
                              config.horizon,
                              config.hidden_size,
                              config.n_layers,
                              config.bidirectional).to(device)

        loss_fn = nn.HuberLoss()
        # loss_fn = nn.MSELoss()
        optimiser = Adam(lstm_regressor.parameters(), lr=config.lr, weight_decay=1e-5)

        lstm_regressor, optimiser, epoch_idx = training_loop(n_epochs=config.epochs,
                                                             model=lstm_regressor,
                                                             optimiser=optimiser,
                                                             loss_fn=loss_fn,
                                                             X_train=train_tensors[0],
                                                             y_train=train_tensors[1],
                                                             X_test=val_tensors[0],
                                                             y_test=val_tensors[1])

        model_training_eval(config,
                            df_t,
                            lstm_regressor,
                            min_max_scaler,
                            standard_scaler,
                            target,
                            train_test_cutoff)

        y_val_true = min_max_scaler.inverse_transform(val_tensors[1])

        # update model with val tensors
        lstm_regressor, optimiser, y_val_hat = update_loop(lstm_regressor, test_tensors=val_tensors, criterion=loss_fn,
                                                           optimiser=optimiser)
        y_val_hat = min_max_scaler.inverse_transform(np.vstack(y_val_hat))
        score = model_eval_scores(y_val_true, y_val_hat, validation=True, epoch_idx=epoch_idx)
        scores.append(score)

        # update and compare with dataset
        lstm_regressor, optimiser, y_test_hat = update_loop(lstm_regressor, test_tensors=test_tensors,
                                                            criterion=loss_fn,
                                                            optimiser=optimiser)
        # format y_test_true and y_test_hat for scoring and plot method
        y_test_true = min_max_scaler.inverse_transform(test_tensors[1])
        y_test_hat = min_max_scaler.inverse_transform(np.vstack(y_test_hat))

        score = model_eval_scores(y_test_true, y_test_hat, validation=False, epoch_idx=epoch_idx)
        scores.append(score)

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv("./lstm/scores.csv")


def model_training_eval(config,
                        df_t,
                        model,
                        min_max_scaler,
                        standard_scaler,
                        target,
                        train_test_cutoff):
    df_X_ss = standard_scaler.transform(df_t.drop(columns=[target]))  # old transformers
    df_y_mm = min_max_scaler.transform(df_t[target].values.reshape(-1, 1))  # old transformers
    # split the sequence
    df_X_ss, df_y_mm = split_sequences(df_X_ss, df_y_mm, config.sequence_length, config.horizon)
    # converting to tensors
    df_X_ss = Variable(torch.Tensor(df_X_ss))
    df_y_mm = Variable(torch.Tensor(df_y_mm))
    # reshaping the dataset
    df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], config.sequence_length, df_X_ss.shape[2]))
    train_predict = model(df_X_ss.to(device))  # forward pass
    data_predict = train_predict[0].data.cpu().numpy()  # numpy conversion
    dataY_plot = df_y_mm.data.numpy()
    data_predict = min_max_scaler.inverse_transform(data_predict)  # reverse transformation
    dataY_plot = min_max_scaler.inverse_transform(dataY_plot)
    true, preds = [], []
    for i in range(len(dataY_plot)):
        true.append(dataY_plot[i][0])
    for i in range(len(data_predict)):
        preds.append(data_predict[i][0])
    plt.figure(figsize=(10, 6))  # plotting
    plt.axvline(x=train_test_cutoff, c='r', linestyle='--')  # size of the training set
    plt.plot(true, label='Actual Data')  # actual plot
    plt.plot(preds, label='Predicted Data')  # predicted plot
    plt.title('Training Prediction')
    plt.legend()
    # plt.savefig("./lstm/training.png", dpi=300)
    plt.show()

    # Create a DataFrame for plotting
    df_train = pd.DataFrame({'Actual Data': true, 'Predicted Data': preds})

    # Plot the data using Plotly Express
    fig = px.line(df_train, title='Training Prediction')
    fig.update_layout(shapes=[
        dict(type='line', x0=train_test_cutoff, x1=train_test_cutoff, y0=0, y1=1, xref='x', yref='paper',
             line=dict(color='red', dash='dash'))])
    # Show the figure (this will open it in the default web browser)
    fig.show()


def model_eval_scores(y_true, y_hat, config=None, validation=True, epoch_idx=None):
    test_rmse = []
    true, preds = [], []
    for i in range(len(y_hat)):
        test_rmse.append(np.sqrt(mean_squared_error([y_hat[i][0]], [y_true[i][0]])))
        true.append(y_true[i][0])
        preds.append(y_hat[i][0])
    rmse = np.sqrt(mean_squared_error(preds, true))
    mae = mean_absolute_error(preds, true)
    mse = mean_squared_error(preds, true)
    y_true_t = np.array(true)
    y_hat_t = np.array(preds)
    mape = mean_absolute_percentage_error(y_true_t, y_hat_t)

    score_print = "Validation" if validation is True else "Test"

    print(f'Validation - RMSE={rmse:.2f}, MAE={mae:.2f}, MSE={mse:.2f}, MAPE={mape:.2f}')
    plt.plot(true, label="Actual Data")
    plt.plot(preds, label="LSTM Predictions")
    plt.title(f'{score_print} Prediction' if validation is True else 'Test Prediction')
    plt.legend()
    # plt.savefig("./lstm/validation.png" if validation is True else './lstm/test.png', dpi=300)
    plt.show()

    # Create a DataFrame for plotting
    df = pd.DataFrame({'Actual Data': true, 'LSTM Predictions': preds})

    # Plot the data using Plotly Express
    fig = px.line(df, title=f'{score_print} Prediction' if validation else 'Test Prediction')
    fig.update_layout(title_text=f'Validation - RMSE={rmse:.2f}, MAE={mae:.2f}, MSE={mse:.2f}, MAPE={mape:.2f}')
    fig.show()

    return {
        "step": "validation" if validation is True else "test",
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "MAPE": mape
    }


def preprocessing(config, df_t, min_max_scaler, standard_scaler, target):
    feature_cols = df_t.columns
    feature_cols = [column for column in feature_cols if column != target]
    print("Features : ", feature_cols)

    X = df_t[feature_cols].values
    X = standard_scaler.fit_transform(X)

    Y = df_t[target].values
    Y = min_max_scaler.fit_transform(Y.reshape(-1, 1))

    # Y = Y.reshape(-1, 1)

    X_ss, y_mm = split_sequences(X, Y, config.sequence_length, config.horizon)

    test_size = 365

    X_t = X_ss[:-test_size]
    X_test = X_ss[-test_size:]
    y_t = y_mm[:-test_size]
    y_test = y_mm[-test_size:]

    val_size = round(0.2 * len(X_t))
    train_test_cutoff = round(0.8 * len(X_t))

    X_train = X_t[:-val_size]
    X_val = X_t[-val_size:]

    y_train = y_t[:-val_size]
    y_val = y_t[-val_size:]

    n_features = X_train.shape[2]

    print("Features count : ", n_features)
    print("Training Shape:", X_train.shape, y_train.shape)
    print("Testing Shape:", X_test.shape, y_test.shape)
    X_train_tensors = Variable(torch.Tensor(X_train))

    X_test_tensors = Variable(torch.Tensor(X_test))
    X_val_tensors = Variable(torch.Tensor(X_val))
    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))
    y_val_tensors = Variable(torch.Tensor(y_val))

    X_train_tensors_final = torch.reshape(X_train_tensors,
                                          (X_train_tensors.shape[0], config.sequence_length,
                                           X_train_tensors.shape[2]))
    X_test_tensors_final = torch.reshape(X_test_tensors,
                                         (X_test_tensors.shape[0], config.sequence_length,
                                          X_test_tensors.shape[2]))
    X_val_tensors_final = torch.reshape(X_val_tensors,
                                        (X_val_tensors.shape[0], config.sequence_length,
                                         X_val_tensors.shape[2]))

    print("Training Shape:", X_train_tensors_final.shape, y_train_tensors.shape)
    print("Val Shape:", X_val_tensors_final.shape, y_val_tensors.shape)
    print("Testing Shape:", X_test_tensors_final.shape, y_test_tensors.shape)

    del X, Y

    gc.collect()

    return n_features, train_test_cutoff, \
        (X_train_tensors_final, y_train_tensors), \
        (X_val_tensors_final, y_val_tensors), \
        (X_test_tensors_final, y_test_tensors), (min_max_scaler, standard_scaler)


if __name__ == "__main__":
    df_t = df.copy()

    online_training(df_t, target="ts",
                    filepath="./config/config_lstm.yml")
