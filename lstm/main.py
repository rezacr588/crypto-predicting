from utils.data_fetcher import fetch_bitcoin_prices
from utils.data_preprocessor import preprocess_data
from utils.model_utils import build_lstm_model, predict_next_hour
import config

# Fetch data
data = fetch_bitcoin_prices()

# Preprocess data
generator, scaler = preprocess_data(data['close'].values, config.SEQUENCE_LENGTH)

# Build and train model
model = build_lstm_model(config.SEQUENCE_LENGTH)
model.fit(generator, epochs=config.EPOCHS)

# Predict next hour
last_sequence = data['close'].values[-config.SEQUENCE_LENGTH:]
predicted_price = predict_next_hour(model, last_sequence, scaler)
print(f"Predicted Price for the Next Hour: {predicted_price}")
