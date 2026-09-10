from tensorflow.keras.models import load_model

# Load the hybrid CNN-LSTM model
model = load_model("voice_model.h5", compile=False)

print("=== DEEP LEARNING MODEL SUMMARY ===")
# Prints the entire neural network layer structure, output shapes, and parameter counts
model.summary()

print("\n=== LAYER DETAILS ===")
for i, layer in enumerate(model.layers):
    print(f"Layer {i+1}: {layer.name} | Input Shape: {layer.input_shape} | Output Shape: {layer.output_shape}")