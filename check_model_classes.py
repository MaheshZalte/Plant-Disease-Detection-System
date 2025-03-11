import tensorflow as tf

model = tf.keras.models.load_model("PlantDisease_Model.h5")
num_classes = model.output_shape[-1]
print(f"✅ Model predicts {num_classes} classes.")

if hasattr(model, 'class_names'):
    print("Model Class Labels:", model.class_names)
else:
    print("⚠ Model does not store class labels. Ensure they match disease_info.json.")
