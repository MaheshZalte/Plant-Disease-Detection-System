import tensorflow as tf # type: ignore

model = tf.keras.models.load_model("PlantDisease_Model.h5")

# Get class indices from your dataset
class_indices = model.class_names if hasattr(model, "class_names") else None

if class_indices:
    print("Your Model's Class Labels: ", class_indices)
else:
    print("Warning: Class names not found! Check your dataset structure.")
