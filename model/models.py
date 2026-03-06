import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type:ignore
from tensorflow.keras.models import load_model #type:ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D #type:ignore
from tensorflow.keras.applications import MobileNetV2 #type:ignore
from tensorflow.keras.models import Model #type:ignore

# ✅ Path to model
MODEL_PATH = "model/corn_disease_model.keras"
FEEDBACK_DIR = "feedback_data"

# ✅ Load or create base model
def load_or_create_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Existing model found, loading...")
        model = load_model(MODEL_PATH)
    else:
        print("⚙️ No existing model found. Creating new model...")
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, activation='relu')(x)
        output = Dense(3, activation='softmax')(x)  # assuming 3 disease classes
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ✅ Retraining function
def retrain_model():
    # Check if feedback data folder exists and has subfolders
    if not os.path.exists(FEEDBACK_DIR):
        print("⚠️ No feedback data directory found.")
        return

    subfolders = [f.path for f in os.scandir(FEEDBACK_DIR) if f.is_dir()]
    if not subfolders:
        print("⚠️ No labeled feedback data found for retraining.")
        return

    print("🔁 Starting retraining process with feedback data...")

    # Data generator
    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        FEEDBACK_DIR,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        FEEDBACK_DIR,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        subset='validation'
    )

    # Load or create model
    model = load_or_create_model()

    # Retrain for few epochs
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=3,
        verbose=1
    )

    # Save the updated model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print("✅ Model retrained and saved successfully.")


if __name__ == "__main__":
    retrain_model()
