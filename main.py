from flask import Flask, jsonify, request, send_from_directory
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from datetime import datetime, timedelta
from bson.objectid import ObjectId
import os, json, numpy as np, shutil, subprocess
from PIL import Image
import tensorflow as tf

app = Flask(__name__, static_folder="static", static_url_path="")
app.secret_key = "supersecretkey"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MONGO_URI"] = "mongodb://localhost:27017/corn_disease_db"
app.config["JWT_SECRET_KEY"] = "cornleaf_secret"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=1)

mongo = PyMongo(app)
users = mongo.db.users
history = mongo.db.history
jwt = JWTManager(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

MODEL_PATH = "model/corn_leaf_cnn.keras"
LABEL_PATH = "model/class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, "r") as f:
    class_indices = json.load(f)
idx_to_label = {v: k for k, v in class_indices.items()}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"jpg", "jpeg", "png"}


def preprocess_image(filepath):
    img = Image.open(filepath).convert("RGB")
    img = img.resize((128, 128))
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    phone = data.get("phone")
    password = data.get("password")
    if not phone or not password:
        return jsonify({"message": "Phone and password required"}), 400
    if users.find_one({"phone": phone}):
        return jsonify({"message": "User already exists"}), 400
    hashed_pw = generate_password_hash(password)
    users.insert_one({"phone": phone, "password": hashed_pw})
    return jsonify({"message": "Registered successfully"}), 200


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    phone = data.get("phone")
    password = data.get("password")
    if not phone or not password:
        return jsonify({"message": "Phone and password required"}), 400
    user = users.find_one({"phone": phone})
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"message": "Invalid phone or password"}), 401
    access_token = create_access_token(identity=phone)
    return jsonify(access_token=access_token), 200


@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    current_user = get_jwt_identity()
    if "file" not in request.files:
        return jsonify({"message": "No file provided"}), 400
    file = request.files["file"]
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        arr = preprocess_image(filepath)
        preds = model.predict(arr)[0]
        class_id = int(np.argmax(preds))
        label = idx_to_label[class_id]
        confidence = float(preds[class_id])
        record = {
            "user": current_user,
            "filename": filename,
            "result": label,
            "confidence": confidence,
            "created_at": datetime.utcnow(),
            "feedback": None,
            "feedback_notes": None,
            "feedback_at": None
        }
        history.insert_one(record)
        return jsonify({
            "disease": label,
            "confidence": confidence,
            "image_url": f"/image/{filename}"
        }), 200
    return jsonify({"message": "Invalid file format"}), 400


@app.route("/feedback/<record_id>", methods=["POST"])
@jwt_required()
def feedback(record_id):
    data = request.get_json()
    feedback_value = "correct" if data.get("is_correct") else "incorrect"
    rec = history.find_one({"_id": ObjectId(record_id)})
    if not rec:
        return jsonify({"message": "Record not found"}), 404

    history.update_one({"_id": ObjectId(record_id)}, {"$set": {
        "feedback": feedback_value,
        "feedback_at": datetime.utcnow()
    }})

    if feedback_value == "incorrect":
        feedback_dir = os.path.join("feedback_data", rec["result"])
        os.makedirs(feedback_dir, exist_ok=True)
        src_path = os.path.join(app.config["UPLOAD_FOLDER"], rec["filename"])
        dst_path = os.path.join(feedback_dir, rec["filename"])
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

    incorrect_count = history.count_documents({"feedback": "incorrect"})
    if incorrect_count % 10 == 0 and incorrect_count > 0:
        try:
            print(f"⚙️ Retraining triggered after {incorrect_count} incorrect feedbacks...")
            subprocess.Popen(["python", "model/model.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print("Error triggering retraining:", e)

    return jsonify({"message": "Feedback saved successfully"}), 200


# ✅ HISTORY ROUTE
@app.route("/history", methods=["GET"])
@jwt_required()
def get_history():
    current_user = get_jwt_identity()
    records = list(history.find({"user": current_user}).sort("created_at", -1))
    output = []
    for rec in records:
        output.append({
            "id": str(rec["_id"]),
            "filename": rec["filename"],
            "result": rec["result"],
            "confidence": rec["confidence"],
            "created_at": rec["created_at"].isoformat(),
            "feedback": rec.get("feedback"),
            "feedback_at": rec.get("feedback_at").isoformat() if rec.get("feedback_at") else None
        })
    return jsonify(output), 200


# ✅ IMAGE SERVE ROUTE
@app.route("/image/<filename>")
def get_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
