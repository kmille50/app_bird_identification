# 🐦 Bird Species Classifier

A simple web application that classifies bird species from an uploaded image using a deep learning model.

The application is built with **Streamlit** and deployed for free using **Hugging Face Spaces**.

---

## 🚀 Live Demo

You can try the application here:

👉 https://huggingface.co/spaces/kmille50/app_bird_identification

---

## 📸 Features

* Upload an image of a bird
* Automatic bird species prediction
* Displays the predicted species with confidence score
* Simple and interactive interface
* Free hosting with Hugging Face Spaces

---

## 🧠 Model

This project uses the pretrained model:

**chriamue/bird-species-classifier**

The model performs **image classification** to identify bird species from photographs.

---

## 🖥️ Application Preview

Upload an image of a bird and the model will predict the species.

Example output:

```
European Goldfinch ==> 97.42%
```

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Hugging Face Hub
* Transformers
* Pillow

---

## 📂 Project Structure

```
app_bird_identification/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Running the App Locally

### 1️⃣ Clone the repository

```
git clone https://github.com/kmille50/app_bird_identification.git
cd YOUR_REPO
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```
streamlit run app.py
```

---

## 🔑 Environment Variables

The application uses a Hugging Face token stored as an environment variable.

Example:

```
HF_TOKEN=your_huggingface_token
```

When deploying on Hugging Face Spaces, you can add it in:

**Settings → Repository secrets**

---

## ☁️ Deployment

The app is deployed on **Hugging Face Spaces** using the **Streamlit SDK**.

Steps:

1. Create a new Space
2. Select **Streamlit**
3. Upload the project files
4. Add required dependencies in `requirements.txt`

---

## 📜 License

MIT License

---

## 👤 Author

Personal project exploring **deep learning image classification for bird species recognition**.

---
