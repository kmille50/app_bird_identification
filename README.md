
# 🦅 ttt_images_oiseaux

**ttt_images_oiseaux** est un outil de traitement d'image automatisé conçu spécifiquement pour les photographes animaliers utilisant des boîtiers Fujifilm. Il combine la puissance de l'**Intelligence Artificielle (YOLOv8)** et la précision du **traitement RAW 16-bit** pour isoler un oiseau, améliorer sa netteté et optimiser le rendu global de l'image sans perte de qualité.

---

## ✨ Fonctionnalités

* **Lecture Native .RAF** : Support complet des fichiers RAW Fujifilm via `rawpy`.
* **Identification par IA** : Segmentation précise du sujet (oiseau) grâce à **YOLOv8-seg**.
* **Workflow 16-bit Haute Fidélité** : Traitement intégral en 16 bits (ou float32) pour préserver la plage dynamique originale.
* **Netteté Sélective** : Application d'un filtre *Unsharp Mask* uniquement sur l'oiseau détecté (évite de faire monter le bruit dans l'arrière-plan).
* **Optimisation Adaptative (CLAHE)** : Amélioration intelligente du contraste et de la luminosité via l'espace couleur Lab.
* **Sortie TIFF Master** : Exportation en format TIFF 16-bit sans perte, prêt pour l'impression ou la retouche finale.

---

## 🛠️ Installation

### 1. Cloner le répertoire

```bash
git clone https://github.com/kmille50/ttt_images_oiseaux.git
cd ttt_images_oiseaux

```

### 2. Installer les dépendances

Il est fortement recommandé d'utiliser un environnement virtuel.

```bash
pip install rawpy opencv-python numpy pillow ultralytics

```

---

## 🚀 Utilisation

1. Placez vos fichiers `.RAF` dans le dossier data/inputs du projet.
2. Modifiez les variables `fichier_entree_raf` et `fichier_sortie_tiff` dans le notebook `ttt_image.ipynb`.
3. Lancez le notebook

*Note : Au premier lancement, le script téléchargera automatiquement le modèle `yolov8s-seg.pt` (environ 22 Mo).*

---

## 🧠 Pipeline de Traitement

Le script suit une logique de développement "Digital Darkroom" rigoureuse :

1. **Démosaïçage** : Conversion des données brutes du capteur en image 16-bit.
2. **Inférence IA** : Création d'un masque binaire par segmentation d'instance.
3. **Filtrage Spatial** : Augmentation de l'acutance locale sur le masque.
4. **Égalisation d'Histogramme** : Ajustement de la luminance (canal L) pour déboucher les ombres et gérer les hautes lumières.
5. **Encodage** : Écriture du fichier TIFF avec compression sans perte (Deflate).

---

## 📋 Pré-requis techniques

* **Python 3.8+**
* **OpenCV 4.x** (avec support 16-bit)
* **GPU (Optionnel)** : Si une carte NVIDIA est détectée avec CUDA, YOLOv8 l'utilisera automatiquement pour une détection plus rapide.

---

## ⚖️ Paramétrage avancé

Vous pouvez ajuster les résultats dans les fonctions suivantes :

* **Netteté** : Modifiez les coefficients de `cv2.addWeighted` dans `traiter_image_haute_qualite`.
* **Contraste** : Ajustez `clipLimit` (valeur typique entre 2.0 et 4.0) dans l'objet `CLAHE`.
* **Sensibilité IA** : Modifiez `conf=0.4` dans `model.predict` pour détecter des oiseaux plus petits ou plus lointains.

---
