import pathlib

import helpers.dataset as dataset
import numpy
import skimage
import skimage.color
import skimage.filters
import sklearn
import sklearn.decomposition
import sklearn.preprocessing
from helpers import analysis, classifier, viz
from matplotlib import pyplot as plt

_HERE = pathlib.Path(__file__).parent


def extract_std_rgb(img_normalized: numpy.ndarray) -> numpy.ndarray:
    """Calcule l'écart-type pour les canaux RGB (Contraste global)."""
    return numpy.std(img_normalized, axis=(0, 1))


def extract_noise_fft(image: numpy.ndarray) -> float:
    """Calcule le niveau de bruit / énergie via FFT"""
    gray_image = numpy.mean(image, axis=-1)
    fft_image = numpy.fft.fft2(gray_image)
    fft_shifted = numpy.fft.fftshift(fft_image)
    return float(numpy.mean(numpy.abs(fft_shifted)))


def extract_lab_b_mean(image: numpy.ndarray) -> float:
    """Calcule la couleur moyenne sur l'axe Bleu-Jaune.
    - Côte  : b moyen élevé (Sable jaune dominant)
    - Forêt : b moyen faible/négatif (Végétation verte/brune)
    - Ville : b moyen proche de zéro (Béton gris)
    """
    if image.size == 0:
        return 0.0

    # On normalise en float pour skimage
    img_norm = image / 255.0 if image.max() > 1.0 else image
    image_lab = skimage.color.rgb2lab(img_norm)

    # On prend la moyenne du canal b* (index 2)
    return float(numpy.mean(image_lab[:, :, 2]))


def extract_std_red(image: numpy.ndarray) -> float:
    """Calcule l'écart-type pour le canal Rouge uniquement"""
    return float(numpy.std(image[:, :, 0]))


def extract_ratio_vh(image: numpy.ndarray) -> float:
    """Ratio énergie Sobel vertical / horizontal.

    - Ville  : dominance de lignes verticales (bâtiments) → ratio >> 1
    - Côte   : horizon dominant (ligne horizontale) → ratio < 1
    - Forêt  : isotrope (textures sans direction) → ratio ≈ 1
    """
    if image.size == 0:
        return 1.0
    gray = (
        numpy.mean(image, axis=-1) / 255.0
        if image.max() > 1.0
        else numpy.mean(image, axis=-1)
    )
    edges_v = numpy.abs(skimage.filters.sobel_v(gray))
    edges_h = numpy.abs(skimage.filters.sobel_h(gray))
    return float(numpy.sum(edges_v) / (numpy.sum(edges_h) + 1e-8))


def problematique():
    images = dataset.ImageDataset(_HERE / "data/image_dataset/")

    # TODO Problématique: Générez une représentation des images appropriée
    # pour la classification comme dans le laboratoire 1.
    # -------------------------------------------------------------------------
    features_list = []

    for image, label in images:
        # Extraction via les fonctions séparées (basées sur l'image brute 0-255)
        noise_level = extract_noise_fft(image)
        lab_b_mean = extract_lab_b_mean(image)
        std_red = extract_std_red(image)
        ratio_vh = extract_ratio_vh(image)

        # Assemblage du vecteur de caractéristiques (4 dimensions)
        features_list.append([noise_level, lab_b_mean, std_red, ratio_vh])

    features = numpy.array(features_list, dtype=numpy.float32)
    # -------------------------------------------------------------------------

    # TODO: Problématique: Visualisez cette représentation
    # -------------------------------------------------------------------------
    feature_names = ["Bruit", "Moyenne Lab(b)", "Écart R", "Ratio Vert/Horiz"]

    scaler = sklearn.preprocessing.StandardScaler()
    features_scaled = scaler.fit_transform(features)

    representation_scaled = dataset.Representation(
        data=features_scaled, labels=images.labels
    )

    viz.plot_features_distribution(
        representation_scaled,
        n_bins=32,
        title="Distribution des caractéristiques (Normalisées)",
        features_names=feature_names,
    )

    pca = sklearn.decomposition.PCA(n_components=3)
    features_pca = pca.fit_transform(features_scaled)
    # ----------------------

    representation_pca = dataset.Representation(data=features_pca, labels=images.labels)

    viz.plot_data_distribution(
        representation_pca,
        title="Représentation 3D (PCA sur les 4 caractéristiques normalisées)",
        xlabel="PC 1",
        ylabel="PC 2",
        zlabel="PC 3",
    )

    plt.show()
    # -------------------------------------------------------------------------

    # TODO: Problématique: Comparez différents classificateurs sur cette
    # représentation, comme dans le laboratoire 2 et 3.
    # -------------------------------------------------------------------------
    print("\n--- Évaluation des classificateurs ---")

    # 1. Classificateur Bayésien
    print("\n1. Classificateur Bayésien")
    bayes = classifier.BayesClassifier(density_function=analysis.GaussianPDF)
    bayes.fit(representation_pca)
    pred_bayes = bayes.predict(representation_pca.data)
    pred_bayes_labels = numpy.array(
        [representation_pca.unique_labels[p] for p in pred_bayes]
    )
    err_bayes, _ = analysis.compute_error_rate(
        representation_pca.labels, pred_bayes_labels
    )
    print(f"Taux d'erreur Bayésien : {err_bayes * 100:.2f}%")

    # 2. Classificateur K-PPV (KNN)
    print("\n2. Classificateur K-PPV")
    # Choix préliminaire: k=5, pas de kmeans pour l'instant
    knn = classifier.KNNClassifier(n_neighbors=5, use_kmeans=False)
    knn.fit(representation_pca)
    pred_knn = knn.predict(representation_pca.data)
    err_knn, _ = analysis.compute_error_rate(representation_pca.labels, pred_knn)
    print(f"Taux d'erreur K-PPV : {err_knn * 100:.2f}%")

    # 3. Réseau de Neurones Artificiels (RNA)
    print("\n3. Réseau de Neurones")
    # Choix préliminaire: 1 couche cachée de 8 neurones
    rna = classifier.NeuralNetworkClassifier(
        input_dim=representation_pca.dim,
        output_dim=len(representation_pca.unique_labels),
        n_hidden=2,
        n_neurons=8,
        lr=0.01,
        n_epochs=50,
        batch_size=16,
    )
    rna.fit(representation_pca)
    pred_rna_idx = rna.predict(representation_pca.data)
    pred_rna_labels = numpy.array(
        [representation_pca.unique_labels[i] for i in pred_rna_idx]
    )
    err_rna, _ = analysis.compute_error_rate(representation_pca.labels, pred_rna_labels)
    print(f"Taux d'erreur RNA : {err_rna * 100:.2f}%")
    # -------------------------------------------------------------------------


if __name__ == "__main__":
    problematique()
