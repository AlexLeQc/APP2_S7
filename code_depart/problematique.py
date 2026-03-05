import pathlib

import helpers.dataset as dataset
import numpy
import scipy.signal
import skimage
import sklearn
import sklearn.decomposition
import sklearn.preprocessing
from helpers import analysis, classifier, viz
from matplotlib import pyplot as plt


def extract_std_rgb(img_normalized: numpy.ndarray) -> numpy.ndarray:
    """Calcule l'écart-type pour les canaux RGB (Contraste global)."""
    return numpy.std(img_normalized, axis=(0, 1))


def extract_noise_fft(image: numpy.ndarray) -> float:
    """Calcule le niveau de bruit / énergie via FFT"""
    gray_image = numpy.mean(image, axis=-1)
    fft_image = numpy.fft.fft2(gray_image)
    fft_shifted = numpy.fft.fftshift(fft_image)
    return float(numpy.mean(numpy.abs(fft_shifted)))


def extract_lab_b_peaks(image: numpy.ndarray) -> float:
    """Calcule le nombre de pics dans le canal b de l'espace Lab"""
    image_lab = skimage.color.rgb2lab(image / 255.0)
    scaled_lab = analysis.rescale_lab(image_lab, n_bins=256)
    peaks_b, _ = scipy.signal.find_peaks(scaled_lab[:, :, 2].flatten())
    return float(len(peaks_b))


def extract_std_red(image: numpy.ndarray) -> float:
    """Calcule l'écart-type pour le canal Rouge uniquement"""
    return float(numpy.std(image[:, :, 0]))


def extract_ratio_vh(image: numpy.ndarray) -> float:
    """Calcule le ratio entre les contours verticaux et horizontaux"""
    gray_image = numpy.mean(image, axis=-1)
    edges = skimage.filters.sobel(gray_image)
    vertical_edges = numpy.sum(numpy.abs(edges[:, :-1] - edges[:, 1:]))
    horizontal_edges = numpy.sum(numpy.abs(edges[:-1, :] - edges[1:, :]))
    return float(vertical_edges / (horizontal_edges + 1e-8))


def problematique():
    images = dataset.ImageDataset("data/image_dataset/")

    # -------------------------------------------------------------------------
    # REPRÉSENTATION
    # -------------------------------------------------------------------------
    features_list = []

    for image, label in images:
        noise_level = extract_noise_fft(image)
        lab_b_peaks = extract_lab_b_peaks(image)
        std_red = extract_std_red(image)
        ratio_vh = extract_ratio_vh(image)
        features_list.append([noise_level, lab_b_peaks, std_red, ratio_vh])

    features = numpy.array(features_list, dtype=numpy.float32)

    # -------------------------------------------------------------------------
    # VISUALISATION DE LA REPRÉSENTATION BRUTE
    # -------------------------------------------------------------------------
    feature_names = ["Bruit FFT", "Pics Lab(b)", "Écart-type R", "Ratio V/H"]

    scaler = sklearn.preprocessing.StandardScaler()
    features_scaled = scaler.fit_transform(features)

    representation_scaled = dataset.Representation(
        data=features_scaled, labels=images.labels
    )

    # Distribution des features par classe
    viz.plot_features_distribution(
        representation_scaled,
        n_bins=32,
        title="Distribution des caractéristiques (normalisées)",
        features_names=feature_names,
    )

    # Modèles gaussiens par classe (moyenne, covariance, valeurs/vecteurs propres)
    print("\n--- Modèles gaussiens par classe (espace features normalisées) ---")
    for label in representation_scaled.unique_labels:
        class_data = representation_scaled.get_class(label)
        mean, cov, eigvals, eigvecs = analysis.compute_gaussian_model(class_data)
        print(f"\n----- Classe {label} -----")
        viz.print_gaussian_model(mean, cov, eigvals, eigvecs)

    # -------------------------------------------------------------------------
    # PCA - Décorrélation et réduction de dimensionnalité
    # -------------------------------------------------------------------------
    # On utilise les vecteurs propres globaux (sur toutes les données) comme au labo 1
    mean_global, cov_global, eigvals_global, eigvecs_global = (
        analysis.compute_gaussian_model(features_scaled)
    )
    print("\n--- Modèle gaussien global (pour PCA manuelle) ---")
    viz.print_gaussian_model(mean_global, cov_global, eigvals_global, eigvecs_global)

    # Variance expliquée par composante
    total_variance = numpy.sum(eigvals_global)
    # Les valeurs propres de eigh sont en ordre croissant, on les inverse
    eigvals_sorted = eigvals_global[::-1]
    explained_variance_ratio = eigvals_sorted / total_variance
    print("\n--- Variance expliquée par composante principale ---")
    for i, ratio in enumerate(explained_variance_ratio):
        print(
            f"  PC{i + 1}: {ratio * 100:.1f}%  (cumulé: {numpy.sum(explained_variance_ratio[: i + 1]) * 100:.1f}%)"
        )

    # Projection sur les 3 premières composantes principales (via sklearn PCA)
    pca = sklearn.decomposition.PCA(n_components=3)
    features_pca = pca.fit_transform(features_scaled)

    representation_pca = dataset.Representation(data=features_pca, labels=images.labels)

    # Visualisation 3D avec ellipses
    viz.plot_data_distribution(
        representation_pca,
        title="Représentation 3D PCA (4 features normalisées)",
        xlabel="PC 1",
        ylabel="PC 2",
        zlabel="PC 3",
        show_ellipses=True,
    )

    # -------------------------------------------------------------------------
    # SÉPARATION ENTRAÎNEMENT / TEST
    # -------------------------------------------------------------------------
    from sklearn.model_selection import train_test_split

    train_data, test_data, train_labels, test_labels = train_test_split(
        representation_pca.data,
        representation_pca.labels,
        test_size=0.2,
        random_state=42,
        stratify=representation_pca.labels,
    )
    train_repr = dataset.Representation(data=train_data, labels=train_labels)

    n_classes = len(train_repr.unique_labels)

    # -------------------------------------------------------------------------
    # 1. CLASSIFICATEUR BAYÉSIEN - Modèle Gaussien
    # -------------------------------------------------------------------------
    print("\n\n========== 1. Classificateur Bayésien (Gaussien) ==========")

    aprioris = numpy.array([1 / n_classes] * n_classes)

    cost_matrix = numpy.ones((n_classes, n_classes)) - numpy.eye(n_classes)

    bayes_gauss = classifier.BayesClassifier(
        aprioris=aprioris,
        cost_matrix=cost_matrix,
        density_function=analysis.GaussianPDF,
    )
    bayes_gauss.fit(train_repr)
    pred_bayes = bayes_gauss.predict(test_data)
    pred_bayes_labels = numpy.array([train_repr.unique_labels[p] for p in pred_bayes])
    err_bayes, _ = analysis.compute_error_rate(test_labels, pred_bayes_labels)
    print(f"Taux d'erreur Bayésien (Gaussien) : {err_bayes * 100:.2f}%")
    viz.show_confusion_matrix(
        test_labels,
        pred_bayes_labels,
        train_repr.unique_labels,
        plot=True,
        title="Matrice de confusion - Bayes Gaussien",
    )

    # -------------------------------------------------------------------------
    # 1b. CLASSIFICATEUR BAYÉSIEN - PDF Arbitraire (Histogramme)
    # -------------------------------------------------------------------------
    print("\n========== 1b. Classificateur Bayésien (Histogramme) ==========")

    print("\n--- Recherche du meilleur n_bins pour HistogramPDF ---")
    for n_bins_test in [3, 4, 5, 6, 8, 10]:
        bayes_test = classifier.BayesClassifier(
            aprioris=aprioris,
            cost_matrix=cost_matrix,
            density_function=lambda data, nb=n_bins_test: analysis.HistogramPDF(
                data, n_bins=nb
            ),
        )
        bayes_test.fit(train_repr)
        pred_test = bayes_test.predict(test_data)
        pred_test_labels = numpy.array([train_repr.unique_labels[p] for p in pred_test])
        err_test, _ = analysis.compute_error_rate(test_labels, pred_test_labels)
        print(f"  n_bins={n_bins_test}: taux d'erreur = {err_test * 100:.2f}%")

    # Utiliser le meilleur n_bins trouvé
    bayes_hist = classifier.BayesClassifier(
        aprioris=aprioris,
        cost_matrix=cost_matrix,
        density_function=lambda data: analysis.HistogramPDF(data, n_bins=5),
    )
    bayes_hist.fit(train_repr)
    pred_bayes_hist = bayes_hist.predict(test_data)
    pred_bayes_hist_labels = numpy.array(
        [train_repr.unique_labels[p] for p in pred_bayes_hist]
    )
    err_bayes_hist, _ = analysis.compute_error_rate(test_labels, pred_bayes_hist_labels)
    print(f"Taux d'erreur Bayésien (Histogramme) : {err_bayes_hist * 100:.2f}%")
    viz.show_confusion_matrix(
        test_labels,
        pred_bayes_hist_labels,
        train_repr.unique_labels,
        plot=True,
        title="Matrice de confusion - Bayes Histogramme",
    )

    # -------------------------------------------------------------------------
    # 2. CLASSIFICATEUR K-PPV (KNN)
    # -------------------------------------------------------------------------
    print("\n\n========== 2. Classificateur K-PPV ==========")

    # 2a. KNN sans k-moyennes, k=5
    print("\n2a. KNN (k=5, sans k-moyennes)")
    knn = classifier.KNNClassifier(n_neighbors=5, use_kmeans=False)
    knn.fit(train_repr)
    pred_knn = knn.predict(test_data)
    err_knn, _ = analysis.compute_error_rate(test_labels, pred_knn)
    print(f"Taux d'erreur K-PPV (k=5) : {err_knn * 100:.2f}%")
    viz.show_confusion_matrix(
        test_labels,
        pred_knn,
        train_repr.unique_labels,
        plot=True,
        title="Matrice de confusion - KNN k=5",
    )

    # 2b. KNN avec k-moyennes (quantification vectorielle)
    print("\n2b. KNN (k=1, avec k-moyennes, 5 représentants/classe)")
    knn_kmeans = classifier.KNNClassifier(
        n_neighbors=1, use_kmeans=True, n_representatives=5
    )
    knn_kmeans.fit(train_repr)
    pred_knn_kmeans = knn_kmeans.predict(test_data)
    err_knn_kmeans, _ = analysis.compute_error_rate(test_labels, pred_knn_kmeans)
    print(f"Taux d'erreur K-PPV (k-moyennes, 5 repr.) : {err_knn_kmeans * 100:.2f}%")
    viz.show_confusion_matrix(
        test_labels,
        pred_knn_kmeans,
        train_repr.unique_labels,
        plot=True,
        title="Matrice de confusion - KNN k-moyennes",
    )

    # -------------------------------------------------------------------------
    # 3. RÉSEAU DE NEURONES ARTIFICIELS (RNA)
    # -------------------------------------------------------------------------
    print("\n\n========== 3. Réseau de Neurones ==========")

    rna = classifier.NeuralNetworkClassifier(
        input_dim=train_repr.dim,
        output_dim=len(train_repr.unique_labels),
        n_hidden=3,
        n_neurons=8,
        lr=0.005,
        n_epochs=100,
        batch_size=32,
    )
    rna.fit(train_repr)

    # Sauvegarde du modèle entraîné
    rna.save(pathlib.Path(__file__).parent / "saves/problematique_rna.keras")

    viz.plot_metric_history(rna.history)

    pred_rna_idx = rna.predict(test_data)
    pred_rna_labels = numpy.array([train_repr.unique_labels[i] for i in pred_rna_idx])
    err_rna, _ = analysis.compute_error_rate(test_labels, pred_rna_labels)
    print(f"Taux d'erreur RNA : {err_rna * 100:.2f}%")
    viz.show_confusion_matrix(
        test_labels,
        pred_rna_labels,
        train_repr.unique_labels,
        plot=True,
        title="Matrice de confusion - RNA",
    )

    # -------------------------------------------------------------------------
    # COMPARAISON DES CLASSIFICATEURS
    # -------------------------------------------------------------------------
    print("\n\n========== Comparaison des classificateurs ==========")
    print(f"{'Classificateur':<35} {'Taux d erreur':>15} {'Exactitude':>12}")
    print("-" * 65)

    results = [
        ("Bayes Gaussien", err_bayes),
        ("Bayes Histogramme", err_bayes_hist),
        ("KNN k=5", err_knn),
        ("KNN k-moyennes (5 rep)", err_knn_kmeans),
        ("RNA (3 couches, 16 neu)", err_rna),
    ]

    for name, err in results:
        accuracy = (1 - err) * 100
        print(f"{name:<35} {err * 100:>14.2f}%  {accuracy:>10.2f}%")

    # Calcul F1-score (macro) pour chaque classificateur via sklearn
    from sklearn.metrics import f1_score, precision_score, recall_score

    all_preds = [
        ("Bayes Gaussien", pred_bayes_labels),
        ("Bayes Histogramme", pred_bayes_hist_labels),
        ("KNN k=5", pred_knn),
        ("KNN k-moyennes", pred_knn_kmeans),
        ("RNA", pred_rna_labels),
    ]

    print(f"\n{'Classificateur':<35} {'Précision':>10} {'Rappel':>8} {'F1':>8}")
    print("-" * 65)
    for name, preds in all_preds:
        p = precision_score(test_labels, preds, average="macro", zero_division=0)
        r = recall_score(test_labels, preds, average="macro", zero_division=0)
        f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
        print(f"{name:<35} {p:>10.3f} {r:>8.3f} {f1:>8.3f}")

    plt.show()


if __name__ == "__main__":
    problematique()
