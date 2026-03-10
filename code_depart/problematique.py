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
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


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


def extract_lab_a_peaks(image: numpy.ndarray) -> float:
    """Calcule le nombre de pics dans le canal a de l'espace Lab"""
    image_lab = skimage.color.rgb2lab(image / 255.0)
    scaled_lab = analysis.rescale_lab(image_lab, n_bins=256)
    peaks_a, _ = scipy.signal.find_peaks(scaled_lab[:, :, 1].flatten())
    return float(len(peaks_a))


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


def extract_blue_top_bottom_ratio(image: numpy.ndarray) -> float:
    """Calcule le ratio d'intensité du canal Bleu entre le haut et le bas de l'image"""
    h = image.shape[0]
    top_half = image[: h // 2, :, 2]
    bottom_half = image[h // 2 :, :, 2]

    mean_blue_top = numpy.mean(top_half)
    mean_blue_bottom = numpy.mean(bottom_half)

    return float(mean_blue_top / (mean_blue_bottom + 1e-8))


def extract_mean_saturation_hsv(image: numpy.ndarray) -> float:
    """Calcule la saturation moyenne (espace HSV"""
    image_hsv = skimage.color.rgb2hsv(image / 255.0)
    return float(numpy.mean(image_hsv[:, :, 1]))


def extract_mean_luminance_lab(image: numpy.ndarray) -> float:
    """Calcule la luminance moyenne (canal L*)"""
    image_lab = skimage.color.rgb2lab(image / 255.0)
    return float(numpy.mean(image_lab[:, :, 0]))


def extract_mean_a_lab(image: numpy.ndarray) -> float:
    """Calcule la moyenne de l'axe Vert-Rouge (canal a*)"""
    image_lab = skimage.color.rgb2lab(image / 255.0)
    return float(numpy.mean(image_lab[:, :, 1]))


def extract_mean_b_lab(image: numpy.ndarray) -> float:
    """Calcule la moyenne de l'axe Bleu-Jaune (canal b*)"""
    image_lab = skimage.color.rgb2lab(image / 255.0)
    return float(numpy.mean(image_lab[:, :, 2]))


def compute_permutation_importance(
    clf, X, y, unique_labels, n_repeats=30, random_state=42
):
    """Permutation importance pour un classificateur qui retourne des indices de classe."""
    rng = numpy.random.default_rng(random_state)

    def score(X_eval):
        indices = clf.predict(X_eval)
        preds = numpy.array([unique_labels[i] for i in indices])
        return numpy.mean(preds == y)

    baseline = score(X)
    n_features = X.shape[1]
    importances = numpy.zeros((n_features, n_repeats))

    for j in range(n_features):
        for r in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            importances[j, r] = baseline - score(X_perm)

    return importances.mean(axis=1), importances.std(axis=1)


def problematique():
    images = dataset.ImageDataset("data/image_dataset/")

    # -------------------------------------------------------------------------
    # REPRÉSENTATION
    # -------------------------------------------------------------------------
    features_list = []

    for image, label in images:
        noise_level = extract_noise_fft(image)
        lab_b_peaks = extract_lab_b_peaks(image)
        # lab_a_peaks = extract_lab_a_peaks(image)
        std_red = extract_std_red(image)
        ratio_vh = extract_ratio_vh(image)
        # blue_ratio = extract_blue_top_bottom_ratio(image)
        # mean_sat = extract_mean_saturation_hsv(image)
        # mean_lum = extract_mean_luminance_lab(image)
        # mean_a = extract_mean_a_lab(image)
        # mean_b = extract_mean_b_lab(image)
        features_list.append(
            [
                noise_level,
                lab_b_peaks,
                # lab_a_peaks,
                std_red,
                ratio_vh,
                # blue_ratio,
                # mean_sat,
                # mean_lum,
                # mean_a,
                # mean_b,
            ]
        )

    features = numpy.array(features_list, dtype=numpy.float32)

    # -------------------------------------------------------------------------
    # VISUALISATION DE LA REPRÉSENTATION BRUTE
    # -------------------------------------------------------------------------
    feature_names = [
        "Bruit FFT",
        "Pics Lab(b)",
        # "Pics Lab(a)",
        "Écart-type R",
        "Ratio V/H",
        # "Ratio Bleu Haut/Bas",
        # "Saturation HSV",
        # "Luminance Lab",
        # "A* Lab",
        # "B* Lab",
    ]

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
    )

    # -------------------------------------------------------------------------
    # SÉPARATION ENTRAÎNEMENT / TEST
    # -------------------------------------------------------------------------

    train_data, test_data, train_labels, test_labels = train_test_split(
        features_pca,
        images.labels,
        test_size=0.2,
        random_state=42,
        stratify=images.labels,
    )
    train_repr = dataset.Representation(data=train_data, labels=train_labels)

    n_classes = len(train_repr.unique_labels)

    # -------------------------------------------------------------------------
    # 1. CLASSIFICATEUR BAYÉSIEN - Modèle Gaussien
    # -------------------------------------------------------------------------
    print("\n\n========== 1. Classificateur Bayésien (Gaussien) ==========")

    aprioris = numpy.array([1 / n_classes] * n_classes)

    cost_matrix = numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
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

    # Permutation importance — bayes_gauss entraîné sur les composantes PCA
    pca_names = [f"PC {i + 1}" for i in range(features_pca.shape[1])]
    imp_mean, imp_std = compute_permutation_importance(
        bayes_gauss,
        test_data,
        test_labels,
        train_repr.unique_labels,
        n_repeats=30,
        random_state=42,
    )

    print("\n--- Importance des composantes PCA (Bayes Gaussien) ---")
    for i, name in enumerate(pca_names):
        print(f"{name:<8}: {imp_mean[i]:.4f} +/- {imp_std[i]:.4f}")
    # # -------------------------------------------------------------------------
    # # 1b. CLASSIFICATEUR BAYÉSIEN - PDF Arbitraire (Histogramme)
    # # -------------------------------------------------------------------------
    # print("\n========== 1b. Classificateur Bayésien (Histogramme) ==========")

    # print("\n--- Recherche du meilleur n_bins pour HistogramPDF ---")
    # best_n_bins = None
    # best_error = float("inf")

    # for n_bins_test in [1, 2, 3, 4]:
    #     bayes_test = classifier.BayesClassifier(
    #         aprioris=aprioris,
    #         cost_matrix=cost_matrix,
    #         density_function=lambda data, nb=n_bins_test: analysis.HistogramPDF(
    #             data, n_bins=nb
    #         ),
    #     )
    #     bayes_test.fit(train_repr)
    #     pred_test = bayes_test.predict(test_data)
    #     pred_test_labels = numpy.array([train_repr.unique_labels[p] for p in pred_test])
    #     err_test, _ = analysis.compute_error_rate(test_labels, pred_test_labels)
    #     print(f"  n_bins={n_bins_test}: taux d'erreur = {err_test * 100:.2f}%")

    #     if err_test < best_error:
    #         best_error = err_test
    #         best_n_bins = n_bins_test

    # print(f"Meilleur n_bins trouvé : {best_n_bins}")

    # bayes_hist = classifier.BayesClassifier(
    #     aprioris=aprioris,
    #     cost_matrix=cost_matrix,
    #     density_function=lambda data: analysis.HistogramPDF(data, n_bins=best_n_bins),
    # )
    # bayes_hist.fit(train_repr)
    # pred_bayes_hist = bayes_hist.predict(test_data)
    # pred_bayes_hist_labels = numpy.array(
    #     [train_repr.unique_labels[p] for p in pred_bayes_hist]
    # )
    # err_bayes_hist, _ = analysis.compute_error_rate(test_labels, pred_bayes_hist_labels)
    # print(f"Taux d'erreur Bayésien (Histogramme) : {err_bayes_hist * 100:.2f}%")
    # viz.show_confusion_matrix(
    #     test_labels,
    #     pred_bayes_hist_labels,
    #     train_repr.unique_labels,
    #     plot=True,
    #     title="Matrice de confusion - Bayes Histogramme",
    # )

    # -------------------------------------------------------------------------
    # 2. CLASSIFICATEUR K-PPV (KNN)
    # -------------------------------------------------------------------------
    print("\n\n========== 2. Classificateur K-PPV ==========")
    # KNN avec k-moyennes (quantification vectorielle)
    print("\nKNN (k=1, avec k-moyennes, 5 représentants/classe)")
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
        n_epochs=40,
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
        ("KNN k-moyennes (5 rep)", err_knn_kmeans),
        ("RNA (3 couches, 16 neu)", err_rna),
    ]

    for name, err in results:
        accuracy = (1 - err) * 100
        print(f"{name:<35} {err * 100:>14.2f}%  {accuracy:>10.2f}%")

    all_preds = [
        ("Bayes Gaussien", pred_bayes_labels),
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
