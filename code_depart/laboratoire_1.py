# pylint: disable = missing-function-docstring, missing-module-docstring, wrong-import-position
import os
import pathlib

import matplotlib.pyplot as plt
import numpy
import skimage

# Must be call before any other TensorFlow/Keras import
# Suppress oneDNN custom operations info
# Suppress INFO and WARNING messages from TF (0=all, 1=no INFO, 2=no INFO/WARN, 3=no error)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import helpers.analysis as analysis
import helpers.dataset as dataset
import helpers.viz as viz


def prep_1_distributions_statistiques():
    # L1.P1.1 Visualiser la distribution de points échantillonnés à partir d'une distribution gaussienne
    # -------------------------------------------------------------------------
    mean = [3, -1]
    covariance = [[1, 0], [0, 1]]
    N = 500
    N_values = [10, 20, 50, 100, 500]

    # mean_samples_list = []
    # cov_samples_list = []

    sigma_m_list = []
    sigma_sigma_list = []

    for N in N_values:
        mean_samples_list = []
        cov_samples_list = []
        for i in range(10):
            samples = numpy.random.multivariate_normal(mean, covariance, N)
            mean_samples = numpy.mean(samples, axis=0)
            mean_samples_list.append(mean_samples)
            cov_sample = numpy.cov(samples, rowvar=False)
            cov_samples_list.append(cov_sample)

            # print("mean_samples:", mean_samples)
            # print("cov_sample:", cov_sample)

        mean_samples_array = numpy.array(mean_samples_list)
        cov_samples_array = numpy.array(cov_samples_list)

        mm = numpy.mean(mean_samples_array, axis=0)
        sigma_m = numpy.std(mean_samples_array, axis=0)
        # print("Moyenne des échantillons de moyennes:", mm)
        # print("Écart-type des échantillons de moyennes:", sigma_m)

        mSigma = numpy.mean(cov_samples_array, axis=0)
        sigma_sigma = numpy.std(cov_samples_array, axis=0)
        # print("Moyenne des échantillons de covariances:", mSigma)
        # print("Écart-type des échantillons de covariances:", sigma_sigma)
        sigma_m_list.append(sigma_m)
        sigma_sigma_list.append(sigma_sigma)

    sigma_m_array = numpy.array(sigma_m_list)
    sigma_sigma_array = numpy.array(sigma_sigma_list)

    # Tracé des écart-types des moyennes
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(N_values, sigma_m_array[:, 0], marker="o", label="σm[0]")
    plt.plot(N_values, sigma_m_array[:, 1], marker="o", label="σm[1]")
    plt.xlabel("Taille N")
    plt.ylabel("Écart-type des moyennes (σm)")
    plt.title("Écart-type des moyennes vs N")
    plt.legend()
    plt.grid(True)

    # Tracé des écart-types des éléments de la matrice de covariance
    plt.figure(figsize=(10, 8))
    labels = ["σΣ[0,0]", "σΣ[0,1]", "σΣ[1,0]", "σΣ[1,1]"]
    for i in range(2):
        for j in range(2):
            plt.subplot(2, 2, i * 2 + j + 1)
            plt.plot(N_values, sigma_sigma_array[:, i, j], marker="o")
            plt.xlabel("Taille N")
            plt.ylabel(f"Écart-type {labels[i * 2 + j]}")
            plt.title(f"{labels[i * 2 + j]} vs N")
            plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.scatter(
    #     samples[:, 0], samples[:, 1], color="blue", label="Points échantillonnés"
    # )
    # plt.axhline(0, color="black", linewidth=0.5)
    # plt.axvline(0, color="black", linewidth=0.5)
    # plt.title(f"Distribution Gaussienne 2D (N={N})")
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    # plt.grid(True, linestyle="--", alpha=0.7)
    # plt.legend()
    # plt.show()

    # -------------------------------------------------------------------------


def exercice_2_decorrelation():
    mean = [0, 0, 0]
    covariance = numpy.array([[2, 1, 0], [1, 2, 0], [0, 0, 7]])

    # L1.E2.1 Compléter le code ci-dessus pour calculer les valeurs propres et vecteurs propres de la matrice de covariance
    # -------------------------------------------------------------------------
    # Utilisez la fonction appropriée pour calculer les valeurs propres et vecteurs propres
    # À la place des vecteurs et valeurs propres nulles ci-dessous
    eigenvalues, eigenvectors = numpy.linalg.eigh(covariance)

    print("Exercice 2.1: Calcul des valeurs propres et vecteurs propres")
    viz.print_gaussian_model(mean, covariance, eigenvalues, eigenvectors)
    print("\n")
    # -------------------------------------------------------------------------

    # L1.E2.3 Visualisez les valeurs et vecteurs propres obtenus ainsi que la distribution des points
    # -------------------------------------------------------------------------
    samples = numpy.random.multivariate_normal(mean, covariance, 200)

    representation = dataset.Representation(
        data=samples, labels=numpy.array(["Data"] * samples.shape[0])
    )
    gaussian_model = (mean, covariance, eigenvalues, eigenvectors)
    viz.plot_data_distribution_with_custom_components(
        representation, model=gaussian_model, title="Données échantillonnées"
    )
    # -------------------------------------------------------------------------

    # L1.E2.5 Projetez la représentation des données sur la première composante principale
    # -------------------------------------------------------------------------
    first_principal_component = eigenvectors[:, [2]]
    # Sélectionnez la première composante principale
    decorrelated_samples = analysis.project_onto_new_basis(
        samples, first_principal_component
    )  # Complétez la fonction project_onto_new_basis dans analysis.py

    representation = dataset.Representation(
        data=decorrelated_samples,
        labels=numpy.array(["Data"] * decorrelated_samples.shape[0]),
    )
    # viz.plot_pdf(representation, n_bins=10, title="Projection des données sur la 1er composante")

    plt.figure()
    histogram, bin_edges = numpy.histogram(decorrelated_samples, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(
        bin_centers,
        histogram,
        width=bin_edges[1] - bin_edges[0],
        alpha=0.6,
        color="g",
        label="Données projetées",
    )
    plt.title("Projection des données sur la 1ère composante principale")
    plt.xlabel("Valeur projetée")
    plt.ylabel("Densité de probabilité")
    plt.legend()
    # ---------------------------------------------------------------#----------

    # L1.E2.6 Projetez la représentation des données sur les 2e et 3e composantes principales
    # -------------------------------------------------------------------------
    e23 = eigenvectors[:, [0, 1]]  # Sélectionnez la 2e et 3e composante principale
    reduced_samples = analysis.project_onto_new_basis(
        samples, e23
    )  # Projetez les données sur les 2e et 3e composantes principales

    projected_covariance = numpy.cov(
        reduced_samples, rowvar=False
    )  # Utilisez la fonction appropriée pour calculer la matrice de covariance des données projetées
    projected_eigenvalues, projected_eigenvectors = numpy.linalg.eigh(
        projected_covariance
    )  # Utilisez la fonction appropriée pour calculer les valeurs propres et vecteurs propres des données projetées

    print(
        "Exercice 2.6: Calcul de la matrice de covariance, vecteurs et valeurs propres projetées"
    )
    viz.print_gaussian_model(
        mean[1:3], projected_covariance, projected_eigenvalues, projected_eigenvectors
    )
    print("\n")

    # On reprojettent les données dans l'espace original pour visualiser l'effet de la réduction de dimensionnalité
    reconstruction = analysis.project_onto_new_basis(reduced_samples, e23.T)

    representation = dataset.Representation(
        data=reconstruction, labels=numpy.array(["Data"] * reconstruction.shape[0])
    )
    viz.plot_data_distribution_with_custom_components(
        representation,
        model=gaussian_model,
        title="Données projetées sur les 2e et 3e composantes principales",
    )
    # -------------------------------------------------------------------------

    plt.show()


def exercice_3_visualisation_representation():
    # L1.E3.1 Visualiser la distribution des points pour les 3 classes.
    # -------------------------------------------------------------------------
    data3classes = dataset.MultimodalDataset(
        pathlib.Path(__file__).parent / "data/data_3classes/"
    )
    reprensentation = dataset.Representation(
        data=data3classes.data, labels=data3classes.labels
    )

    viz.plot_data_distribution(
        reprensentation,
        title="Représentation de MultimodalDataset",
        show_components=True,
    )
    # -------------------------------------------------------------------------

    # L1.E3.2 et L1.E3.5 (complétez la fonction compute_gaussian_model dans analysis.py)
    # -------------------------------------------------------------------------
    print("Exercice 3.2: Modèles gaussiens pour chaque classe")
    for class_name in reprensentation.unique_labels:
        class_data = reprensentation.get_class(class_name)

        # Completez la fonction compute_gaussian_model dans analysis.py
        mean, covariance, eigenvalues, eigenvectors = analysis.compute_gaussian_model(
            class_data
        )

        print(f"Classe {class_name}")
        print("-------------------------------")
        viz.print_gaussian_model(mean, covariance, eigenvalues, eigenvectors)
        print("-------------------------------\n")
    # -------------------------------------------------------------------------

    # L1.E3.4 Calculer les variances sur chaque dimension pour la classe C1 ainsi que leur corrélations
    # -------------------------------------------------------------------------
    data_C1 = reprensentation.get_class("C1")
    variances = numpy.var(data_C1, axis=0)
    # Utilisez la fonction appropriée pour calculer les variances
    correlations = numpy.corrcoef(
        data_C1, rowvar=False
    )  # Utilisez la fonction appropriée pour calculer les corrélations
    print("Exercice 3.4: Variances et corrélations pour la classe C1")
    print(f"Variances : {variances}")
    print(f"Corrélations : \n{correlations}")
    # -------------------------------------------------------------------------

    # L1.E3.6 Décorrélez les données basé sur les composantes principales de la classe C1
    # -------------------------------------------------------------------------
    _, _, _, eigenvectors_C1 = analysis.compute_gaussian_model(data_C1)

    # Utilisez la fonction appropriée pour projeter les données sur la nouvelle base
    # Indice: Utilisez la fonction project_onto_new_basis définie précédement pour créer une nouvelle représentation des données
    decorrelated_data = analysis.project_onto_new_basis(
        data3classes.data, eigenvectors_C1
    )
    decorrelated_representation = dataset.Representation(
        data=decorrelated_data, labels=data3classes.labels
    )

    print("\nExercice 3.6: Données décorrelées de la classe C1")
    decorrelated_data_C1 = decorrelated_representation.get_class("C1")
    _, covariance_decorrelated, _, _ = analysis.compute_gaussian_model(
        decorrelated_data_C1
    )
    print(
        f"Matrice de covariance des données décorrelées : \n{covariance_decorrelated}"
    )
    # -------------------------------------------------------------------------

    # L1.E3.7 Est-ce que la décorrélation serait applicable à l'ensemble des classes?
    # -------------------------------------------------------------------------
    viz.plot_data_distribution(
        decorrelated_representation,
        title="Représentation décorrelée de MultimodalDataset",
        show_components=True,
    )
    # -------------------------------------------------------------------------

    plt.show()


def exercice_4_choix_representation():
    images = dataset.ImageDataset(pathlib.Path(__file__).parent / "data/image_dataset/")

    # L1.E4.1 Visualiser quelques images du dataset
    # ---------------------------------------------------------------------
    N = 6
    samples = images.sample(N)
    viz.plot_images(samples, title="Exemples d'images du dataset")
    # -------------------------------------------------------------------------

    # L1.E4.3 Observer l'histograme de couleur d'une image
    # -------------------------------------------------------------------------
    viz.plot_images_histograms(
        samples,
        n_bins=256,
        title="Histogrammes des intensités de pixels RGB",
        x_label="Valeur",
        y_label="Nombre de pixels",
        channel_names=["Red", "Green", "Blue"],
        colors=["r", "g", "b"],
    )
    # -------------------------------------------------------------------------

    # L1.E4.5 Utilisez `scikit-image.color` pour explorer d'autres espaces de couleur (LAB et HSV)
    # -------------------------------------------------------------------------
    samples_lab = []
    samples_hsv = []
    for image, label in samples:
        image_lab = skimage.color.rgb2lab(image / 255.0)
        scaled_lab = analysis.rescale_lab(image_lab, n_bins=256)
        samples_lab.append((scaled_lab, label))

        image_hsv = skimage.color.rgb2hsv(image / 255.0)
        scaled_hsv = analysis.rescale_hsv(image_hsv, n_bins=256)
        samples_hsv.append((scaled_hsv, label))

    # Histogrammes en espace LAB (L=luminosité, A=vert-rouge, B=bleu-jaune)
    # L* est robuste aux changements d'éclairage car il encode la luminance perceptuelle.
    # Les canaux a* et b* encodent la chrominance indépendamment de la luminosité.
    viz.plot_images_histograms(
        samples_lab,
        n_bins=256,
        title="Histogrammes des canaux LAB",
        x_label="Valeur (mise à l'échelle 0-255)",
        y_label="Nombre de pixels",
        channel_names=["L* (luminosité)", "a* (vert→rouge)", "b* (bleu→jaune)"],
        colors=["gray", "r", "b"],
    )

    # Histogrammes en espace HSV (H=teinte, S=saturation, V=luminosité)
    # Le canal H encode la couleur pure, indépendamment de l'éclairage (V) et
    # de l'intensité (S). Des images similaires sous éclairage différent auront
    # donc un histogramme de H similaire mais un V différent → H est plus robuste.
    viz.plot_images_histograms(
        samples_hsv,
        n_bins=256,
        title="Histogrammes des canaux HSV",
        x_label="Valeur (mise à l'échelle 0-255)",
        y_label="Nombre de pixels",
        channel_names=["H (teinte)", "S (saturation)", "V (luminosité)"],
        colors=["purple", "orange", "gray"],
    )
    # -------------------------------------------------------------------------

    # L1.E4.6 Calculer la moyen de chaque canal R, G et B pour chaque classe du dataset
    # =========================================================================
    features = numpy.zeros((len(images), 6))  # 3 moyennes RGB + 3 écarts-types RGB
    features_hsv = numpy.zeros((len(images), 3))  # moyennes H, S, V
    features_lab = numpy.zeros((len(images), 3))  # moyennes L*, a*, b*

    for i, (image, _) in enumerate(images):
        image_norm = (
            image / 255.0
        )  # Normalisation [0,1] → réduit la sensibilité à la luminosité

        # Moyenne de chaque canal R, G, B
        channels_mean = numpy.mean(image_norm, axis=(0, 1))  # [mean_R, mean_G, mean_B]

        # L1.E4.7 Répéter pour une autre métrique de votre choix
        # ---------------------------------------------------------------------
        # Écart-type de chaque canal R, G, B : mesure la variabilité/texture.
        other_feature = numpy.std(image_norm, axis=(0, 1))  # [std_R, std_G, std_B]
        # ---------------------------------------------------------------------

        features[i] = numpy.concatenate((channels_mean, other_feature))

        # Moyennes en espace HSV
        # H (teinte) : discrimine la couleur dominante indépendamment de l'éclairage
        # S (saturation) : faible pour les villes grises, élevée pour forêts/plages
        # V (luminosité) : sensible à l'éclairage → moins discriminant seul
        image_hsv = skimage.color.rgb2hsv(image_norm)
        features_hsv[i] = numpy.mean(image_hsv, axis=(0, 1))  # [mean_H, mean_S, mean_V]

        # Moyennes en espace LAB
        # L* : luminance perceptuelle (insensible à la couleur)
        # a* : axe vert(-) ↔ rouge(+) → discrimine les forêts (négatif)
        # b* : axe bleu(-) ↔ jaune(+) → discrimine les plages (positif = sable)
        image_lab = skimage.color.rgb2lab(image_norm)
        features_lab[i] = numpy.mean(image_lab, axis=(0, 1))  # [mean_L, mean_a, mean_b]

    features = numpy.array(features)
    # =========================================================================

    # L1.E4.8 Étudier si les quelques métriques obtenu sont corrélées, discriminantes, etc.
    # -------------------------------------------------------------------------
    representation_mean = dataset.Representation(
        data=features[:, :3], labels=images.labels
    )
    viz.plot_data_distribution(
        representation_mean,
        title="Distribution des images basée sur les moyennes des canaux RGB",
        xlabel="Rouge",
        ylabel="Verte",
        zlabel="Bleue",
    )

    viz.plot_features_distribution(
        representation_mean,
        n_bins=32,
        title="Histogrammes des moyennes des canaux RGB",
        features_names=["Rouge", "Vert", "Bleu"],
        xlabel="Valeur moyenne",
        ylabel="Nombre d'images",
    )

    # Complétez l'affichage pour la métrique au choix
    representation_other_feature = dataset.Representation(
        data=features[:, 3:], labels=images.labels
    )
    viz.plot_data_distribution(
        representation_other_feature,
        title="Distribution des images basée sur l'écart-type des canaux RGB",
        xlabel="std Rouge",
        ylabel="std Vert",
        zlabel="std Bleu",
    )

    viz.plot_features_distribution(
        representation_other_feature,
        n_bins=32,
        title="Histogrammes de l'écart-type des canaux RGB",
        features_names=["std Rouge", "std Vert", "std Bleu"],
        xlabel="Écart-type",
        ylabel="Nombre d'images",
    )

    # Étude de la corrélations
    representation = dataset.Representation(data=features, labels=images.labels)

    coast_data = representation.get_class("coast")
    variances = numpy.var(coast_data, axis=0)
    correlations = numpy.corrcoef(coast_data, rowvar=False)
    print("Variances et corrélations pour la classe coast")
    print(f"Variances : {variances}")
    print(f"Corrélations : \n{correlations}")
    # -------------------------------------------------------------------------

    # Distribution des images en espace HSV
    # -------------------------------------------------------------------------
    representation_hsv = dataset.Representation(data=features_hsv, labels=images.labels)

    viz.plot_data_distribution(
        representation_hsv,
        title="Distribution des images — moyennes des canaux HSV",
        xlabel="H (teinte)",
        ylabel="S (saturation)",
        zlabel="V (luminosité)",
    )
    viz.plot_features_distribution(
        representation_hsv,
        n_bins=32,
        title="Histogrammes des moyennes HSV par classe",
        features_names=[
            "mean_H (teinte)",
            "mean_S (saturation)",
            "mean_V (luminosité)",
        ],
        xlabel="Valeur moyenne",
        ylabel="Nombre d'images",
    )
    # -------------------------------------------------------------------------

    # Distribution des images en espace LAB
    # -------------------------------------------------------------------------
    representation_lab = dataset.Representation(data=features_lab, labels=images.labels)

    viz.plot_data_distribution(
        representation_lab,
        title="Distribution des images — moyennes des canaux LAB",
        xlabel="L* (luminance)",
        ylabel="a* (vert↔rouge)",
        zlabel="b* (bleu↔jaune)",
    )
    viz.plot_features_distribution(
        representation_lab,
        n_bins=32,
        title="Histogrammes des moyennes LAB par classe",
        features_names=[
            "mean_L* (luminance)",
            "mean_a* (vert↔rouge)",
            "mean_b* (bleu↔jaune)",
        ],
        xlabel="Valeur moyenne",
        ylabel="Nombre d'images",
    )
    # -------------------------------------------------------------------------

    # Représentation 3D combinée : H (HSV) × b* (LAB) × Blue (RGB)
    # Ces 3 features viennent d'espaces différents et capturent des aspects
    # complémentaires :
    #   H (teinte HSV)   : couleur dominante robuste à l'éclairage
    #   b* (LAB)         : axe bleu↔jaune → sépare plage (sable=jaune) vs forêt/ville
    #   mean_Blue (RGB)  : quantité brute de bleu → détecte ciel/océan (plage/ville)
    # -------------------------------------------------------------------------
    mean_H_all = numpy.zeros(len(images))
    mean_bstar_all = numpy.zeros(len(images))
    mean_Blue_all = numpy.zeros(len(images))

    for i, (image, _) in enumerate(images):
        image_norm = image / 255.0

        hsv = skimage.color.rgb2hsv(image_norm)
        mean_H_all[i] = numpy.mean(hsv[:, :, 0])  # canal H uniquement

        lab = skimage.color.rgb2lab(image_norm)
        mean_bstar_all[i] = numpy.mean(lab[:, :, 2])  # canal b* uniquement

        mean_Blue_all[i] = numpy.mean(image_norm[:, :, 2])  # canal Bleu RGB uniquement

    # Normalisation min-max de chaque feature dans [0, 1] pour que les 3 axes
    # soient comparables (b* est dans [-128,127], H dans [0,1], Blue dans [0,1])
    def minmax(x):
        return (x - x.min()) / (x.max() - x.min())

    combined_3d = numpy.column_stack(
        [
            minmax(mean_H_all),
            minmax(mean_bstar_all),
            minmax(mean_Blue_all),
        ]
    )
    representation_combined = dataset.Representation(
        data=combined_3d, labels=images.labels
    )
    viz.plot_data_distribution(
        representation_combined,
        title="Représentation combinée : H (HSV) × b* (LAB) × Blue (RGB)",
        xlabel="mean_H (teinte HSV)",
        ylabel="mean_b* (bleu↔jaune LAB)",
        zlabel="mean_Blue (RGB)",
    )
    # -------------------------------------------------------------------------

    plt.show()


def main():
    # pylint: disable = using-constant-test, multiple-statements

    # if True:
    #     # prep_1_distributions_statistiques()
    #     exercice_2_decorrelation()

    # if True:
    #     exercice_2_decorrelation()
    # if True:
    #     exercice_3_visualisation_representation()
    if True:
        exercice_4_choix_representation()


if __name__ == "__main__":
    main()
