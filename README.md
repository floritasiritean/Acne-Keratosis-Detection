# Acne-Keratosis-Detection
Proiect de Detectare a bolilor dermatologice: Acnee vs. Cheratoză

Proiectul propune implementarea unui clasificator pentru a diferenția imaginile care prezintă leziuni cutanate de tip acnee de cele care prezintă leziuni cutanate de tip cheratoză, folosind algoritmii de învățare supervizată K-Nearest Neighbors (KNN) și Naive Bayes. Acest proiect de învățare automată se concentrează pe utilizarea caracteristicilor extrase din imagini pentru clasificarea acestora.

1. Setul de Date:
Setul de date este compus din două clase principale: imagini cu leziuni cutanate de tip acnee și imagini cu leziuni cutanate de tip cheratoză, având extensia .jpg. Aceste imagini sunt organizate în două directoare distincte, "Acnee" și "Cheratoza". Imaginile au fost preprocesate prin redimensionare la o dimensiune standard, iar etichetele asociate fiecărei imagini sunt atribuite în funcție de numele fișierului.

2 .Algoritmii Utilizați:
Pentru clasificare, am folosit doi algoritmi: K-Nearest Neighbors (KNN) și Naive Bayes (NB), implementați prin intermediul bibliotecii Scikit-learn în Python. Setul de date a fost împărțit întrun set de antrenare (70%), un set de validare (20%) și un set de testare (10%) folosind funcția train_test_split. Clasificatorii au fost antrenați pe setul de antrenare și evaluați pe setul de validare pentru ajustarea parametrilor.

3. Implementarea Codului:
Proiectul este implementat în Python și utilizează biblioteci precum: Scikit-learn pentru implementarea KNN și NB, NumPy pentru manipularea datelor, Matplotlib pentru vizualizare și Seaborn pentru afișarea matricei de confuzie.
