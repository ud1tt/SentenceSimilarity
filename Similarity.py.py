""" roll_number = 102203558 """

import numpy as np

def topsis(decision_matrix, weights, beneficial_flags, model_names):
    denom = np.sqrt((decision_matrix ** 2).sum(axis=0))
    denom[denom == 0] = 1e-9
    normalized_matrix = decision_matrix / denom
    weighted_matrix = normalized_matrix * weights
    n_criteria = decision_matrix.shape[1]
    ideal_best = np.zeros(n_criteria)
    ideal_worst = np.zeros(n_criteria)
    for j in range(n_criteria):
        if beneficial_flags[j]:
            ideal_best[j] = weighted_matrix[:, j].max()
            ideal_worst[j] = weighted_matrix[:, j].min()
        else:
            ideal_best[j] = weighted_matrix[:, j].min()
            ideal_worst[j] = weighted_matrix[:, j].max()
    s_plus = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    s_minus = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
    c_coeffs = s_minus / (s_plus + s_minus)
    return sorted(zip(model_names, c_coeffs), key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    model_names = [
        "BERT-base",
        "Sentence-BERT",
        "UniversalSentenceEncoder",
        "MiniLM",
        "GPT-embeddings"
    ]
    decision_data = np.array([
        [0.84, 200,  420],
        [0.88, 180,  405],
        [0.80, 350,  900],
        [0.83, 400,  117],
        [0.86, 120, 1300]
    ], dtype=float)
    weights = np.array([0.5, 0.3, 0.2], dtype=float)
    beneficial_flags = [True, True, False]
    results = topsis(decision_data, weights, beneficial_flags, model_names)
    print("TOPSIS Ranking:")
    for rank, (model, score) in enumerate(results, start=1):
        print(rank, model, round(score, 4))
