import torch
def predict_nationality(surname, classifier, vectorizer):
    vectorized_surname, vec_length = vectorizer.vectorize(surname)
    vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(dim=0)
    vec_length = torch.tensor([vec_length], dtype=torch.int64)

    result = classifier(vectorized_surname, vec_length, apply_softmax=True)
    probability_values, indices = result.max(dim=1)

    index = indices.item()
    prob_value = probability_values.item()

    predicted_nationality = vectorizer.nationality_vocab.lookup_index(index)

    return {'nationality': predicted_nationality, 'probability': prob_value, 'surname': surname}

# 要预测的姓氏列表
surnames = ["McMahan", "Nakamoto", "Wan", "Cho"]

# 对每个姓氏进行推理并打印结果
for surname in surnames:
    prediction = predict_nationality(surname, classifier, vectorizer)
    print(f"{prediction['surname']}: Predicted Nationality = {prediction['nationality']}, Confidence = {prediction['probability']:.2%}")
