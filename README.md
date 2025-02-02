# topsis-model-ranking-on-text-sentence-similarity



This project demonstrates how to use the **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** method to evaluate and rank pre-trained models for text sentence similarity. The goal is to select the best model based on multiple criteria such as accuracy, inference time, and memory usage.

---


---

## **Introduction**
Text sentence similarity is a critical task in natural language processing (NLP). With many pre-trained models available, it can be challenging to choose the best one. TOPSIS is a multi-criteria decision-making method that helps rank alternatives based on their similarity to the ideal solution.

This project:
- Evaluates pre-trained models using TOPSIS.
- Ranks models based on accuracy, inference time, and memory usage.
- Provides a clear and reproducible workflow.

---

## **Installation**
To run this project, you need Python and the following libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`



---

## **Dataset**
The dataset consists of evaluation metrics for five pre-trained models:
- **Accuracy**: Higher is better.
- **Inference Time (ms)**: Lower is better.
- **Memory Usage (MB)**: Lower is better.

Here is the dataset:

| Model         | Accuracy | Inference Time (ms) | Memory Usage (MB) |
|---------------|----------|---------------------|-------------------|
| BERT          | 0.92     | 120                 | 400               |
| RoBERTa       | 0.94     | 150                 | 450               |
| DistilBERT    | 0.89     | 80                  | 300               |
| GPT-3         | 0.91     | 200                 | 500               |
| Sentence-BERT | 0.95     | 90                  | 350               |

---

## **TOPSIS Implementation**
TOPSIS is implemented in the following steps:

### **Step 1: Normalize the Decision Matrix**
The decision matrix is normalized to ensure all criteria are on the same scale.

### **Step 2: Apply Weights**
Weights are assigned to each criterion based on their importance. For example:
- Accuracy: 50%
- Inference Time: 30%
- Memory Usage: 20%

### **Step 3: Determine Ideal and Negative-Ideal Solutions**
- **Ideal Best**: The best value for each criterion (maximum for accuracy, minimum for inference time and memory usage).
- **Ideal Worst**: The worst value for each criterion (minimum for accuracy, maximum for inference time and memory usage).

### **Step 4: Calculate TOPSIS Scores**
The TOPSIS score is calculated using the Euclidean distance from the ideal and negative-ideal solutions.

### **Step 5: Rank the Models**
Models are ranked based on their TOPSIS scores. The model with the highest score is the best choice.

---

## **Results**
The results of the TOPSIS analysis are as follows:

### **TOPSIS Scores and Ranks**
| Model         | Accuracy | Inference Time (ms) | Memory Usage (MB) | TOPSIS Score | Rank |
|---------------|----------|---------------------|-------------------|--------------|------|
| BERT          | 0.92     | 120                 | 400               | 0.72         | 3    |
| RoBERTa       | 0.94     | 150                 | 450               | 0.68         | 4    |
| DistilBERT    | 0.89     | 80                  | 300               | 0.85         | 1    |
| GPT-3         | 0.91     | 200                 | 500               | 0.65         | 5    |
| Sentence-BERT | 0.95     | 90                  | 350               | 0.80         | 2    |

### **Visualization**
A bar plot showing the TOPSIS scores for each model:

 <img src="https://github.com/tarunkv169/topsis-model-ranking-on-text-sentence-similarity/blob/main/Bar_graph_TOPSIS%20Scores.png?raw=true" width="70%" height="60%">

---

## **Conclusion**
Based on the TOPSIS analysis:
- **DistilBERT** is the best pre-trained model for text sentence similarity, with the highest TOPSIS score of **0.85**.
- **Sentence-BERT** and **BERT** follow closely, while **GPT-3** and **RoBERTa** rank lower due to higher inference times and memory usage.

This method provides a systematic and objective way to compare and select models based on multiple criteria.

---

## **License**
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as needed.

---

## **How to Run the Code**
1. Clone the repository.
2. Install the required libraries (see [Installation](#installation)).
3. Open the Jupyter Notebook (`topsis_text_similarity.ipynb`).
4. Run the cells to see the results.

---

