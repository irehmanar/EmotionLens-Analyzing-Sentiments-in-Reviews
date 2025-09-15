# EmotionLens-Analyzing-Sentiments-in-Reviews

## Overview
ReviewIntel is a sentiment analysis project designed to classify Amazon product reviews as positive or negative. By leveraging advanced natural language processing (NLP) techniques, this project aims to extract valuable insights from customer feedback. The analysis helps businesses and researchers understand consumer sentiment and trends effectively.

---

## Dataset
The dataset used for this project consists of Amazon product reviews from the Electronics category. It includes 1,689,188 entries, with:
- Each reviewer having at least 5 reviews.
- Each product having at least 5 reviews.

### Dataset Features:
- **asin**: Product ID.
- **helpful**: Helpfulness rating.
- **overall**: Product rating.
- **reviewText**: Text of the review.
- **reviewTime**: Raw time of the review.
- **reviewerID**: Reviewer ID.
- **reviewerName**: Reviewer name.
- **summary**: Summary of the review.
- **unixReviewTime**: Unix timestamp.

---

## Objectives
1. **Sentiment Classification**: Predict whether a review is positive or negative.
2. **Clustering**: Group similar words per product to identify trends in feedback.
3. **Model Evaluation**: Compare multiple models to determine the best-performing one.

---

## Methodology
### Data Preprocessing
- Text cleaning, stopword removal, and stemming/lemmatization.
- Feature extraction using two methods:
  1. **Bag of Words (BoW)**: Implemented with `CountVectorizer`.
  2. **TF-IDF (Term Frequency-Inverse Document Frequency)**.

### Clustering
- Grouped similar words per product to uncover patterns in feedback.
- Provided deeper insights into customer preferences and complaints.

### Models Used
- **Logistic Regression (LR)**
- **Support Vector Machine (SVM)**
- **Multinomial Naive Bayes (MNB)**
- **AdaBoost**

### Results
- TF-IDF achieved the best results:

| **Model**           | **10k** | **20k** | **30k** | **60k** |
|---------------------|---------|---------|---------|---------|
| Logistic Regression | 88.3%   | 90.0%   | 92.1%   | 93.7%   |
| SVM                 | 87.1%   | 89.2%   | 91.5%   | 93.1%   |
| Multinomial Naive Bayes | 85.6% | 87.5%   | 89.7%   | 91.0%   |
| AdaBoost            | 83.2%   | 85.0%   | 87.1%   | 89.4%   |

---

## Key Features
1. **Preprocessing Pipeline**: Cleans, tokenizes, and vectorizes textual data.
2. **Clustering**: Groups similar words to identify trends and patterns.
3. **Model Comparison**: Benchmarks multiple algorithms for performance.
4. **Saved Model**: The trained Logistic Regression model is saved for deployment.

---

## Installation and Usage
### Prerequisites
1. Python (version 3.8 or higher).
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/reviewintel.git
   cd reviewintel
   ```
2. Preprocess the dataset and train models:
   ```bash
   python preprocess_and_train.py
   ```

---

## Contributions
Contributions are welcome! Please feel free to fork this repository, create feature branches, and submit pull requests.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments
Special thanks to the open-source community and the contributors of Scikit-learn, NLTK, and Pandas for providing excellent tools and libraries to make this project successful.
