# Machine Learning Midterm

This is my midterm project for the Machine Learning class.
I used two datasets from scikit-learn: Breast Cancer and Wine.
I tested Decision Tree, KNN and Naive Bayes on them. I also tried K-Means.

## Files
- `breast_cancer.ipynb` – all the work for the breast cancer data
- `wine.ipynb` – all the work for the wine data
- `utils.py` – functions that both notebooks use (so I don't write the same code two times)
- `results/` – CSV files with the results
- `figures/` – all the plots
- `report.docx` – my report
- `AI_DISCLOSURE.md` – how I used AI
- `requirements.txt` – the packages I used

## Data
Both datasets are inside scikit-learn. You don't need to download anything.
- Breast Cancer: `load_breast_cancer()` – 569 rows, 30 features, 0 = malignant, 1 = benign
- Wine: `load_wine()` – 178 rows, 13 features, 3 classes

## What I did
 I loaded the data and looked at it. No missing values, no duplicate rows.

 I split the data. 20% is the test set and it never changes (seed 42). The rest is split into train (70%) and validation (10%) with 3 seeds: 42, 52 and 62.

 I scaled the data with StandardScaler. The scaler only learns from the train data.

 I trained Decision Tree, KNN and Naive Bayes on scaled and unscaled data. That is 66 runs for each dataset.

 I chose the best model with the validation results. Then I tested only that model (seed 42) on the test set.

 I ran K-Means without the labels and checked how close the clusters are to the real classes (ARI).

Total: 132 classifier runs + 12 K-Means runs = 144 runs.

## Results

The best model was KNN with scaling for both datasets.
- Breast Cancer: k=7. Validation F1 was 0.968 and test F1 was 0.971.
- Wine: k=11. Validation F1 was 0.981 and test F1 was 1.0.

What I saw:
- Scaling helped KNN a lot. On wine it went from 0.60 to 0.98. The reason is `proline`, it has very big numbers.
- Scaling did not change the Decision Tree at all.
- Deep trees learned the train data by heart (train F1 = 1.0), but validation did not get better.
- On breast cancer the model missed 3 malignant cases. It made no other mistakes.
- Wine got 36 out of 36 on the test set. But the test set is small, so this does not mean the model is perfect.
- K-Means with scaling found the wine classes very well (ARI 0.91). For breast cancer it was not as good (ARI 0.74).

## How to run

I used Python 3.14.3 on Windows with VS Code.

First install the packages with `py -3.14 -m pip install -r requirements.txt`.

I opened the project folder in VS Code (with the Python and Jupyter extensions) and chose Python 3.14.3 as the kernel.
Open `breast_cancer.ipynb` and click **Restart**, then **Run All**. Then do the same for `wine.ipynb`.

## Problems I had
- My Python 3.14 had no `pip` at first. I fixed it with `py -3.14 -m ensurepip --upgrade`.
- Every time I added a new function to `utils.py`, I had to restart the kernel, because the notebook keeps the old version of the file in memory.
- The value `max_depth=None` is saved as the text "None" in the CSV files. pandas normally turns "None" into an empty cell when it reads a CSV, so I read the files with `keep_default_na=False`.

## Links
- **Student:** Selena Sürücü - 23040102020
- **Repo:** https://github.com/selenasurucu/machine-learning-midterm
- **Video:** https://youtu.be/CbCbHezj6e8