# Pumpkin Price Classification using Decision Tree

## 1️⃣ Project Overview
This project aims to classify pumpkin prices into three categories:
- Low
- Medium
- High
using a Decision Tree Classifier based on historical pumpkin market data.

### Goal
Build a machine learning model that:
- Understands factors affecting pumpkin prices
- Classifies prices accurately
- Provides interpretable results (not black-box)


### Why Decision Tree?

We selected Decision Tree because:

- Easy to interpret (you can visualize decisions)
- Works well with categorical + numerical data
- No need for feature scaling
- Provides feature importance
- ❗Downside: prone to overfitting → solved later with tuning

### Dataset
- Rows: 1757
- Columns: 26 → cleaned to 17
['City Name', 'Type', 'Package', 'Variety', 'Sub Variety', 'Date', 'Low Price', 'High Price', 'Mostly Low', 'Mostly High', 'Origin', 'Origin District', 'Item Size', 'Color', 'Unit of Sale', 'Repack', 'Unnamed: 25']
- This project uses the dataset from [Microsoft ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners).

---

### Phase 1: Data Exploration (EDA)
'''python
print(df.head())
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())
'''
### Why?
- Understanding data structure
- Discovering values
- Identifying the type of each feature

### Data Cleaning
Delete all empty columns
'''python
df_cleaned = df.dropna(axis=1, how='all')
'''

### Why?
Because empty columns serve no purpose and cause noise in the model.

### Handling missing values
Categorical
'''python
df_cleaned[col] = df_cleaned[col].fillna('Unknown')
'''
Numerical
'''python
df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
'''
### Why?

Categorical → We preserve data without deletion.
Numerical → median is better than mean because it is resistant to outliers.

### Feature Engineering
Creating Average Price
'''python
df_cleaned['Average_Price'] = (df_cleaned['Low Price'] + df_cleaned['High Price']) / 2
'''
### Why?
Because we have two prices → we need one value that represents the true price.

Convert price to rating
'''python
df_cleaned['Price_Range'] = pd.qcut(df_cleaned['Average_Price'], q=3, labels=['Low', 'Medium', 'High'])
'''

### Why?
We changed the problem from Regression  → Classification. 
We used qcut because it divides the data in a balanced way.

### Encoding
'''python
le = LabelEncoder()
df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))
'''

### Why?
The Decision Tree doesn't understand the text → we need to convert it to numbers.

### Feature & Target Split
'''python
X = df_cleaned.drop(['Average_Price', 'Price_Range', 'Low Price', 'High Price', 'Mostly Low', 'Mostly High'], axis=1)
y = df_cleaned['Price_Range']
y = le.fit_transform(y)
'''

### Why did we delete the columns?
Because they leaked direct price information (data leakage).

---

### Phase 2: Train/Test Split

'''python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''

### Why?
80% training / 20% testing
random_state → Same results every time

---

### Phase 3: Model Building
'''python
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
'''

### Why default?
To see the model's performance without any optimization (baseline).

---

### Phase 4: Model Evaluation
Accuracy
'''python
accuracy = accuracy_score(y_test, y_pred)
'''
Result: Accuracy = 0.97
Classification Report
High   → Precision: 0.94 | Recall: 0.97  
Low    → Precision: 0.99 | Recall: 1.00  
Medium → Precision: 0.98 | Recall: 0.94  
Analysis:
The model is very powerful.
Best performance was in the Low category.
Some confusion exists between Medium and High.

### Confusion Matrix 

<img width="649" height="545" alt="image" src="https://github.com/user-attachments/assets/5466beda-ee37-48cc-90ef-979db2930db2" />

### Notes:
- Low is rated 100% accurately 
- Mistakes only exist between Medium and High
- This is normal because their prices are similar

---
### Phase 5: Decision Tree Visualization
<img width="1570" height="810" alt="image" src="https://github.com/user-attachments/assets/3f8ab38e-3b55-452f-b88f-3571f160ffaf" />

# Most important note:
First split: Package <= 4.5
Explanation: Package is the most important factor in pricing.
# Next:
- City Name
- Variety
- Item Size

# Feature Importance
| Feature   | Importance |
| --------- | ---------- |
| Package   | 0.46       |
| Variety   | 0.15       |
| City Name | 0.10       |
| Item Size | 0.07       |
| Origin    | 0.06       |

# Conclusion:
Packaging has the greatest impact on price.
Location (city) is important.
Size and type have a moderate impact.









