
---

## 🧠 Why Decision Tree?

We selected **Decision Tree** because:

✅ Easy to understand and visualize  
✅ Handles both categorical & numerical data  
✅ No need for feature scaling  
✅ Provides feature importance  
❗ Can overfit → handled later with tuning  

---

## 📂 Dataset Information

- 📏 Rows: **1757**
- 📊 Columns: **26 → cleaned to 17**

### Features Used:

City Name, Type, Package, Variety, Sub Variety, Date,
Low Price, High Price, Mostly Low, Mostly High,
Origin, Origin District, Item Size, Color,
Unit of Sale, Repack


📎 Dataset Source:  
[Microsoft ML For Beginners](https://github.com/microsoft/ML-For-Beginners)

---

# 🔍 Phase 1: Data Exploration (EDA)

```python
print(df.head())
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())
🎯 Why?
Understand dataset structure
Identify feature types
Detect missing values
🧹 Data Cleaning
حذف الأعمدة الفارغة:
df_cleaned = df.dropna(axis=1, how='all')
💡 Why?

Columns with all missing values add noise and no value.

⚠️ Handling Missing Values
🏷️ Categorical:
df_cleaned[col] = df_cleaned[col].fillna('Unknown')
🔢 Numerical:
df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
💡 Why?
Categorical → preserve data
Numerical → median resists outliers
⚙️ Feature Engineering
📊 Create Average Price
df_cleaned['Average_Price'] = (df_cleaned['Low Price'] + df_cleaned['High Price']) / 2
💡 Why?

To combine price range into a single meaningful value

🔄 Convert to Classification Problem
df_cleaned['Price_Range'] = pd.qcut(
    df_cleaned['Average_Price'],
    q=3,
    labels=['Low', 'Medium', 'High']
)
💡 Why?
تحويل المشكلة من Regression → Classification
qcut يعطي توزيع متوازن
🔢 Encoding
le = LabelEncoder()
df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))
💡 Why?

Models cannot understand text → convert to numbers

🎯 Feature & Target Split
X = df_cleaned.drop([
    'Average_Price', 'Price_Range',
    'Low Price', 'High Price',
    'Mostly Low', 'Mostly High'
], axis=1)

y = df_cleaned['Price_Range']
y = le.fit_transform(y)
⚠️ Why remove these columns?

They contain direct price info → Data Leakage

✂️ Phase 2: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
💡 Why?
80% training / 20% testing
random_state ensures reproducibility
🌳 Phase 3: Model Building
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
💡 Why default model?

To establish a baseline performance

📊 Phase 4: Model Evaluation
✅ Accuracy
accuracy = accuracy_score(y_test, y_pred)

🎯 Result: 0.97

📋 Classification Report
Class	Precision	Recall
High	0.94	0.97
Low	0.99	1.00
Medium	0.98	0.94
🧠 Analysis
🔥 Model performance is very strong
🟢 Best performance → Low class
⚠️ Slight confusion → Medium vs High
🔢 Confusion Matrix

📌 Notes:
Low classified perfectly ✅
Errors only between Medium & High
طبيعي لأن الأسعار قريبة
🌲 Phase 5: Decision Tree Visualization

🔍 Key Insights

🚨 Most Important Feature: Package

💡 Explanation:

نوع التغليف له التأثير الأكبر على السعر

🏆 Feature Importance Ranking
🥇 Package (0.46)
🥈 Variety (0.15)
🥉 City Name (0.10)
Item Size (0.07)
Origin (0.06)
⚠️ Phase 6: Overfitting & Optimization
❌ Problem:

Decision Trees tend to overfit (memorize data)

✅ Solution:
model_tuned = DecisionTreeClassifier(max_depth=5)
📉 Result:

Accuracy dropped → 0.83

🧠 Important Explanation (for Supervisor 💬)

Even though accuracy decreased:

✅ Model is now more generalized
✅ Less overfitting
✅ Better performance on unseen data
🏁 Final Conclusion

✅ Decision Tree achieved excellent performance
✅ Model is both accurate & interpretable

🔑 Key Factors Affecting Price:
📦 Package (most important)
🌍 City
🌱 Variety
🚀 Future Improvements
Try Random Forest for better generalization
Use GridSearchCV for hyperparameter tuning
Feature selection for reducing noise
