import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Create our own dataset
np.random.seed(42)
n = 200

study_hours = np.random.uniform(1, 10, n)
attendance = np.random.uniform(50, 100, n)
prev_grade = np.random.uniform(40, 100, n)
sleep_hours = np.random.uniform(4, 9, n)

# Grade is influenced by all these factors
grade = (
    study_hours * 4.5 +
    attendance * 0.3 +
    prev_grade * 0.3 +
    sleep_hours * 1.5 +
    np.random.normal(0, 5, n)
).clip(0, 100)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "prev_grade": prev_grade,
    "sleep_hours": sleep_hours,
    "grade": grade
})

print(f"Dataset size: {len(df)} students")
print(f"Average grade: {df['grade'].mean():.1f}")
print(f"Highest grade: {df['grade'].max():.1f}")
print(f"Lowest grade: {df['grade'].min():.1f}")
print()

# Step 2: Split into training and testing
features = ["study_hours", "attendance", "prev_grade", "sleep_hours"]
X = df[features]
y = df["grade"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Test accuracy
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Absolute Error: {mae:.1f} marks")
print(f"R2 Score: {r2:.2f}")
print()

# Step 5: See which factors matter most
print("--- Factor Importance ---")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.2f}")

print()

# Step 6: Predict for a new student
print("--- Predict a new student ---")
new_student = pd.DataFrame([{
    "study_hours": 7,
    "attendance": 85,
    "prev_grade": 75,
    "sleep_hours": 7
}])
predicted = model.predict(new_student)[0]
print(f"A student who studies 7hrs, attends 85%, had 75 prev grade, sleeps 7hrs")
print(f"Predicted grade: {predicted:.1f}")