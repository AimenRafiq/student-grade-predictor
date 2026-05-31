# Student Grade Predictor

A machine learning model that predicts a student's grade based on their study habits and academic history.

## About
Built using Python and Scikit-Learn. Trained on a dataset of 200 students using Linear Regression. Achieves a 90% R2 score with a mean error of only 3.8 marks.

## How it works
1. Generates a realistic dataset of 200 students
2. Uses four key features to predict final grade
3. Trains a Linear Regression model on the data
4. Predicts grades for new students based on their inputs

## Features Used
- Study hours per day
- Attendance percentage
- Previous grade
- Sleep hours per night

## Key Findings
- Study hours is the strongest predictor — every extra hour adds ~4.5 marks
- Sleep quality has a meaningful impact on performance
- The model can predict a student's grade within 3.8 marks on average

## Tech Stack
- Python
- Scikit-Learn
- Pandas
- NumPy

## Results
- R2 Score: 0.90
- Mean Absolute Error: 3.8 marks
- Dataset size: 200 students