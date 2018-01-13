# Import Pandas Library
import pandas as pd
train_loc = "train.csv"
train = pd.read_csv(train_loc)
# print (train.head())
test_loc = "test.csv"
test = pd.read_csv(train_loc)
# print(test.head())
# print(train.describe())
# print(train.shape)

# print(train["Survived"].value_counts())
# print(train["Survived"].value_counts(normalize = True))
print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True))
print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True))
print("NOTE: Women are more likely to survive")

### Prediciting based on Sex ###
test_Sex = test
test_Sex["Survived"] = 0

test_Sex["Survived"][test_Sex["Sex"] == "female"] = 1
# print(test_Sex.Survived)

### Make a new column, Child ###
train["Child"] = float('NaN')
train["Child"][train["Age"] <18] = 1
train["Child"][train["Age"] >=18] = 0

print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))
print("NOTE: age plays a role but is not strong enough to make a prediciton alone")

print(train["Survived"][train["Pclass"] == 1].value_counts(normalize = True))
print(train["Survived"][train["Pclass"] == 2].value_counts(normalize = True))
print(train["Survived"][train["Pclass"] == 3].value_counts(normalize = True))
print("NOTE: First class passengers are much more likely to survive")

### Predicting based on Class ###
test_Class = test
test_Class["Survived"] = 0

test_Class["Survived"][test_Class["Pclass"] == 1] = 1
# print(test_Class.Survived)

print(train["Survived"][train["Embarked"] == "S"].value_counts(normalize = True))
print(train["Survived"][train["Embarked"] == "C"].value_counts(normalize = True))
print(train["Survived"][train["Embarked"] == "Q"].value_counts(normalize = True))
print("NOTE: Embarked might play a role but is not strong enough to make a prediciton alone")

print(train["Survived"][train["Fare"] <= 70].value_counts(normalize = True))
print(train["Survived"][train["Fare"] >= 70].value_counts(normalize = True))
print("NOTE: fare greater than $70 more likely to survive, but may be strongly correlated to class")
