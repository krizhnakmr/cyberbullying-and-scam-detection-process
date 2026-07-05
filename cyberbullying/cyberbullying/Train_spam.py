import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import chardet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load CSV file
df = pd.read_csv("spam.csv", encoding="ISO-8859-1")
with open("spam.csv", "rb") as f:
    result = chardet.detect(f.read())
df = pd.read_csv("spam.csv", encoding=result["encoding"])
print(df.head(10))

# Map 'spam' to 1 and 'ham' to 0
df['v1'] = df['v1'].map({'spam': 1, 'ham': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2, random_state=42)

# Tokenize and pad the sequences
max_len = 100  # Choose an appropriate max length for your sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

vocab_size = len(tokenizer.word_index) + 1
embedding_size = 50

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size = 32
epochs = 5

history=model.fit(X_train_seq, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
model.save("CNN.h5")
# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_seq, y_test)
# Plot the accuracy and loss curves
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()



# Extract training and validation accuracy and loss values
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot bar graphs
labels = ['Train Accuracy', 'Validation Accuracy', 'Train Loss', 'Validation Loss']
values = [train_accuracy[-1], val_accuracy[-1], train_loss[-1], val_loss[-1]]

plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.title('Training and Validation Metrics')
plt.ylabel('Metric Value')
plt.show()


y_pred_probs = model.predict(X_test_seq)

# Convert probabilities to binary predictions using a threshold (0.5 in this case)
threshold = 0.5
y_pred = (y_pred_probs > threshold).astype(int)

# Flatten the predictions
y_pred = y_pred.flatten()
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Generate and print classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Generate and print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


print(f"Test Loss of CNN: {loss:.4f}, Test Accuracy of CNN: {accuracy:.4f}")

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Build and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test_vec)

# Calculate and print metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy of Random Forest: {accuracy:.4f}")

# Generate and print classification report
class_report = classification_report(y_test, y_pred)
print("Classification Random Forest:\n", class_report)

# Generate and print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix Random Forest:\n", conf_matrix)
 


new_messages = ["Buy our product now!", "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
new_sequences = pad_sequences(tokenizer.texts_to_sequences(new_messages), maxlen=max_len)
predictions = model.predict(new_sequences)

for message, prediction in zip(new_messages, predictions):
    label = "Spam" if prediction > 0.5 else "Not Spam"
    print(f"Message: {message}\nPrediction: {label}\n")
