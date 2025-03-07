import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict

# Load the dataset
file_path = r"D:\CSV-01-12\01-12\DrDoS_DNS.csv" # Replace with the actual file path
df = pd.read_csv(file_path)

# Display basic info
print("Dataset Overview:")
print(df.info())

# Drop unnecessary columns (example: 'Flow ID', 'Timestamp', etc., that are not useful for learning)
drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Handle missing values
df.fillna(df.median(), inplace=True)

# Convert categorical labels (e.g., attack types)
if 'Label' in df.columns:
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split into features and target
X = df.drop(columns=['Label'])  # Features
y = df['Label']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluate the model
accuracy = xgb_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# Q-Learning Implementation
class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (
            reward + self.discount_factor * self.q_table[next_state][best_next_action] - self.q_table[state][action]
        )

# Initialize Q-learning
agent = QLearningAgent(actions=list(range(len(np.unique(y_train)))))

# Simulate Q-learning on the dataset
for index, row in X_train.iterrows():
    state = tuple(row)  # Convert row to a hashable state
    action = agent.choose_action(state)
    reward = 1 if y_train.iloc[index] == action else -1  # Simple reward mechanism
    next_state = tuple(X_train.iloc[np.random.randint(0, len(X_train))])  # Random next state
    agent.update_q_value(state, action, reward, next_state)

print("Q-Learning Training Completed!")
