import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Read data and perform preprocessing
data = pd.read_csv(r"C:\Users\VAISHNAVI\OneDrive\Desktop\PLACES (2).csv")  # Replace with your file path
x = data.drop(columns=["EXAMPLES OF TREES THAT GROW THERE"])
y = data["EXAMPLES OF TREES THAT GROW THERE"]
x = pd.get_dummies(x, columns=["PLACES", "SOIL TYPE"])
scaler = StandardScaler()
x[["PRECIPITATION", "TEMPERATURE", "ALTITUDE"]] = scaler.fit_transform(x[["PRECIPITATION", "TEMPERATURE", "ALTITUDE"]])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Fit the model on all processes
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # Fit the model on all processes

# Input from the user
if rank == 0:
    input_place = input("Enter a place name in capitals: ")
else:
    input_place = None

# Broadcast input_place to all processes
input_place = comm.bcast(input_place, root=0)

# Create input_data on all processes
input_data = pd.DataFrame(0, index=[0], columns=x.columns)
input_data["PLACE_" + input_place] = 1 if rank == 0 else 0
input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

# Scale input data
input_data[["PRECIPITATION", "TEMPERATURE", "ALTITUDE"]] = scaler.transform(input_data[["PRECIPITATION", "TEMPERATURE", "ALTITUDE"]])

# Predict on all processes
predicted_trees = model.predict(input_data)

# Gather predictions to process 0
gathered_predictions = comm.gather(predicted_trees, root=0)

# Process 0 prints the result
if rank == 0:
    print("Predicted Trees for", input_place, ": ", gathered_predictions)
