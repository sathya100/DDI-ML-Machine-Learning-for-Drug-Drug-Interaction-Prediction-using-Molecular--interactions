import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pickle
import xgboost as xgb
from google.cloud import storage

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Drug–Drug Interaction Predictor", layout="wide")
st.title("💊 Drug–Drug Interaction Prediction (RF | XGBoost | GNN)")

# -------------------------------
# Helper Functions
# -------------------------------
def smiles_to_descriptor(smiles):
    """Convert SMILES to numeric descriptors (for RF/XGB)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descs = [d[1](mol) for d in Descriptors.descList]
    return np.array(descs).reshape(1, -1)

def load_from_gcs(bucket_name, blob_name, local_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_name)
    return local_name

# -------------------------------
# Load Models from GCS
# -------------------------------
bucket_name = "your-gcs-bucket-name"

rf_path = load_from_gcs(bucket_name, "models/ddi_rf_model.pkl", "ddi_rf_model.pkl")
xgb_path = load_from_gcs(bucket_name, "models/ddi_xgb_model.json", "ddi_xgb_model.json")
gnn_path = load_from_gcs(bucket_name, "models/ddi_gnn_model1.pt", "ddi_gnn_model1.pt")

# Random Forest
with open(rf_path, 'rb') as f:
    rf_model = pickle.load(f)

# XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(xgb_path)

# Define GNN
class DDI_GNN_Model(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(DDI_GNN_Model, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.fc(x)

gnn_model = DDI_GNN_Model(num_features=128, hidden_dim=256, num_classes=86)
gnn_model.load_state_dict(torch.load(gnn_path, map_location=torch.device('cpu')))
gnn_model.eval()

# -------------------------------
# Streamlit Inputs
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    smiles1 = st.text_input("Enter SMILES for Drug 1", "CCO")
with col2:
    smiles2 = st.text_input("Enter SMILES for Drug 2", "CCN")

if st.button("🔍 Predict Interaction"):
    if smiles1 and smiles2:
        st.write("Processing...")

        # ---- Prepare Feature Vectors ----
        d1 = smiles_to_descriptor(smiles1)
        d2 = smiles_to_descriptor(smiles2)
        if d1 is None or d2 is None:
            st.error("❌ Invalid SMILES input.")
        else:
            features = np.abs(d1 - d2)   # difference feature for pair interaction
            st.subheader("📈 Model Predictions")

            # ---- Random Forest ----
            rf_pred = rf_model.predict(features)[0]
            st.success(f"🌲 Random Forest Prediction: Class {rf_pred}")

            # ---- XGBoost ----
            xgb_pred = xgb_model.predict(features)[0]
            xgb_prob = np.max(xgb_model.predict_proba(features))
            st.info(f"⚡ XGBoost Prediction: Class {xgb_pred} (Confidence: {xgb_prob:.2f})")

            # ---- GNN ----
            # Placeholder: in real use, you'd load molecular graphs and edge_index
            dummy_x = torch.randn(1, 128)
            dummy_edge = torch.tensor([[0], [0]], dtype=torch.long)
            dummy_batch = torch.tensor([0])
            with torch.no_grad():
                gnn_out = gnn_model(dummy_x, dummy_edge, dummy_batch)
                gnn_pred = torch.argmax(gnn_out).item()
                gnn_conf = torch.softmax(gnn_out, dim=1).max().item()
            st.warning(f"🔮 GNN Prediction: Class {gnn_pred} (Confidence: {gnn_conf:.2f})")

            # Combine Results Table
            df = pd.DataFrame({
                "Model": ["Random Forest", "XGBoost", "GNN"],
                "Predicted Class": [rf_pred, xgb_pred, gnn_pred],
                "Confidence": ["-", f"{xgb_prob:.2f}", f"{gnn_conf:.2f}"]
            })
            st.table(df)
    else:
        st.error("Please enter valid SMILES for both drugs.")

# -------------------------------
# Notes
# -------------------------------
st.markdown("""
---
**Note:**  
- Models are securely loaded from your Google Cloud Storage bucket.  
- The GNN prediction here uses a placeholder graph representation (you can replace it with RDKit→PyTorch Geometric conversion).  
- Add your own model class labels for richer interpretation.
""")
