import os
import tempfile
import re
from glob import glob
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from natsort import natsorted
from nilearn.maskers import NiftiLabelsMasker
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------------
# SETUP & ROBUST PATHS
# -------------------------------
st.set_page_config(page_title="Propagation Mapping Toolbox", layout="wide")
st.title("Propagation Mapping Toolbox (v0.3 - Optimized)")

# BASE_DIR fix: Anchors to the folder containing this script
BASE_DIR = Path(__file__).resolve().parent

# Markdown references
st.markdown("##### Please cite:")
st.markdown("• Dugré, JR. (2025). Propagation Mapping: A Precision Framework for Reconstructing the Neural Circuitry of Brain Maps. *bioRxiv*, DOI: Not Yet")

# -------------------------------
# CACHED DATA LOADING (Modern Streamlit API)
# -------------------------------
@st.cache_resource(show_spinner="Loading Atlas Masker...")
def get_masker(atlas_choice):
    atlas_path = BASE_DIR / "atlases" / f"ATLAS_{atlas_choice}.nii.gz"
    if not atlas_path.exists():
        return None
    return NiftiLabelsMasker(labels_img=str(atlas_path), strategy="mean", resampling_target="labels")

@st.cache_data(show_spinner="Loading Static Files...")
def load_static_data(atlas_choice):
    # Connectome
    conn_path = BASE_DIR / "normative_connectomes" / f"ATLAS_{atlas_choice}_precomputed.csv"
    # Spins
    spin_path = BASE_DIR / "atlases" / "surfaces" / "spins" / f"spins_{atlas_choice}_hungarian1k.csv"
    # Labels
    label_path = BASE_DIR / "atlases" / f"listnames_ATLAS_{atlas_choice}.csv"
    
    if not all([conn_path.exists(), spin_path.exists(), label_path.exists()]):
        return None, None, None
        
    return pd.read_csv(conn_path, index_col=0), pd.read_csv(spin_path, index_col=0), pd.read_csv(label_path)

# -------------------------------
# SESSION STATE
# -------------------------------
if 'tmp_dir' not in st.session_state:
    st.session_state.tmp_dir = tempfile.mkdtemp()

for key in ['masked_df', 'propagation_results', 'atlas_choice_prev']:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------------
# SIDEBAR: DATA INPUT
# -------------------------------
st.sidebar.header("1. Data Input")
uploaded_files = st.sidebar.file_uploader("Upload NIFTI file(s)", type=['nii', 'nii.gz'], accept_multiple_files=True)
folder_path = st.sidebar.text_input("OR Folder Path:")

st.sidebar.header("2. Export Settings")
ext_choice = st.sidebar.selectbox("File Extension", [".csv", ".txt"])
use_header_index = st.sidebar.checkbox("Include Header & Index", value=True)

# -------------------------------
# COLLECT FILES
# -------------------------------
nii_files = []
def clean_name(name):
    return re.sub(r'\.nii(\.gz)?$', '', name)

if uploaded_files:
    for up in uploaded_files:
        tmp_path = os.path.join(st.session_state.tmp_dir, up.name)
        with open(tmp_path, "wb") as f: f.write(up.getbuffer())
        nii_files.append(tmp_path)
elif folder_path and os.path.exists(folder_path):
    nii_files = natsorted(glob(os.path.join(folder_path, '*.nii*')))

st.sidebar.write(f"📂 Files detected: {len(nii_files)}")

def regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5: return np.nan, np.nan, np.nan
    X, y = y_pred[mask].reshape(-1, 1), y_true[mask]
    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)
    return r2_score(y, y_hat), mean_absolute_error(y, y_hat), np.sqrt(mean_squared_error(y, y_hat))

# -------------------------------
# 1. ATLAS SELECTION
# -------------------------------
st.header("1. Atlas Selection")
atlas_choice = st.radio("Choose Atlas:", ("Schaefer7", "Desikan", "Gordon", "Glasser", "Schaefer17"), horizontal=True)

if st.session_state.atlas_choice_prev != atlas_choice:
    st.session_state.masked_df = None
    st.session_state.propagation_results = None
    st.session_state.atlas_choice_prev = atlas_choice

# -------------------------------
# 2. RUN PARCELLATION
# -------------------------------
if nii_files and st.button("▶️ RUN PARCELLATION"):
    with st.spinner("Parcellating images..."):
        masker = get_masker(atlas_choice)
        conn_df, _, _ = load_static_data(atlas_choice)

        if masker and conn_df is not None:
            n_rois = conn_df.shape[0]
            data = []
            prog = st.progress(0)
            for i, f in enumerate(nii_files):
                vec = masker.fit_transform(f).flatten()
                full_vec = np.zeros(n_rois)
                n_detected = min(len(vec), n_rois)
                full_vec[:n_detected] = vec[:n_detected]
                data.append(full_vec)
                prog.progress((i + 1) / len(nii_files))

            st.session_state.masked_df = pd.DataFrame(np.array(data).T)
            st.session_state.masked_df.columns = [clean_name(Path(f).name) for f in nii_files]
            st.success(f"Parcellation Complete! Found {n_rois} ROIs.")
        else:
            st.error("Missing atlas or connectome files in the repository.")

# -------------------------------
# 3. ANALYSIS CONFIG (RAM-Safe Version)
# -------------------------------
if st.session_state.masked_df is not None:
    st.header("2. Analysis Configuration")
    do_std = st.checkbox("Standardize Input (Z-score)", value=True)
    alpha_input = st.number_input("Significance Alpha", value=0.05, min_value=0.001, max_value=0.5)

    if st.button("🚀 LAUNCH PROPAGATION & SPIN TEST"):
        conn_df, spins_df, labels_df = load_static_data(atlas_choice)
        W_full = conn_df.values
        np.fill_diagonal(W_full, 0)
        W_full[np.isinf(W_full)] = 0

        spins = spins_df.values.astype(int)
        nspins = spins.shape[1]
        n_nodes = W_full.shape[0]

        # Align labels
        cortical_mask = labels_df['regions'].astype(str).str.strip().str.lower() == 'cortex'
        cortical_idx = labels_df[cortical_mask].index.values
        subcortical_idx = np.setdiff1d(np.arange(n_nodes), cortical_idx)
        final_label_names = labels_df['Label'].astype(str).tolist()

        # Scale data
        df = st.session_state.masked_df.copy()
        if do_std:
            df = df.apply(lambda x: (x - x.mean()) / x.std())

        results_store = []
        prog = st.progress(0)

        for idx, col in enumerate(df.columns):
            act_raw = df[col].values
            
            # observed
            prod_obs = 0.5 * ((act_raw[:, None] * W_full) + (act_raw[:, None] * W_full).T)
            pred_obs = np.nansum(prod_obs, axis=0)
            r2_f, mae_f, rmse_f = regression_metrics(act_raw, pred_obs)

            # RAM-SAFE STATS: Online Accumulators (Sum and Sum of Squares)
            sum_null = np.zeros((n_nodes, n_nodes))
            sum_sq_null = np.zeros((n_nodes, n_nodes))
            delta_r2_spin = []

            for s in range(nspins):
                permuted = np.zeros(n_nodes)
                # Cortical spin
                permuted[cortical_idx] = act_raw[cortical_idx][spins[:, s]]
                # Subcortical shuffle
                sub_val = act_raw[subcortical_idx].copy()
                np.random.shuffle(sub_val)
                permuted[subcortical_idx] = sub_val

                # Construct null matrix
                p_null = np.zeros((n_nodes, n_nodes))
                # CC & SS
                for i_set in [cortical_idx, subcortical_idx]:
                    rows = np.ix_(i_set, i_set)
                    cc = permuted[i_set][:, None] * W_full[rows]
                    p_null[rows] = 0.5 * (cc + cc.T)
                # CS
                rows_cs = np.ix_(cortical_idx, subcortical_idx)
                rows_sc = np.ix_(subcortical_idx, cortical_idx)
                cs = permuted[cortical_idx][:, None] * W_full[rows_cs]
                p_null[rows_cs] = cs
                p_null[rows_sc] = cs.T

                # Accumulate for Z-score later
                sum_null += p_null
                sum_sq_null += (p_null ** 2)

                # Metrics
                r2_n, _, _ = regression_metrics(act_raw, np.nansum(p_null, axis=0))
                delta_r2_spin.append(r2_f - r2_n)

            # Calculate Stats (Welford-style variance)
            null_mean = sum_null / nspins
            null_var = (sum_sq_null - (sum_null**2 / nspins)) / (nspins - 1)
            null_std = np.sqrt(np.maximum(null_var, 0)) + 1e-10
            
            z_matrix = (prod_obs - null_mean) / null_std
            sig_mask = np.abs(z_matrix) > norm.ppf(1 - alpha_input / 2)
            masked_mat = np.zeros_like(prod_obs)
            masked_mat[sig_mask] = prod_obs[sig_mask]
            
            p_delta = (1 + np.sum(np.array(delta_r2_spin) <= 0)) / (1 + nspins)

            results_store.append({
                'name': col,
                'raw_mat': prod_obs,
                'masked_mat': masked_mat,
                'z_mat': z_matrix,
                'metrics': [r2_f, mae_f, rmse_f, np.mean(delta_r2_spin), p_delta],
                'label_names': final_label_names
            })
            prog.progress((idx + 1) / len(df.columns))

        st.session_state.propagation_results = results_store
        st.success("Propagation and Spin Tests Complete!")

# -------------------------------
# 4. EXPORT
# -------------------------------
if st.session_state.get('propagation_results'):
    st.header("3. Results & Export")
    # (Rest of zip and dataframe logic follows)
    res_summary = []
    for r in st.session_state.propagation_results:
        res_summary.append([r['name']] + r['metrics'])
    
    sum_df = pd.DataFrame(res_summary, columns=['Subject','R2_Full','MAE_Full','RMSE_Full','Mean_Delta_R2','p_Delta'])
    st.dataframe(sum_df)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        sep = ',' if ext_choice == '.csv' else '\t'
        for r in st.session_state.propagation_results:
            name, labs = r['name'], r['label_names']
            for m_type, m_data in zip(['raw','masked','zscore'], [r['raw_mat'], r['masked_mat'], r['z_mat']]):
                save_df = pd.DataFrame(m_data, index=labs, columns=labs) if use_header_index else pd.DataFrame(m_data)
                csv_buf = io.StringIO()
                save_df.to_csv(csv_buf, sep=sep)
                zipf.writestr(f"propagation_maps/{m_type}/{name}_{m_type}{ext_choice}", csv_buf.getvalue())
    
    st.download_button("💾 DOWNLOAD ALL RESULTS", zip_buffer.getvalue(), "Results.zip", "application/zip")
