import os
import tempfile
import re
from glob import glob
from pathlib import Path
import io
import zipfile

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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------------
# SETUP
# -------------------------------
st.set_page_config(page_title="Propagation Mapping Toolbox", layout="wide")
st.title("Propagation Mapping Toolbox (v0.2)")

# Markdown references with DOI hyperlinks
st.markdown("##### Please cite:")
st.markdown("• Dugré, JR. (2025). Propagation Mapping: A Precision Framework for Reconstructing the Neural Circuitry of Brain Maps. *bioRxiv*, DOI: Not Yet")
st.markdown(
    """
    see also:<br>
    • Cocuzza, Sanchez-Romero, et Cole (2022). *STAR Protoc*, DOI: <a href="https://pubmed.ncbi.nlm.nih.gov/27723746/" target="_blank">10.1016/j.xpro.2021.101094</a><br>
    • Cole, Ito, Bassett et Schultz (2016). *Nature Neurosci*, DOI: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8808261/" target="_blank">10.1038/nn.4406</a>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# BASE DIRECTORY
# -------------------------------
BASE_DIR = Path(__file__).parent

# Framework image
framework_img_path = BASE_DIR / "miscellaneous" / "Framework.png"
st.image(
    str(framework_img_path),
    caption=("Propagation Mapping is a new precision framework aiming at reconstructing "
             "the neural circuitry that explains the spatial organization of human brain maps. "
             "This method assumes that regional measures can be best understood as a dot product "
             "between activity and functional connectivity. Uploading your NIfTI file will predict "
             "the spatial pattern of your uploaded map and return 1) the observed (raw) parcellated map, "
             "2) the predicted map, and 3) a propagation map, which reflects the circuitry predicting your "
             "statistical map, prior to summation."),
    width=800
)

# -------------------------------
# SESSION STATE
# -------------------------------
if 'tmp_dir' not in st.session_state:
    st.session_state.tmp_dir = tempfile.mkdtemp()  # store as string

for key in ['masked_df', 'standardized_df', 'propagation_results', 'atlas_choice_prev']:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------------
# SIDEBAR: DATA INPUT
# -------------------------------
st.sidebar.header("1. Data Input")
uploaded_files = st.sidebar.file_uploader(
    "Upload NIFTI file(s)",
    type=['nii', 'nii.gz'],
    accept_multiple_files=True
)
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
        tmp_path = Path(st.session_state.tmp_dir) / up.name  # ✅ wrap here
        with open(tmp_path, "wb") as f:
            f.write(up.getbuffer())
        nii_files.append(tmp_path)
elif folder_path and os.path.exists(folder_path):
    nii_files = natsorted(glob(os.path.join(folder_path, '*.nii*')))

st.sidebar.write(f"📂 Files detected: {len(nii_files)}")

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        return np.nan, np.nan, np.nan
    X, y = y_pred[mask].reshape(-1, 1), y_true[mask]
    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)
    return r2_score(y, y_hat), mean_absolute_error(y, y_hat), np.sqrt(mean_squared_error(y, y_hat))

# -------------------------------
# ATLAS SELECTION
# -------------------------------
st.header("1. Atlas Selection")
atlas_choice = st.radio(
    "Choose Atlas:",
    ("Schaefer7", "Desikan", "Gordon", "Glasser", "Schaefer17"),
    horizontal=True
)

if st.session_state.atlas_choice_prev != atlas_choice:
    st.session_state.masked_df = None
    st.session_state.propagation_results = None
    st.session_state.atlas_choice_prev = atlas_choice

# -------------------------------
# RUN PARCELLATION
# -------------------------------
if nii_files and st.button("▶️ RUN PARCELLATION"):
    with st.spinner("Parcellating images..."):
        atlas_path = BASE_DIR / "atlases" / f"ATLAS_{atlas_choice}.nii.gz"
        conn_path = BASE_DIR / "normative_connectomes" / f"ATLAS_{atlas_choice}_precomputed.csv"

        if conn_path.exists():
            conn_df = pd.read_csv(conn_path, index_col=0)
            n_rois = conn_df.shape[0]

            masker = NiftiLabelsMasker(
                labels_img=str(atlas_path),
                strategy="mean",
                resampling_target="labels"
            )

            progress_text = st.empty()
            prog = st.progress(0)
            data = []

            for i, f in enumerate(nii_files):
                progress_text.text(f"Parcellating file {i+1} / {len(nii_files)}")
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
            st.error(f"Connectome not found at: {conn_path}")

# -------------------------------
# ANALYSIS CONFIG
# -------------------------------
if st.session_state.masked_df is not None:
    st.header("2. Analysis Configuration")
    do_std = st.checkbox("Standardize Input (Z-score)", value=True)
    alpha = st.number_input("Significance Alpha", value=0.05, min_value=0.001, max_value=0.5)

    if st.button("🚀 LAUNCH PROPAGATION & SPIN TEST"):
        conn_path = BASE_DIR / "normative_connectomes" / f"ATLAS_{atlas_choice}_precomputed.csv"
        W_full = pd.read_csv(conn_path, index_col=0).values
        np.fill_diagonal(W_full, 0)
        W_full[np.isinf(W_full)] = 0

        # Spins
        spin_path = BASE_DIR / "atlases" / "surfaces" / "spins" / f"spins_{atlas_choice}_hungarian1k.csv"
        spins_df = pd.read_csv(spin_path, index_col=0)
        spins = spins_df.values.astype(int)
        nspins = spins.shape[1]

        # Labels
        label_path = BASE_DIR / "atlases" / f"listnames_ATLAS_{atlas_choice}.csv"
        labels_df = pd.read_csv(label_path)
        n_nodes_connectome = W_full.shape[0]

        # Align labels with connectome
        if len(labels_df) > n_nodes_connectome:
            labels_df = labels_df.iloc[:n_nodes_connectome].copy()
        elif len(labels_df) < n_nodes_connectome:
            pad_n = n_nodes_connectome - len(labels_df)
            pad_df = pd.DataFrame({
                'Label': [f'ROI_{i+1+len(labels_df)}' for i in range(pad_n)],
                'regions': ['cortex'] * pad_n
            })
            labels_df = pd.concat([labels_df, pad_df], ignore_index=True)

        cortical_mask = labels_df['regions'].astype(str).str.strip().str.lower() == 'cortex'
        cortical_idx = labels_df[cortical_mask].index.values
        subcortical_idx = labels_df[~cortical_mask].index.values
        final_label_names = labels_df['Label'].astype(str).tolist()

        # Align data
        df = st.session_state.masked_df.copy()
        if df.shape[0] < n_nodes_connectome:
            pad = np.zeros((n_nodes_connectome - df.shape[0], df.shape[1]))
            df = pd.DataFrame(np.vstack([df.values, pad]), columns=df.columns)
        elif df.shape[0] > n_nodes_connectome:
            df = pd.DataFrame(df.values[:n_nodes_connectome, :], columns=df.columns)

        if do_std:
            df = df.apply(lambda x: (x - x.mean()) / x.std())

        results_store = []
        progress_text = st.empty()
        prog = st.progress(0)

        # ================= LOOP =================
        for idx, col in enumerate(df.columns):
            progress_text.text(f"Processing subject {idx+1} / {len(df.columns)}")
            act_raw = df[col].values
            if len(act_raw) != W_full.shape[0]:
                tmp = np.zeros(W_full.shape[0])
                m = min(len(act_raw), W_full.shape[0])
                tmp[:m] = act_raw[:m]
                act_raw = tmp

            n_nodes = len(act_raw)
            # observed
            prod_obs = 0.5 * ((act_raw[:, None] * W_full) + (act_raw[:, None] * W_full).T)
            pred_obs = np.nansum(prod_obs, axis=0)
            r2_f, mae_f, rmse_f = regression_metrics(act_raw, pred_obs)

            # nulls
            null_edgewise = np.full((nspins, n_nodes, n_nodes), np.nan)
            delta_r2_spin = []

            for s in range(nspins):
                permuted = np.zeros(n_nodes)
                spin_idx = spins[:, s].copy()
                spin_idx = np.clip(spin_idx, 0, len(cortical_idx)-1)
                permuted[cortical_idx] = act_raw[cortical_idx][spin_idx]
                sub_val = act_raw[subcortical_idx].copy()
                np.random.shuffle(sub_val)
                permuted[subcortical_idx] = sub_val

                p_null = np.zeros((n_nodes, n_nodes))
                for i_set in [cortical_idx, subcortical_idx]:
                    rows = np.ix_(i_set, i_set)
                    cc = permuted[i_set][:, None] * W_full[rows]
                    p_null[rows] = 0.5 * (cc + cc.T)
                rows_cs = np.ix_(cortical_idx, subcortical_idx)
                rows_sc = np.ix_(subcortical_idx, cortical_idx)
                cs = permuted[cortical_idx][:, None] * W_full[rows_cs]
                p_null[rows_cs] = cs
                p_null[rows_sc] = cs.T

                null_edgewise[s] = p_null
                pred_null = np.nansum(p_null, axis=0)
                r2_n, _, _ = regression_metrics(act_raw, pred_null)
                delta_r2_spin.append(r2_f - r2_n)

            null_mean = np.nanmean(null_edgewise, axis=0)
            null_std = np.nanstd(null_edgewise, axis=0, ddof=1) + 1e-10
            z_matrix = (prod_obs - null_mean) / null_std
            sig_mask = np.abs(z_matrix) > norm.ppf(1 - alpha / 2)
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
# EXPORT RESULTS
# -------------------------------
if st.session_state.get('propagation_results'):
    st.header("3. Results & Export")

    residuals_dict = {}
    res_summary = []

    for r in st.session_state.propagation_results:
        name = r['name']
        r2_f, mae_f, rmse_f, mean_delta, p_delta = r['metrics']
        masked_vec = np.nansum(r['masked_mat'], axis=0)
        act_vec = st.session_state.masked_df[name].values
        r2_m, mae_m, rmse_m = regression_metrics(act_vec, masked_vec)
        residuals_dict[name] = act_vec - masked_vec
        res_summary.append([name, r2_f, mae_f, rmse_f, mean_delta, p_delta, r2_m, mae_m, rmse_m])

    sum_df = pd.DataFrame(
        res_summary,
        columns=['Subject','R2_Full','MAE_Full','RMSE_Full','Mean_Delta_R2','p_Delta','R2_Masked','MAE_Masked','RMSE_Masked']
    )
    st.dataframe(sum_df)

    # Residuals DataFrame
    label_names = st.session_state.propagation_results[0]['label_names']
    residuals_df = pd.DataFrame(residuals_dict)
    n_labels = len(label_names)
    n_rows = residuals_df.shape[0]

    if n_rows > n_labels:
        residuals_df = residuals_df.iloc[:n_labels, :]
    elif n_rows < n_labels:
        pad = pd.DataFrame(np.nan, index=range(n_labels-n_rows), columns=residuals_df.columns)
        residuals_df = pd.concat([residuals_df, pad], axis=0)

    residuals_df.index = label_names

    # Create ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        sep = ',' if ext_choice == '.csv' else '\t'
        for r in st.session_state.propagation_results:
            name, labs = r['name'], r['label_names']
            for m_type, m_data in zip(['raw','masked','zscore'], [r['raw_mat'], r['masked_mat'], r['z_mat']]):
                save_df = pd.DataFrame(m_data)
                if use_header_index:
                    save_df.index = labs
                    save_df.columns = labs
                zip_path = f"propagation_maps/{m_type}/{name}_{m_type}{ext_choice}"
                csv_buffer = io.StringIO()
                save_df.to_csv(csv_buffer, sep=sep, index=use_header_index, header=use_header_index)
                zipf.writestr(zip_path, csv_buffer.getvalue())

        # summary
        summary_buffer = io.StringIO()
        sum_df.to_csv(summary_buffer, index=False)
        zipf.writestr("propagation_maps/summary_metrics.csv", summary_buffer.getvalue())

        # residuals
        resid_buffer = io.StringIO()
        residuals_df.to_csv(resid_buffer, sep=sep)
        zipf.writestr("propagation_maps/residuals.csv", resid_buffer.getvalue())

    zip_buffer.seek(0)
    st.download_button(
        label="💾 DOWNLOAD ALL RESULTS",
        data=zip_buffer,
        file_name="PropagationMapping_Results.zip",
        mime="application/zip"
    )
