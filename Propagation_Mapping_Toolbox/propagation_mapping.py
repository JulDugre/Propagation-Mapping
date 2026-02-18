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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------------
# SETUP
# -------------------------------
st.set_page_config(page_title="Propagation Mapping Toolbox", layout="wide")
st.title("Propagation Mapping Toolbox (v0.3.1)")

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
# BASE DIRECTORY (ROBUST FIX)
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
framework_img_path = BASE_DIR / "miscellaneous" / "Framework.png"

if os.path.exists(framework_img_path):
    st.image(str(framework_img_path), width=800, caption="Methodological Overview of Propagation Mapping")
else:
    st.warning(f"Framework image not found at: {framework_img_path}")

# -------------------------------
# CACHED LOADING
# -------------------------------
@st.cache_resource(show_spinner="Loading Atlas Masker...")
def get_masker(atlas_choice):
    atlas_path = BASE_DIR / "atlases" / f"ATLAS_{atlas_choice}.nii.gz"
    if not atlas_path.exists(): return None
    return NiftiLabelsMasker(labels_img=str(atlas_path), strategy="mean", resampling_target="labels")

@st.cache_data(show_spinner="Loading Atlas Data...")
def load_atlas_data(atlas_choice):
    conn_path = BASE_DIR / "normative_connectomes" / f"ATLAS_{atlas_choice}_precomputed.csv"
    spin_path = BASE_DIR / "atlases" / "surfaces" / "spins" / f"spins_{atlas_choice}_hungarian1k.csv"
    label_path = BASE_DIR / "atlases" / f"listnames_ATLAS_{atlas_choice}.csv"
    if not all(p.exists() for p in [conn_path, spin_path, label_path]):
        return None, None, None
    return pd.read_csv(conn_path, index_col=0), pd.read_csv(spin_path, index_col=0), pd.read_csv(label_path)

# -------------------------------
# SESSION STATE
# -------------------------------
if 'tmp_dir' not in st.session_state:
    st.session_state.tmp_dir = tempfile.mkdtemp()
for key in ['masked_df', 'propagation_results', 'atlas_choice_prev']:
    if key not in st.session_state: st.session_state[key] = None

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("1. Data Input")
uploaded_files = st.sidebar.file_uploader("Upload NIFTI file(s)", type=['nii', 'nii.gz'], accept_multiple_files=True)
folder_path = st.sidebar.text_input("OR Folder Path:")
st.sidebar.header("2. Export Settings")
ext_choice = st.sidebar.selectbox("File Extension", [".csv", ".txt"])
use_header_index = st.sidebar.checkbox("Include Header & Index", value=True)

nii_files = []
def clean_name(name): return re.sub(r'\.nii(\.gz)?$', '', name)

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
        conn_df, _, _ = load_atlas_data(atlas_choice)
        if masker and conn_df is not None:
            n_rois = conn_df.shape[0]
            data = []
            progress_text = st.empty()
            prog = st.progress(0)
            for i, f in enumerate(nii_files):
                progress_text.text(f"Parcellating file {i+1} / {len(nii_files)}")
                vec = masker.fit_transform(f).flatten()
                full_vec = np.zeros(n_rois)
                full_vec[:min(len(vec), n_rois)] = vec[:min(len(vec), n_rois)]
                data.append(full_vec)
                prog.progress((i + 1) / len(nii_files))
            st.session_state.masked_df = pd.DataFrame(np.array(data).T)
            st.session_state.masked_df.columns = [clean_name(Path(f).name) for f in nii_files]
            st.success(f"Parcellation Complete! Found {n_rois} ROIs.")

# -------------------------------
# 3. ANALYSIS (MEMORY SAFE)
# -------------------------------
if st.session_state.masked_df is not None:
    st.header("2. Analysis Configuration")
    do_std = st.checkbox("Standardize Input (Z-score)", value=True)
    alpha_input = st.number_input("Significance Alpha", value=0.05, min_value=0.001, max_value=0.5)

    if st.button("🚀 LAUNCH PROPAGATION & SPIN TEST"):
        conn_df, spins_df, labels_df = load_atlas_data(atlas_choice)
        W_full = conn_df.values
        np.fill_diagonal(W_full, 0)
        spins = spins_df.values.astype(int)
        nspins, n_nodes = spins.shape[1], W_full.shape[0]

        cortical_idx = labels_df[labels_df['regions'].str.strip().str.lower() == 'cortex'].index.values
        subcortical_idx = np.setdiff1d(np.arange(n_nodes), cortical_idx)

        df_proc = st.session_state.masked_df.copy()
        if do_std: df_proc = df_proc.apply(lambda x: (x - x.mean()) / x.std())

        results_store = []
        progress_text_ana = st.empty()
        prog_ana = st.progress(0)

        for idx, col in enumerate(df_proc.columns):
            progress_text_ana.text(f"Processing subject {idx+1} / {len(df_proc.columns)}")
            act_raw = df_proc[col].values
            prod_obs = 0.5 * ((act_raw[:, None] * W_full) + (act_raw[:, None] * W_full).T)
            r2_f, mae_f, rmse_f = regression_metrics(act_raw, np.nansum(prod_obs, axis=0))

            # Running sums to save RAM
            sum_null, sum_sq_null, delta_r2_spin = np.zeros((n_nodes, n_nodes)), np.zeros((n_nodes, n_nodes)), []

            for s in range(nspins):
                perm = np.zeros(n_nodes)
                perm[cortical_idx] = act_raw[cortical_idx][spins[:, s]]
                shuff = act_raw[subcortical_idx].copy()
                np.random.shuffle(shuff)
                perm[subcortical_idx] = shuff

                p_null = np.zeros((n_nodes, n_nodes))
                for i_set in [cortical_idx, subcortical_idx]:
                    rows = np.ix_(i_set, i_set)
                    cc = perm[i_set][:, None] * W_full[rows]
                    p_null[rows] = 0.5 * (cc + cc.T)
                r_cs, r_sc = np.ix_(cortical_idx, subcortical_idx), np.ix_(subcortical_idx, cortical_idx)
                cs = perm[cortical_idx][:, None] * W_full[r_cs]
                p_null[r_cs], p_null[r_sc] = cs, cs.T

                sum_null += p_null
                sum_sq_null += (p_null ** 2)
                r2_n, _, _ = regression_metrics(act_raw, np.nansum(p_null, axis=0))
                delta_r2_spin.append(r2_f - r2_n)

            null_mean = sum_null / nspins
            null_std = np.sqrt(np.maximum((sum_sq_null - (sum_null**2 / nspins)) / (nspins - 1), 0)) + 1e-10
            z_matrix = (prod_obs - null_mean) / null_std
            sig_mask = np.abs(z_matrix) > norm.ppf(1 - alpha_input / 2)
            masked_mat = np.zeros_like(prod_obs)
            masked_mat[sig_mask] = prod_obs[sig_mask]

            results_store.append({
                'name': col, 'raw_mat': prod_obs, 'masked_mat': masked_mat, 'z_mat': z_matrix,
                'metrics': [r2_f, mae_f, rmse_f, np.mean(delta_r2_spin), (1 + np.sum(np.array(delta_r2_spin) <= 0))/(1+nspins)],
                'label_names': labels_df['Label'].tolist()
            })
            prog_ana.progress((idx + 1) / len(df_proc.columns))
        st.session_state.propagation_results = results_store
        st.success("Complete!")

# -------------------------------
# 4. EXPORT RESULTS
# -------------------------------
if st.session_state.get('propagation_results'):
    st.header("3. Results & Export")

    residuals_dict, res_summary = {}, []

    for r in st.session_state.propagation_results:
        name = r['name']
        r2_f, mae_f, rmse_f, mean_delta, p_delta = r['metrics']
        masked_vec = np.nansum(r['masked_mat'], axis=0)
        act_vec = st.session_state.masked_df[name].values
        
        # Calculate masked metrics
        r2_m, mae_m, rmse_m = regression_metrics(act_vec, masked_vec)
        residuals_dict[name] = act_vec - masked_vec
        res_summary.append([name, r2_f, mae_f, rmse_f, mean_delta, p_delta, r2_m, mae_m, rmse_m])

    sum_df = pd.DataFrame(res_summary, columns=['Subject','R2_Full','MAE_Full','RMSE_Full','Mean_Delta_R2','p_Delta','R2_Masked','MAE_Masked','RMSE_Masked'])
    st.dataframe(sum_df)

    # Residuals alignment
    label_names = st.session_state.propagation_results[0]['label_names']
    residuals_df = pd.DataFrame(residuals_dict)
    if residuals_df.shape[0] > len(label_names):
        residuals_df = residuals_df.iloc[:len(label_names), :]
    elif residuals_df.shape[0] < len(label_names):
        pad = pd.DataFrame(np.nan, index=range(len(label_names)-residuals_df.shape[0]), columns=residuals_df.columns)
        residuals_df = pd.concat([residuals_df, pad], axis=0)
    residuals_df.index = label_names

    # ZIP Creation
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        sep = ',' if ext_choice == '.csv' else '\t'
        for r in st.session_state.propagation_results:
            name, labs = r['name'], r['label_names']
            for m_type, m_data in zip(['raw','masked','zscore'], [r['raw_mat'], r['masked_mat'], r['z_mat']]):
                save_df = pd.DataFrame(m_data, index=labs, columns=labs) if use_header_index else pd.DataFrame(m_data)
                csv_buffer = io.StringIO()
                save_df.to_csv(csv_buffer, sep=sep)
                zipf.writestr(f"propagation_maps/{m_type}/{name}_{m_type}{ext_choice}", csv_buffer.getvalue())

        # Summary and Residuals
        sum_buf, res_buf = io.StringIO(), io.StringIO()
        sum_df.to_csv(sum_buf, index=False)
        residuals_df.to_csv(res_buf, sep=sep)
        zipf.writestr("propagation_maps/summary_metrics.csv", sum_buf.getvalue())
        zipf.writestr("propagation_maps/residuals.csv", res_buf.getvalue())

    st.download_button("💾 DOWNLOAD ALL RESULTS", zip_buffer.getvalue(), "PropagationMapping_Results.zip", "application/zip")
