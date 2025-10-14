import os
from tempfile import NamedTemporaryFile
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import smooth_img
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from glob import glob
from natsort import natsorted
from matplotlib import pyplot as plt
from neuromaps.datasets import fetch_fsaverage
from nilearn import plotting, datasets
import seaborn as sns
import re
from sklearn.preprocessing import RobustScaler, StandardScaler
import tempfile
from sklearn.linear_model import LinearRegression
from pathlib import Path
import shutil
from zipfile import ZipFile
import time

# --- Load into session state ---
if "func_df" not in st.session_state:
    st.session_state.func_df = None
if "struct_df" not in st.session_state:
    st.session_state.struct_df = None
if "masked_df" not in st.session_state:
    st.session_state.masked_df = None
if "propagation_maps" not in st.session_state:
    st.session_state.propagation_maps = []
if "predicted_regional_scaled" not in st.session_state:
    st.session_state.predicted_regional_scaled = []
if 'tmp_dir' not in st.session_state:
    st.session_state.tmp_dir = tempfile.mkdtemp()  # folder persists
if "saved_files" not in st.session_state:
    st.session_state.saved_files = []
if "nii_files" not in st.session_state:
    st.session_state.nii_files = []
if "col_names" not in st.session_state:
    st.session_state.col_names = []
if "parcellated" not in st.session_state:
    st.session_state.parcellated = False
if "launch_btn" not in st.session_state:
    st.session_state.launch_btn = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


tmp_dir = Path(st.session_state.tmp_dir)

# Create subfolders
results_dir = tmp_dir / "results"
plots_dir = tmp_dir / "plots"
results_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
obs_dir = results_dir / "observed_maps"
pred_dir = results_dir / "predicted_maps"
prop_dir = results_dir / "propagation_maps"
resid_dir = results_dir / "residual_maps"

for folder in [obs_dir, pred_dir, prop_dir, resid_dir]:
    folder.mkdir(parents=True, exist_ok=True)

# Define the Streamlit app UI
st.title("Propagation Mapping Toolbox")
st.markdown("##### Please cite:")
st.markdown("• Dugré, JR. (2025). Propagation Mapping: A Precision Framework for Reconstructing the Neural Circuitry of Brain Maps. *bioRxiv*, [DOI: 10.1101/2025.09.27.678975](https://doi.org/10.1101/2025.09.27.678975)")
st.markdown("For more information, please refer to the [Github Page](https://github.com/JulDugre/Propagation-Mapping/)")

# --- Display the framework image here ---
BASE_DIR = Path(__file__).parent
framework_img_path = BASE_DIR / "miscellaneous" / "Framework.png"
st.image(framework_img_path, caption="Propagation Mapping is a new precision framework aiming at reconstructing the neural circuitry that explains the spatial organization of human brain maps.This method assume that regional measures can be best understood as a dot product between activity and functional connectivity.Uploading your NIfTI file will predict the spatial pattern of your uploaded map and return 1) the observed (raw) parcellated map, 2) the predicted map, 3) the propagation map, and 4) a residual map", width='stretch')

st.sidebar.markdown("# UPLOAD IMAGE(S)")
st.sidebar.markdown("#### ⚠️ Note that the toolbox does not retain any data")

nii_files = []
col_names = []

def clean_name(name):
    return re.sub(r'\.nii(\.gz)?$', '', name)

def save_uploaded_nii(uf):
    """Save uploaded NIfTI file to a temp file, keeping the right suffix."""
    if uf.name.endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif uf.name.endswith(".nii"):
        suffix = ".nii"
    else:
        raise ValueError(f"Unsupported file type: {uf.name}")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(uf.getbuffer())
    tmp_file.close()  # important to flush everything
    return Path(tmp_file.name)


def clean_name(name):
    return re.sub(r'\.nii(\.gz)?$', '', name)


def reset_uploader():
    """Reset uploader-related session state without creating widgets."""
    st.session_state.nii_files = []
    st.session_state.col_names = []
    st.session_state.masked_df = None
    st.session_state.propagation_maps = []
    st.session_state.predicted_regional_scaled = []
    st.session_state.parcellated = False
    st.session_state.launch_btn = False
    st.session_state.plot_prop_btn = False
    st.session_state.plot_pred_btn = False
    st.session_state.uploader_key += 1

# --- Single file uploader widget ---
uploaded_files = st.sidebar.file_uploader(
    "Upload NIFTI file(s)",
    type=["nii", "gz"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"
)

# Optional manual reset button
if st.sidebar.button("Reset"):
    reset_uploader()
	
# Process uploaded files
if uploaded_files:
    # Clear old files first
    reset_uploader()  # safely reset session state

    for uf in uploaded_files:
        if not (uf.name.endswith(".nii") or uf.name.endswith(".nii.gz")):
            st.warning(f"Skipped unsupported file: {uf.name}")
            continue

        # Save uploaded file to temp
        suffix = ".nii.gz" if uf.name.endswith(".gz") else ".nii"
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.write(uf.getbuffer())
        tmp_file.close()
        tmp_path = Path(tmp_file.name)

        st.session_state.nii_files.append(tmp_path)
        st.session_state.col_names.append(uf.name)

    # Ensure unique column names
    unique_cols = []
    for i, name in enumerate(st.session_state.col_names):
        base_name = re.sub(r'\.nii(\.gz)?$', '', name)
        if base_name not in unique_cols:
            unique_cols.append(base_name)
        else:
            suffix = 1
            new_name = f"{base_name}_{suffix}"
            while new_name in unique_cols:
                suffix += 1
                new_name = f"{base_name}_{suffix}"
            unique_cols.append(new_name)
    st.session_state.col_names = unique_cols

    st.session_state.parcellated = False
    st.success(f"Loaded {len(st.session_state.nii_files)} NIfTI file(s).")
	
# --- Load images ---
if st.session_state.nii_files:
    loaded_imgs = [nib.load(f) for f in st.session_state.nii_files]
else:
    loaded_imgs = []
    st.write("No images loaded yet.")
  
# --- Atlas selection ---
# --- Atlas selection ---
if loaded_imgs:    
    st.header("Please select a Brain Atlas for Parcellation:")
    atlas_choice = st.radio(
        "Choose one atlas:",
        ("Schaefer7", "Desikan", "Gordon", "Glasser"),
        index=0
    )
    st.session_state.atlas_choice = atlas_choice  # Save choice

    atlas_files = {
        "Schaefer7": BASE_DIR / "atlases" / "ATLAS_Schaefer7.nii.gz",
        "Desikan": BASE_DIR / "atlases" / "ATLAS_Desikan.nii.gz",
        "Gordon": BASE_DIR / "atlases" / "ATLAS_Gordon.nii.gz",
        "Glasser": BASE_DIR / "atlases" / "ATLAS_Glasser.nii.gz"
    }
    atlas_path = atlas_files[atlas_choice]

    # Load atlas image and ROI labels immediately
    st.session_state.atlas_img = nib.load(atlas_path)
    atlas_csv = BASE_DIR / "atlases" / f"listnames_ATLAS_{atlas_choice}.csv"
    labels_info = pd.read_csv(atlas_csv)
    st.session_state.roi_labels = labels_info['Label'].tolist()

    # Load connectomes
    func_file = BASE_DIR / "normative_connectomes" / "func" / f"ATLAS_{atlas_choice}_resting_Fz.csv"
    struct_file = BASE_DIR / "normative_connectomes" / "struct" / f"ATLAS_{atlas_choice}_structural_Fz.csv"

    st.session_state.func_df = pd.read_csv(func_file, index_col=0) if func_file.exists() else None
    st.session_state.struct_df = pd.read_csv(struct_file, index_col=0) if struct_file.exists() else None

    # --- Clean connectomes ---
if st.session_state.func_df is not None and st.session_state.struct_df is not None:
    func_connectome = st.session_state.func_df.values.copy()
    struct_connectome = st.session_state.struct_df.values.copy()
    np.fill_diagonal(func_connectome, 0)
    np.fill_diagonal(struct_connectome, 0)
    func_connectome[np.isinf(func_connectome)] = 0
    struct_connectome[np.isinf(struct_connectome)] = 0
	
# Reset buttons when a new file/folder is selected
if uploaded_files:
    st.session_state.launch_btn = False
    st.session_state.plot_prop_btn = False
    st.session_state.plot_pred_btn = False

# Create three columns
col1_btn, col2_btn, col3_btn = st.columns(3)

# Place buttons in each column

with col1_btn:
    if st.button("LAUNCH THE ROCKET 🚀"):
        st.session_state.launch_btn = True

with col2_btn:
    if st.button("PLOT PROPAGATION MAP 🎨"):
        st.session_state.plot_prop_btn = True

with col3_btn:
    if st.button("PLOT PREDICTION MAP 🧠"):
        st.session_state.plot_pred_btn = True


if st.session_state.launch_btn:
    if not st.session_state.nii_files:
        st.warning("No NIfTI files uploaded!")
    elif "atlas_img" not in st.session_state or "roi_labels" not in st.session_state:
        st.warning("No atlas selected!")
    elif st.session_state.func_df is None or st.session_state.struct_df is None:
        st.warning("Functional or structural connectomes not loaded!")
    else:
        # Access atlas and labels from session state
        atlas_img = st.session_state.atlas_img
        roi_labels = st.session_state.roi_labels
        atlas_choice = st.session_state.atlas_choice
        func_connectome = st.session_state.func_df.values.copy()
        struct_connectome = st.session_state.struct_df.values.copy()
        
        # Proceed with parcellation and propagation mapping...
        # --- Load images ---
        loaded_imgs = [nib.load(f) for f in st.session_state.nii_files]

        # --- Parcellation ---
        masker = NiftiLabelsMasker(
            labels_img=st.session_state.atlas_img,
            strategy='mean',
            keep_masked_labels=True,
            resampling_target="labels"
        )

        parcellation_progress = st.progress(0)
        progress_text = st.empty()
        start_time = time.time()

        masked_data = []
        for i, img in enumerate(loaded_imgs):
            roi_values = masker.fit_transform(img).squeeze()
            masked_data.append(roi_values)

            # Progress bar update
            frac_done = (i + 1) / len(loaded_imgs)
            parcellation_progress.progress(frac_done)

            elapsed = time.time() - start_time
            time_per_img = elapsed / (i + 1)
            remaining = time_per_img * (len(loaded_imgs) - (i + 1))
            progress_text.text(f"Time for a  ☕, Parcellating image {i+1}/{len(loaded_imgs)} — ~{int(remaining)} sec remaining")

        # --- Save to session state ---
        st.session_state.masked_df = pd.DataFrame(np.array(masked_data).T, columns=st.session_state.col_names)
        st.session_state.parcellated = True  # mark as done
		
        # --- Display results ---
        st.header("Parcellated Data")
        st.write("Shape of parcellated data (ROIs × Subjects):", st.session_state.masked_df.shape)
        st.dataframe(st.session_state.masked_df)

        # --- Connectomes ---
        func_connectome = st.session_state.func_df.values.copy()
        struct_connectome = st.session_state.struct_df.values.copy()
        np.fill_diagonal(func_connectome, 0)
        np.fill_diagonal(struct_connectome, 0)
        func_connectome[np.isinf(func_connectome)] = 0
        struct_connectome[np.isinf(struct_connectome)] = 0

        pred_accuracy = []
        pred_corr_accuracy = []
        predicted_regional = []     
        true_regional = []
        pred_hubs_accuracy = []
        st.session_state.propagation_maps = []
        st.session_state.predicted_regional_scaled = []
        st.session_state.predicted_regional_scaledcorr = []
		
        # Create results folder if it does not exist
        output_folder = BASE_DIR / "results"
        output_folder.mkdir(parents=True, exist_ok=True)

        n_subjects = st.session_state.masked_df.shape[1]
        rocket_progress = st.progress(0)
        progress_text = st.empty()  # For text info

        def residualize(y, X):
           """
           Removes linear effects of X from y.
           Both y and X can be 1D arrays of the same length.
           """
           y = np.asarray(y).reshape(-1, 1)
           X = np.asarray(X).reshape(-1, 1)
           model = LinearRegression().fit(X, y)
           y_pred = model.predict(X)
           return (y - y_pred).flatten()

		
        # Loop over each subject/column in masked_df
        for idx in range(n_subjects):
            feature_vector = st.session_state.masked_df.iloc[:, idx].values
            filename =st.session_state.masked_df.columns[idx]  # use uploaded filename
            
         # --- Functional connectome ---
            connectome_FC = np.clip(func_connectome, a_min=0, a_max=None)
            product_FC = feature_vector[:, np.newaxis] * connectome_FC
            product_FC_sym = 0.5 * (product_FC + product_FC.T)

            # --- CHANGE: Pre-scale FC propagation matrix ---
            triu_idx = np.triu_indices_from(product_FC_sym, k=1)  # get upper triangle indices
            upper_vals_FC = product_FC_sym[triu_idx].reshape(-1, 1)
            upper_vals_FC_scaled = rob_scaler.fit_transform(upper_vals_FC).flatten()  # scale before summing
            product_FC_sym_scaled = np.zeros_like(product_FC_sym)
            product_FC_sym_scaled[triu_idx] = upper_vals_FC_scaled
            product_FC_sym_scaled = product_FC_sym_scaled + product_FC_sym_scaled.T
            np.fill_diagonal(product_FC_sym_scaled, np.diag(product_FC_sym))

            # --- Structural covariance ---
            connectome_SC = np.clip(struct_connectome, a_min=0, a_max=None)
            product_SC = feature_vector[:, np.newaxis] * connectome_SC
            product_SC_sym = 0.5 * (product_SC + product_SC.T)

            # --- CHANGE: Pre-scale SC propagation matrix ---
            upper_vals_SC = product_SC_sym[triu_idx].reshape(-1, 1)
            upper_vals_SC_scaled = rob_scaler.fit_transform(upper_vals_SC).flatten()  # scale before summing
            product_SC_sym_scaled = np.zeros_like(product_SC_sym)
            product_SC_sym_scaled[triu_idx] = upper_vals_SC_scaled
            product_SC_sym_scaled = product_SC_sym_scaled + product_SC_sym_scaled.T
            np.fill_diagonal(product_SC_sym_scaled, np.diag(product_SC_sym))

            # --- CHANGE: Average the pre-scaled matrices ---
            avg_BOTH_sym_scaled = 0.5 * (product_FC_sym_scaled + product_SC_sym_scaled)
            st.session_state.propagation_maps.append(avg_BOTH_sym_scaled)

            # --- Predict regional values ---
            pred_regional = avg_BOTH_sym_scaled.sum(axis=0)  # sum is now already on scaled matrices

            # --- Predict regional values ---
            #pred_regional = avg_BOTH_sym.sum(axis=0)
            pred_regional_scaled = rob_scaler.fit_transform(pred_regional.reshape(-1, 1)).flatten()

            # --- Correlation with true feature vector ---
            corr_pos, _ = pearsonr(pred_regional, feature_vector)
            pred_accuracy.append(corr_pos)
            
			# --- Density correction ---
            density = 0.5 * (connectome_FC.mean(axis=0) + connectome_SC.mean(axis=0))
            corr_density_act, _ = pearsonr(density, feature_vector)
            pred_hubs_accuracy.append(corr_density_act)
            pred_resid = residualize(pred_regional, density)
            act_resid  = residualize(feature_vector, density)
			
            # --- Now scale both residualized observed and predicted ---
            act_resid_scaled  = rob_scaler.fit_transform(act_resid.reshape(-1, 1)).flatten()
            pred_resid_scaled = rob_scaler.fit_transform(pred_resid.reshape(-1, 1)).flatten()

            # --- Correlation with true feature vector (hub-corrected) ---
            corr_scaled_dencorr, _ = pearsonr(pred_resid_scaled, act_resid_scaled)
            pred_corr_accuracy.append(corr_scaled_dencorr)
			
            # --- Store results ---
            st.session_state.predicted_regional_scaled.append(pred_regional_scaled)
            st.session_state.predicted_regional_scaledcorr.append(pred_resid_scaled)
            predicted_regional.append(pred_regional_scaled)
            true_regional.append(feature_vector)
            
            # --- Compute residuals (Observed - Predicted) ---
            model = LinearRegression().fit(pred_regional.reshape(-1, 1), feature_vector)
            y_hat = model.predict(pred_regional.reshape(-1, 1))
            residuals = feature_vector - y_hat
            residuals_z = (residuals - np.mean(residuals)) / np.std(residuals)
            mae_per_region = np.abs(residuals)
			
            # --- Save CSVs ---
            pred_file = output_folder / f"{filename}_pred_scaled_map.csv"
            pred_corr_file = output_folder / f"{filename}_pred_scaled_corr_map.csv"
            obs_file = output_folder / f"{filename}_obs_map.csv"
            prop_file = output_folder / f"{filename}_propagationmap.csv"
            resid_file = output_folder / f"{filename}_z_residualmap.csv"
			
            pd.DataFrame({"Observed":feature_vector}, index=roi_labels).to_csv(obs_dir / f"{filename}_{atlas_choice}_obs_map.csv")
            pd.DataFrame({"Predicted_scaled": pred_regional_scaled}, index=roi_labels).to_csv(pred_dir / f"{filename}_{atlas_choice}_pred_scaled_map.csv")
            pd.DataFrame({"Predicted_scaledcorr": pred_resid_scaled}, index=roi_labels).to_csv(pred_dir / f"{filename}_{atlas_choice}_pred_scaled_corr_map.csv")
            pd.DataFrame(avg_BOTH_sym_scaled, index=roi_labels, columns=roi_labels).to_csv(prop_dir / f"{filename}_{atlas_choice}_propagationmap.csv")
            pd.DataFrame({"Residual_z": residuals_z, "MAE": mae_per_region}, index=roi_labels).to_csv(resid_dir / f"{filename}_{atlas_choice}_z_residualmap.csv")
			
            st.session_state.saved_files.extend([
				obs_dir / f"{filename}_{atlas_choice}_obs_map.csv",
				pred_dir / f"{filename}_{atlas_choice}_pred_scaled_map.csv",
				pred_dir / f"{filename}_{atlas_choice}_pred_scaled_corr_map.csv",
				prop_dir / f"{filename}_{atlas_choice}_propagationmap.csv",
				resid_dir / f"{filename}_{atlas_choice}_z_residualmap.csv"])
			
            rocket_progress.progress((idx + 1) / n_subjects)
            progress_text.text(f"Another sip  ☕, Processing subject {idx + 1} of {n_subjects}: {filename}")
			
        # --- Save all pred accuracies ---
        all_acc_df = pd.DataFrame({"Raw_r": np.array(pred_accuracy),"Corrected_r": np.array(pred_corr_accuracy), "Hubs_r": np.array(pred_hubs_accuracy)}, index=st.session_state.col_names)        
        all_acc_df.to_csv(results_dir / f"{atlas_choice}_accuracy.csv")		
        st.session_state.saved_files.extend([results_dir / f"{atlas_choice}_accuracy.csv"])	
		
        # --- Show summary ---
        st.success("🚀 Propagation mapping complete!")

        # --- Compute summary stats if more than one subject ---
        if len(pred_accuracy) == 1:
            st.header(f"Prediction Accuracy for {st.session_state.masked_df.columns[0]}")            
            st.write("Raw-correlation:", pred_accuracy[0])
            st.write("Corrected-correlation:", pred_corr_accuracy[0])
            st.write("Hubs-correlation:", pred_hubs_accuracy[0])

        else:
            st.header("Prediction Accuracy Across Subjects") 
            summary_stats = all_acc_df.agg(["mean", "std", "min", "max"])
            st.subheader("Summary statistics across all subjects")
            st.table(summary_stats)

            # --- Density plot of prediction accuracy ---
            import seaborn as sns
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6,4))
            sns.kdeplot(all_acc_df["Raw_r"], fill=True, label="Raw", ax=ax, clip=(0,1))
            sns.kdeplot(all_acc_df["Corrected_r"], fill=True, label="Scaled+Dencorr", ax=ax, clip=(0,1))
            sns.kdeplot(all_acc_df["Hubs_r"], fill=True, label="Hubs", ax=ax, clip=(0,1))
            ax.set_xlim(0.5, 1)
            ax.set_xlabel("Predictive Accuracies\n(correlation)")
            ax.set_ylabel("Density")
            st.pyplot(fig)

# ------------------ PLOTTING ------------------
if st.session_state.get("plot_prop_btn", False) and st.session_state.masked_df is not None and not st.session_state.masked_df.empty and st.session_state.propagation_maps:

    # Create a container for the atlas image
    one, two, three, four, five = st.columns(5)
    with two:
        atlas_img_path = BASE_DIR / "miscellaneous" / "schaefer_net.png"
        st.image(
            str(atlas_img_path),
            width=400,
            caption="Schaefer-400 7 Networks with 14 Subcortical, and 7 Cerebellar Regions"
        )

    st.header(f"Plot Propagation Map of: {st.session_state.masked_df.columns[0]}")
    propagation_maps = st.session_state.propagation_maps  
    plots_folder = BASE_DIR / "plots"
    plots_folder.mkdir(parents=True, exist_ok=True)

    # ----------------- PREP DATA -----------------
    atlas_csv = BASE_DIR / "atlases" / "listnames_ATLAS_Schaefer7.csv"
    labels_info = pd.read_csv(atlas_csv)
    roi_labels = labels_info['Label'].tolist()
    connectome = pd.DataFrame(propagation_maps[0], index=roi_labels, columns=roi_labels)
    ctx_sctx_conn = connectome.iloc[:414, :414]
    coords = list(zip(labels_info.x[:414], labels_info.y[:414], labels_info.z[:414]))

# ----------------- COL1: HEATMAP -----------------
col1, col2 = st.columns(2)

def render_heatmap(subject_idx=0, labels_info=None):
    # Check if data exists
    if not st.session_state.propagation_maps or labels_info is None:
        st.warning("No propagation maps found or labels missing. Click 'LAUNCH THE ROCKET 🚀' first.")
        return

    filename = st.session_state.masked_df.columns[subject_idx]
    propagation_maps = st.session_state.propagation_maps
    roi_labels = labels_info['Label'].tolist()

    prop_map = propagation_maps[subject_idx]
    connectome_matrix = pd.DataFrame(prop_map, index=roi_labels, columns=roi_labels)

    # Sort by network
    labels_info['Order'] = labels_info['Network']
    sorted_labels = labels_info.sort_values('Order')['Label'].tolist()
    reordered_matrix = connectome_matrix.loc[sorted_labels, sorted_labels]

    network_counts = labels_info.groupby('Network').size()
    network_names = network_counts.index.tolist()
    network_colors = {n: labels_info[labels_info['Network']==n]['colors'].unique()[0] for n in network_names}

    # Save heatmap
    heatmap_file = plots_dir / f"{filename}_heatmap.png"
    plt.figure(figsize=(16, 10))
    cmap = plt.get_cmap('Blues')
    cmap.set_under('white')
    ax = sns.heatmap(
        reordered_matrix,
        cmap=cmap,
        cbar=True,
        square=True,
        annot=False,
        vmax=connectome_matrix.values.max() * 0.9,
        linewidths=0,
        linecolor='white',
        vmin=0,
        cbar_kws={"orientation": "horizontal", "pad": 0.05, "fraction": 0.05, "shrink": 0.5}
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # Add separators and labels
    pos = 0
    for network in network_names:
        count = network_counts[network]
        if pos != 0:
            ax.vlines(pos, 0, len(reordered_matrix), color='black', lw=1, linestyle=(0, (3,5)))
            ax.hlines(pos, 0, len(reordered_matrix), color='black', lw=1, linestyle=(0, (3,5)))
        ax.text(-0.1, pos + count/2, network, ha='right', va='center', rotation=0, transform=ax.get_yaxis_transform())
        pos += count

    # Network color blocks
    color_ax = plt.axes([0.204, 0.145, 0.02, 0.842])
    pos = 0
    for network in network_names[::-1]:
        count = network_counts[network]
        color_ax.add_patch(plt.Rectangle((0, pos), 1, count, color=network_colors[network], alpha=1))
        pos += count
    color_ax.set_xlim(0, 1)
    color_ax.set_ylim(0, len(reordered_matrix))
    color_ax.axis('off')

    # Colorbar label
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Propagation Strength', fontsize=12)
    colorbar.ax.tick_params(labelsize=10)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

    plt.tight_layout()
    plt.savefig(heatmap_file, dpi=300)
    st.pyplot(plt)
    plt.close()

    # Save file path to session_state
    st.session_state.saved_files.append(heatmap_file)

if st.session_state.get("plot_prop_btn", False):
    if st.session_state.masked_df is not None and st.session_state.propagation_maps:
        # Load labels_info safely here
        atlas_csv = BASE_DIR / "atlases" / "listnames_ATLAS_Schaefer7.csv"
        labels_info = pd.read_csv(atlas_csv)
        
        col1, col2 = st.columns(2)
        with col1:
            render_heatmap(subject_idx=0, labels_info=labels_info)

# ----------------- COL2: 3D CONNECTOME -----------------
    @st.fragment
    def render_connectome_fragment():
        # Initialize threshold in session_state
        if "threshold_value" not in st.session_state:
            st.session_state.threshold_value = 99.0  # default top percentile

        # --- Only positive edges ---
        pos_conn_values = ctx_sctx_conn.values.flatten()
        pos_conn_values = pos_conn_values[pos_conn_values > 0]
        cutoff_value = np.percentile(pos_conn_values, st.session_state.threshold_value)
		
        # Mask nodes
        mask_edges = (ctx_sctx_conn.values > cutoff_value).any(axis=0)  # boolean array aligned with rows/cols
        filtered_conn = ctx_sctx_conn.loc[mask_edges, mask_edges]
        filtered_conn = filtered_conn.clip(lower=cutoff_value)
        filtered_coords = [coord for keep, coord in zip(mask_edges, coords) if keep]

        if filtered_conn.shape[0] > 0:
            # Node strength and scaled size
            strength = filtered_conn.sum(axis=1)

            node_size = 30 * (np.log1p(strength) - np.log1p(strength).min()) / (
                np.log1p(strength).max() - np.log1p(strength).min()
            )

            # Colors for filtered nodes
            all_colors = labels_info.colors[:414]  # adjust if more nodes
            filtered_colors = [all_colors[list(ctx_sctx_conn.index).index(node_id)] for node_id in filtered_conn.index]

            # Render connectome
            view_conn = plotting.view_connectome(
                filtered_conn.values,
                filtered_coords,
                edge_threshold=f"{st.session_state.threshold_value}%",  # percentile
                node_size=node_size.values,
                node_color=filtered_colors, linewidth=2,
                colorbar=False
            )
            view_conn.resize(width=400, height=400)
            st.components.v1.html(view_conn._repr_html_(), height=300)

            # Slider to adjust percentile threshold
            st.session_state.threshold_value = st.slider(
                "Connectivity Threshold (Top % of positive edges)",
                min_value=90.0,
                max_value=99.0,
                value=st.session_state.threshold_value,
                step=1.0
            )

        else:
            st.warning("No connections survived at this threshold.")

    # Render inside column
    with col2:
        render_connectome_fragment()

# ----------------- PLOT COMPARISONS BETWEEN OBSERVED AND PREDICTED -----------
if st.session_state.get("plot_pred_btn", False):
  if st.session_state.masked_df is not None and not st.session_state.masked_df.empty and st.session_state.predicted_regional_scaled:
    st.header(f"Compare Maps of: {st.session_state.masked_df.columns[0]}")
    # ----------------- OBSERVED MAP -----------------
    # Define filename for saving joint plot
    filename = st.session_state.masked_df.columns[0]
    df_obs = st.session_state.masked_df.iloc[:, 0:1].reset_index()
    df_obs.columns = ["num", "value"]

    # Construct surface CSV path dynamically
    surface_path = BASE_DIR / "atlases" / "surfaces" / f"ATLAS_{atlas_choice}_fsa5.csv"

    if os.path.exists(surface_path):
        target_lab = pd.read_csv(surface_path, index_col=None, header=None).values.flatten().tolist()
        midpoint = len(target_lab) // 2
        lh_labels = target_lab[:midpoint]
        rh_labels = target_lab[midpoint:]

        # Map observed values
        result_left_obs = np.array([df_obs.loc[df_obs['num'] == x, 'value'].values[0] if x in df_obs['num'].values else 0 for x in lh_labels])
        result_right_obs = np.array([df_obs.loc[df_obs['num'] == x, 'value'].values[0] if x in df_obs['num'].values else 0 for x in rh_labels])

        combined_result_obs = np.concatenate((result_left_obs, result_right_obs), axis=0)

        # ----------------- PREDICTED MAP -----------------
        # Use pred_regional_scaled from previous computation
        pred_scaled = st.session_state.predicted_regional_scaled[0]  # first subject
        df_pred = pd.DataFrame({"num": df_obs['num'], "value": pred_scaled})

        result_left_pred = np.array([df_pred.loc[df_pred['num'] == x, 'value'].values[0] if x in df_pred['num'].values else 0 for x in lh_labels])
        result_right_pred = np.array([df_pred.loc[df_pred['num'] == x, 'value'].values[0] if x in df_pred['num'].values else 0 for x in rh_labels])

        combined_result_pred = np.concatenate((result_left_pred, result_right_pred), axis=0)

        # ----------------- LOAD FSAVERAGE SURFACES -----------------
        surfaces = fetch_fsaverage(density='10k')
        surf_lh, surf_rh = surfaces['pial']

        gii_lh = nib.load(str(surf_lh))
        gii_rh = nib.load(str(surf_rh))

        coords_lh = gii_lh.darrays[0].data
        faces_lh  = gii_lh.darrays[1].data

        coords_rh = gii_rh.darrays[0].data
        faces_rh  = gii_rh.darrays[1].data

        faces_rh_offset = faces_rh + coords_lh.shape[0]
        coords_both = np.vstack([coords_lh, coords_rh])
        faces_both  = np.vstack([faces_lh, faces_rh_offset])
        surf_mesh_both = [coords_both, faces_both]

        # ----------------- CREATE VIEWS -----------------
        view_obs = plotting.view_surf(surf_mesh=surf_mesh_both, surf_map=combined_result_obs, cmap='coolwarm', colorbar=False)
        view_pred = plotting.view_surf(surf_mesh=surf_mesh_both, surf_map=combined_result_pred,  cmap='coolwarm', colorbar=False)

        # ----------------- RESIZE VIEWS -----------------
        view_obs.resize(width=300, height=400)   # smaller width/height
        view_pred.resize(width=300, height=400)

        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)

        # Observed Map in first column
        with col1:
          st.subheader("Observed Map")
          st.components.v1.html(view_obs._repr_html_(), height=300)
      
        # Predicted Map (Standardized) in second column
        with col2:
            st.subheader("Predicted Map")
            st.components.v1.html(view_pred._repr_html_(), height=300)

         # ---------------- JOINT PLOT ----------------------
        st.header(f"Accuracy Joint Plot of: {st.session_state.masked_df.columns[0]}")

	# Create two columns for side-by-side display
        col1, col2, col3 = st.columns(3)

        atlas_csv = BASE_DIR / "atlases" / "listnames_ATLAS_Schaefer7.csv"
        labels_info = pd.read_csv(atlas_csv)
        scaler = RobustScaler()
        df_obs_scaled = scaler.fit_transform(df_obs['value'].values.reshape(-1, 1))
        df_obs_scaled = pd.DataFrame(df_obs_scaled)
        
        df_pred_scaled = pd.DataFrame(pred_scaled, columns=['value'])
        combined = pd.concat([df_obs_scaled, df_pred_scaled], axis=1)
        combined.columns = ['obs','pred']
        combined.index = labels_info['Label'].tolist()
        combined['ColorGroup'] = combined['ColorGroup'] = pd.Categorical(
    labels_info['colors'], 
    categories=labels_info['colors'].drop_duplicates(),  # keep first appearance order
    ordered=True
).codes
        st.dataframe(combined)
        unique_colors = labels_info['colors'].drop_duplicates().tolist()
        palette = {i: color for i, color in enumerate(unique_colors)}

        # Step 3: Jointplot using color group
        plt.figure(figsize=(16, 10))
        plt.tight_layout()

        g = sns.jointplot(data=combined, x="obs", y="pred", hue='ColorGroup',
                          palette=palette, edgecolor='k', height=8)

        # Step 4: Linear regression
        X = combined['obs'].values.reshape(-1, 1)
        y = combined['pred'].values
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)

        g.ax_joint.plot(combined['obs'], y_pred, color="#504E4E", lw=2, alpha=0.6)

        # Label with formatting
        g.ax_joint.set_xlabel(
            r"$\mathbf{Observed\ Map}$" + "\n(zscore)",
            fontsize=14
        )
        g.ax_joint.set_ylabel(
            r"$\mathbf{Predicted\ Map}$" + "\n(zscore)",
            fontsize=14
        )
        g.ax_joint.set_xlim((combined.obs.max()+1)*-1, combined.obs.max()+1)
        g.ax_joint.set_ylim((combined.obs.max()+1)*-1, combined.pred.max()+1)
        g.ax_joint.legend_.remove()
        
		# Save figure
        jointplot_file = plots_dir / f"{filename}_accuracy_jointplot.png"
        g.fig.tight_layout()
        g.fig.savefig(jointplot_file, dpi=300, bbox_inches='tight')
        for f in [jointplot_file]:
             st.session_state.saved_files.append(f)
		# Display in Streamlit
        st.image(jointplot_file, width=670)  # width in pixels
        plt.close(g.fig)

# --- Create ZIP of results for download ---
zip_path = tmp_dir / "Propagation_Results.zip"
with ZipFile(zip_path, 'w') as zipf:
    for file_path in st.session_state.saved_files:
        # Make the archive name relative to tmp_dir
        arcname = file_path.relative_to(tmp_dir)
        zipf.write(file_path, arcname=str(arcname))

# Streamlit download button
with open(zip_path, "rb") as f:
    st.download_button(
        "Download all results",
        data=f,
        file_name="Propagation_Results.zip"
    )
