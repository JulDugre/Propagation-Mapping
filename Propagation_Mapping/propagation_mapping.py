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

if 'tmp_dir' not in st.session_state:
    st.session_state.tmp_dir = tempfile.mkdtemp()  # folder persists

# Define the Streamlit app UI
st.title("Propagation Mapping Tool")
st.markdown("##### Please cite:")
st.markdown("• Dugré, JR. (2025). Propagation Mapping: A Precision Framework for Reconstructing the Neural Circuitry of Brain Maps. *bioRxiv*, DOI: Not Yet")
st.markdown("• Cole, Ito, Bassett et Schultz. (2016). *Nature Neurosci*, DOI:10.1038/nn.4406")


# --- Display the framework image here ---
BASE_DIR = Path(__file__).parent
framework_img_path = BASE_DIR / "miscellaneous" / "Framework.png"
st.image(framework_img_path, caption="Propagation Mapping is a new precision framework aiming at reconstructing the neural circuitry that explains the spatial organization of human brain maps.This method assume that regional measures can be best understood as a dot product between activity and functional connectivity.Uploading your NIfTI file will predict the spatial pattern of your uploaded map and return 1) the observed (raw) parcellated map,2) the predicted map, and 3) a propagation map, which reflects the circuitry predicting your statistical map, prior to summation", width='stretch')

st.sidebar.markdown("# UPLOAD IMAGE(S)")
st.sidebar.markdown("#### ⚠️ Note that the toolbox does not retain any data")

# --- Load into session state ---
if "func_df" not in st.session_state:
    st.session_state.func_df = None
if "struct_df" not in st.session_state:
    st.session_state.struct_df = None
if "masked_df" not in st.session_state:
    st.session_state.masked_df = None
	
nii_files = []
def clean_name(name):
    return re.sub(r'\.nii(\.gz)?$', '', name)

# --- Single NIfTI uploader (click and select) ---
uploaded_file = st.sidebar.file_uploader(
    "Upload a SINGLE NIFTI file",
    type=['nii', 'nii.gz']
)

# --- Folder input for multiple NIfTIs ---
folder_path = st.sidebar.text_input("Enter folder path for multiple NIFTIs:")

if uploaded_file is not None:
    suffix = '.nii.gz' if uploaded_file.name.endswith('.gz') else '.nii'
    col_names = [clean_name(uploaded_file.name)]
    
    tmp_path = os.path.join(st.session_state.tmp_dir, uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    nii_files.append(tmp_path)
    st.success(f"Uploaded file: {uploaded_file.name}")


# --- Handle folder path ---
elif folder_path:
    if os.path.exists(folder_path):
        nii_files = natsorted(glob(os.path.join(folder_path, '*.nii')) +
                              glob(os.path.join(folder_path, '*.nii.gz')))
        col_names = [clean_name(os.path.basename(f)) for f in nii_files]
        if nii_files:
            st.success(f"{len(nii_files)} NIfTI file(s) detected in folder: {folder_path}")
        else:
            st.warning(f"No NIfTI files found in folder: {folder_path}")
    else:
        st.error("Folder path does not exist.")

# --- Load images ---
loaded_imgs = [nib.load(f) for f in nii_files] if nii_files else []
st.write(f"{len(loaded_imgs)} image(s) loaded successfully.")

# If data loaded, prompt for atlas selection
if loaded_imgs:
    st.success(f"{len(loaded_imgs)} NIfTI file(s) successfully loaded.")
    
    # Example: display first image shape
    img_data = loaded_imgs[0].get_fdata()
    st.write(f"First image shape: {img_data.shape}")

    # Prompt for atlas selection
    st.header("Please select a Brain Atlas for Parcellation:")
    atlas_choice = st.radio(
        "Choose one atlas:",
        ("Schaefer7", "Desikan", "Gordon", "Glasser"),
        index=0
    )

    # Map atlas names to file paths
    atlas_files = {
	"Schaefer7": "atlases/ATLAS_Schaefer7.nii.gz",
        "Desikan": "atlases/ATLAS_Desikan.nii.gz",
        "Gordon": "atlases/ATLAS_Gordon.nii.gz",
        "Glasser": "atlases/ATLAS_Glasser.nii.gz",
    }

    atlas_path = atlas_files[atlas_choice]

    if os.path.exists(atlas_path):
        atlas_img = nib.load(atlas_path)
        st.success(f"✅ You selected the **{atlas_choice}** atlas.")

        # --- Define masker AFTER images are loaded ---
        masker = NiftiLabelsMasker(
            labels_img=atlas_img,
            strategy='mean',
            keep_masked_labels=True,resampling_target="labels")

        # --- Parcellate all images ---
        masked_data = []
        for img in loaded_imgs:
            roi_values = masker.fit_transform(img)
            roi_values = roi_values.squeeze()
            masked_data.append(roi_values)

        # --- Convert to DataFrame with generic column names ---

        st.session_state.masked_df = pd.DataFrame(np.array(masked_data).T, columns=col_names)
		
        # --- Display results ---
        st.header("Parcellated Data")
        st.write("Shape of parcellated data (ROIs × Subjects):", masked_df.shape)
        st.dataframe(masked_df)

        func_dir = BASE_DIR / "normative_connectomes" / "func"
        struct_dir = BASE_DIR / "normative_connectomes" / "struct"

        func_file = func_dir / f"ATLAS_{atlas_choice}_resting_Fz.csv"
        struct_file = struct_dir / f"ATLAS_{atlas_choice}_structural_Fz.csv"

        if func_file.exists():
            st.session_state.func_df = pd.read_csv(func_file, index_col=[0])
            st.write(f"Functional connectome shape: {func_df.shape}")
        else:
            st.warning(f"No functional connectome found for {atlas_choice}")

        if struct_file.exists():
            st.session_state.struct_df = pd.read_csv(struct_file, index_col=[0])
            st.write(f"Structural connectome shape: {struct_df.shape}")
        else:
            st.warning(f"No structural connectome found for {atlas_choice}")

    # --- Clean connectomes ---
if st.session_state.func_df is not None and st.session_state.struct_df is not None:
    func_connectome = st.session_state.func_df.values.copy()
    struct_connectome = st.session_state.struct_df.values.copy()
    np.fill_diagonal(func_connectome, 0)
    np.fill_diagonal(struct_connectome, 0)
    func_connectome[np.isinf(func_connectome)] = 0
    struct_connectome[np.isinf(struct_connectome)] = 0
	
# Reset buttons when a new file/folder is selected
if uploaded_file or folder_path:
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

if st.session_state.get("launch_btn", False):
    pred_accuracy = []
    predicted_regional = []     
    true_regional = []
    st.session_state.propagation_maps = []
    st.session_state.predicted_regional_scaled = []


    if st.session_state.masked_df is not None and not st.session_state.masked_df.empty:

        # Create results folder if it does not exist
        output_folder = BASE_DIR / "results"
        output_folder.mkdir(parents=True, exist_ok=True)
		
        # Loop over each subject/column in masked_df
        for subject_index in range(st.session_state.masked_df.shape[1]):
            feature_vector = st.session_state.masked_df.iloc[:, subject_index].values  # (n_rois,)
            filename = masked_df.columns[subject_index]  # use uploaded filename

            # --- Functional connectome ---
            connectome_FC = np.clip(func_connectome, a_min=0, a_max=None)
            product_FC = feature_vector[:, np.newaxis] * connectome_FC
            product_FC_sym = 0.5 * (product_FC + product_FC.T)

            # --- Structural covariance ---
            connectome_SC = np.clip(struct_connectome, a_min=0, a_max=None)
            product_SC = feature_vector[:, np.newaxis] * connectome_SC
            product_SC_sym = 0.5 * (product_SC + product_SC.T)

            # --- Average both matrices ---
            avg_BOTH_sym = 0.5 * (product_FC_sym + product_SC_sym)
            rob_scaler = RobustScaler()
            triu_idx = np.triu_indices_from(avg_BOTH_sym, k=1)
            upper_vals = avg_BOTH_sym[triu_idx].reshape(-1, 1)
            upper_vals_scaled = rob_scaler.fit_transform(upper_vals).flatten()
            avg_BOTH_sym_scaled = np.zeros_like(avg_BOTH_sym)
            avg_BOTH_sym_scaled[triu_idx] = upper_vals_scaled
            avg_BOTH_sym_scaled = avg_BOTH_sym_scaled + avg_BOTH_sym_scaled.T
            np.fill_diagonal(avg_BOTH_sym_scaled, np.diag(avg_BOTH_sym))
            st.session_state.propagation_maps.append(avg_BOTH_sym_scaled)

            # --- Predict regional values ---
            pred_regional = avg_BOTH_sym.sum(axis=0)

            # --- Standardize predicted map before saving ---
            pred_regional_scaled = rob_scaler.fit_transform(pred_regional.reshape(-1, 1)).flatten()
            st.session_state.predicted_regional_scaled.append(pred_regional_scaled)

            # --- Correlation with true feature vector ---
            corr_pos, _ = spearmanr(pred_regional, feature_vector)
            pred_accuracy.append(corr_pos)

            # --- Store results ---
            predicted_regional.append(pred_regional_scaled)
            true_regional.append(feature_vector)
            
            # --- Compute residuals (Observed - Predicted) ---
            model = LinearRegression().fit(pred_regional.reshape(-1, 1), feature_vector)
            y_hat = model.predict(pred_regional.reshape(-1, 1))
            residuals = feature_vector - y_hat

            # --- Save CSVs ---
            pd.DataFrame(pred_regional_scaled).to_csv(output_folder / f"{filename}_pred_map.csv")
            pd.DataFrame(feature_vector).to_csv(output_folder / f"{filename}_obs_map.csv")
            pd.DataFrame(avg_BOTH_sym_scaled).to_csv(output_folder / f"{filename}_propagationmap.csv")
            pd.DataFrame(residuals).to_csv(output_folder / f"{filename}_residualmap.csv")

        # --- Show summary ---
        st.success("🚀 Propagation mapping complete!")

        # --- Compute summary stats if more than one subject ---
        if len(pred_accuracy) == 1:
            st.header(f"Prediction Accuracy for {masked_df.columns[0]}")            
            st.write("Spearman's rank order:", pred_accuracy[0])

        else:
            st.header("Prediction Accuracy Across Subjects") 

            pred_accuracy_array = np.array(pred_accuracy)
            summary_stats = {
                "Mean": np.mean(pred_accuracy_array),
                "Std": np.std(pred_accuracy_array),
                "Min": np.min(pred_accuracy_array),
                "Max": np.max(pred_accuracy_array)
            }
            st.subheader("Summary statistics across all subjects")
            st.table(pd.DataFrame([summary_stats]))

            # --- Density plot of prediction accuracy ---
            import seaborn as sns
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6,4))
            sns.kdeplot(pred_accuracy_array, fill=True, clip=(0,1), ax=ax)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Spearman correlation")
            ax.set_ylabel("Density")
            st.pyplot(fig)

# ------------------ PLOTTING ------------------
if st.session_state.get("plot_prop_btn", False) and masked_df is not None and not masked_df.empty and st.session_state.propagation_maps:

    # Create a container for the atlas image
    one, two, three, four, five = st.columns(5)
    with two:
        atlas_img_path = BASE_DIR / "miscellaneous" / "schaefer_net.png"
        st.image(
            str(atlas_img_path),
            width=400,
            caption="Schaefer-400 7 Networks with 14 Subcortical, and 7 Cerebellar Regions"
        )

    st.header(f"Plot Propagation Map of: {masked_df.columns[0]}")
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

    @st.fragment
    def render_heatmap():
        prop_map = propagation_maps[0]
        connectome_matrix = pd.DataFrame(prop_map, index=roi_labels, columns=roi_labels)

        # Sort by network
        labels_info['Order'] = labels_info['Network']
        sorted_labels = labels_info.sort_values('Order')['Label'].tolist()
        reordered_matrix = connectome_matrix.loc[sorted_labels, sorted_labels]

        network_counts = labels_info.groupby('Network').size()
        network_names = network_counts.index.tolist()
        network_colors = {n: labels_info[labels_info['Network']==n]['colors'].unique()[0] for n in network_names}

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
        st.pyplot(plt)
        plt.close()

    with col1:
        render_heatmap()

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
  if st.session_state.masked_df is not None and not masked_df.empty and st.session_state.predicted_regional_scaled:
    st.header(f"Compare Maps of: {st.session_state.masked_df.columns[0]}")
    # ----------------- OBSERVED MAP -----------------

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

        st.header(f"Accuracy Joint Plot of: {masked_df.columns[0]}")

	# Create two columns for side-by-side display
        col1, col2, col3 = st.columns(3)

        atlas_csv = "./atlases/listnames_ATLAS_Schaefer7.csv"
        labels_info = pd.read_csv(atlas_csv)
        scaler = RobustScaler()
        df_obs_scaled = scaler.fit_transform(df_obs['value'].values.reshape(-1, 1))
        df_obs_scaled = pd.DataFrame(df_obs_scaled)

        combined = pd.concat([df_obs_scaled, df_pred['value']], axis=1)
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
        plots_folder = BASE_DIR / "plots"
        plots_folder.mkdir(parents=True, exist_ok=True)
        plot_file = plots_folder / f"{masked_df.columns[0]}_accuracy_jointplot.png"
        g.fig.tight_layout()
        g.fig.savefig(plot_file, dpi=300, bbox_inches='tight')

        # Display in Streamlit
        st.image(plot_file, width=670)  # width in pixels
        plt.close(g.fig)
