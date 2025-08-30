import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

def read_active_sites(filename):
    """Reads a simple text file of residue numbers."""
    try:
        with open(filename, 'r') as f:
            # Read numbers, converting them to integers, and handle empty lines
            sites = [int(line.strip()) for line in f if line.strip()]
        print(f"Successfully loaded {len(sites)} active site residues from '{filename}'.")
        return sites
    except FileNotFoundError:
        print(f"Warning: Active site file '{filename}' not found. No sites will be highlighted.")
        return []
    except ValueError:
        print(f"Warning: Could not read numbers from '{filename}'. Please ensure it contains one number per line.")
        return []
        
def residues_to_intervals(residues):
    """
    Convert a sorted list of residue numbers into intervals.
    E.g., [5,6,7,10,11] -> [(5,7), (10,11)]
    """
    sorted_res = sorted(residues)
    intervals = []
    start = prev = sorted_res[0]
    for r in sorted_res[1:]:
        if r == prev + 1:
            prev = r
        else:
            intervals.append((start, prev))
            start = prev = r
    intervals.append((start, prev))
    return intervals


def plot_dashboard_with_active_sites():
    """
    Creates a dashboard from a CSV and highlights active site residues on the RMSF plot.
    """
    # --- 1. Define Input Files ---
    csv_filename = 'nirA_rmsf_data.csv'
    active_site_filename = 'active_sites.txt' # <-- New file for active sites

    # --- 2. Load Data ---
    try:
        df = pd.read_csv(csv_filename)
        print(f"Successfully loaded data from '{csv_filename}'.")
    except FileNotFoundError:
        print(f"Error: The file '{csv_filename}' was not found. Please check the filename.")
        return

    # Load active site residues
    active_sites = read_active_sites(active_site_filename)

    # --- 3. Extract Data and Create Matrices ---
    residue_indices = df.iloc[:, 0].to_numpy()
    file_labels = df.columns[1:].tolist()
    rmsf_matrix = df.iloc[:, 1:].to_numpy().T

    print(f"Found {len(file_labels)} simulations: {file_labels}")

    # --- 4. Calculations for Plots ---
    min_per_row = np.min(rmsf_matrix, axis=1, keepdims=True)
    max_per_row = np.max(rmsf_matrix, axis=1, keepdims=True)
    range_per_row = max_per_row - min_per_row
    range_per_row[range_per_row == 0] = 1
    normalized_matrix = (rmsf_matrix - min_per_row) / range_per_row

    avg_per_residue = np.mean(rmsf_matrix, axis=0)
    min_per_residue = np.min(rmsf_matrix, axis=0)
    max_per_residue = np.max(rmsf_matrix, axis=0)

    # --- 5. Setup the Figure Layout ---
    fig = plt.figure(figsize=(20, 1.0 * len(file_labels) + 3), dpi=300)
    gs = fig.add_gridspec(nrows=2, ncols=2, 
                          height_ratios=[2.5, len(file_labels)], 
                          width_ratios=[40, 1], 
                          wspace=0.05, hspace=0.03)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[1, 0], sharex=ax_top)
    cax = fig.add_subplot(gs[1, 1])
    fig.suptitle('Comprehensive RMSF Fluctuation Dashboard', fontsize=20, fontweight='bold')
    cmap_heat = plt.get_cmap('YlOrRd')

    # --- 6. Plot Average Flexibility Profile (Top) with Active Sites ---
    line_color = '#994C00'
    fill_color = '#f5ab7a'

    ax_top.set_title('Average Flexibility Profile with Active Site Regions', fontsize=12)
    ax_top.plot(residue_indices, avg_per_residue, color=line_color, lw=1.5, label='Average RMSF')
    ax_top.fill_between(residue_indices, min_per_residue, max_per_residue,
                        color=fill_color, alpha=0.3, label='Min-Max Range')
    
    # --- MODIFICATION: Add shaded regions for active sites ---
    intervals = residues_to_intervals(active_sites)
    for i, (start, end) in enumerate(intervals):
      label = 'Active Site' if i == 0 else ""
      # Use +/- 0.5 to cover the residue entirely
      ax_top.axvspan(start - 0.75, end + 0.75, color='lightgreen', alpha=0.5, zorder=0, label=label, edgecolor='none', linewidth=0)


    ax_top.set_ylabel(r'Avg. RMSF ($\AA$)')
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.legend(fontsize='small')
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax_top.xaxis.set_major_locator(mticker.MultipleLocator(50))

    # --- 7. Plot the Heatmap ---
    im = ax_heatmap.imshow(normalized_matrix, aspect='auto', cmap=cmap_heat, interpolation='nearest',
                           extent=[residue_indices.min(), residue_indices.max(), len(file_labels)-0.5, -0.5])
    ax_heatmap.set_ylabel('Simulations', fontsize=12)
    ax_heatmap.set_xlabel('Residue Number', fontsize=12)
    ax_heatmap.set_yticks(np.arange(len(file_labels)))
    ax_heatmap.set_yticklabels(file_labels, fontsize=9)

    # --- 8. Plot the Colorbar for the Heatmap ---
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Normalized Fluctuation', rotation=270, labelpad=15, fontsize=10)

    # --- 9. Final layout and save ---
    fig.tight_layout(rect=[0, 0, 1, 0.96]) 
    output_filename = 'RMSF_Dashboard_with_ActiveSites.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSuccessfully saved the dashboard plot to '{output_filename}'")
    
# --- Main execution block ---
if __name__ == '__main__':
    plot_dashboard_with_active_sites()
