import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os

def normalize_data(data):
    """
    Normalize the raster data to the range [0, 1].
    """
    data_min = np.min(data)
    data_max = np.max(data)
    normalized_data = (data - data_min) / (data_max - data_min)  # Min-Max normalization
    return normalized_data

def equalize_histogram(data):
    """
    Perform histogram equalization on the raster data.
    """
    # Flatten the data to 1D array for histogram equalization
    flattened_data = data.flatten()
    
    # Compute histogram and cumulative distribution function (CDF)
    hist, bins = np.histogram(flattened_data, bins=256, range=(0, 255))
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()  # Normalize to 1

    # Interpolate using CDF to map the input data to new values
    data_equalized = np.interp(flattened_data, bins[:-1], cdf_normalized * 255).reshape(data.shape)

    return data_equalized

def display_geotiff(file_path, normalize=False, equalize=False):
    try:
        # Open the GeoTIFF file
        with rasterio.open(file_path) as dataset:
            # Get the number of bands
            num_bands = dataset.count
            print(f"File has {num_bands} bands for {file_path}")

            # Read all the bands into a list (keeping them separate)
            band_data = [dataset.read(i + 1) for i in range(num_bands)]

            # Normalize the data if requested
            if normalize:
                print(f"Normalizing data for {file_path}")
                band_data = [normalize_data(band) for band in band_data]

            # Perform histogram equalization if requested
            if equalize:
                print(f"Performing histogram equalization for {file_path}")
                band_data = [equalize_histogram(band) for band in band_data]

            # Modify title to indicate normalization or equalization
            title_prefix = ""
            if normalize:
                title_prefix = "Normalized "
            if equalize:
                title_prefix += "Equalized "

            # If it's a multi-band file (more than 1 band)
            if num_bands > 1:
                print("Displaying the multi-band data by default.")

                # For 3 bands, assume it's RGB and stack them
                if num_bands == 3:
                    rgb_image = np.dstack(band_data)  # Stack them into a 3D array for RGB display
                    plt.imshow(rgb_image)  # Display the RGB image
                    plt.title(f"{title_prefix}Multi-Band Data (RGB) for {file_path}")
                else:
                    # For more than 3 bands, just show the image as a multi-band data
                    print(f"Displaying the first 3 bands as a composite for {file_path}")
                    rgb_image = np.dstack(band_data[:3])  # Stack the first 3 bands
                    plt.imshow(rgb_image)  # Display the composite image
                    plt.title(f"{title_prefix}First 3 Bands (Composite) for {file_path}")

                plt.axis('off')
                plt.show()

                # Ask user if they want to display the bands separately
                choice = input("Do you want to display the bands separately? (y/n): ").strip().lower()
                if choice == 'y':
                    print(f"Displaying {num_bands} bands separately for {file_path}")

                    # Create subplots with the number of rows and columns based on the number of bands
                    cols = 2  # Fixed number of columns for layout
                    rows = (num_bands + 1) // cols  # Calculate rows based on number of bands

                    # Create subplots
                    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 5 * rows))

                    # Flatten the axes to make indexing easier if there's more than 1 row
                    axes = axes.flatten()

                    # Loop through all the bands and display them
                    for band_num in range(1, num_bands + 1):
                        band = band_data[band_num - 1]

                        # Get the current subplot
                        ax = axes[band_num - 1]

                        # Display the band on the subplot
                        im = ax.imshow(band, cmap='gray')
                        ax.set_title(f"{title_prefix}Band {band_num}")
                        ax.axis('off')  # Hide the axes for a cleaner display

                    # Hide any extra subplots if there are fewer bands than subplots
                    for i in range(num_bands, len(axes)):
                        axes[i].axis('off')

                    # Add a colorbar outside the subplots (adjusted layout to prevent overlap)
                    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])  # Position of colorbar (right side)
                    fig.colorbar(im, cax=cbar_ax)

                    # Adjust layout manually to prevent overlap (use subplots_adjust)
                    fig.subplots_adjust(right=0.9)  # Leave space on the right for the colorbar

                    plt.show()

                else:
                    print("Continuing with the default multi-band display.")

            else:
                print("Displaying single-band data.")

                # If it's a single-band file, just display the first band
                band_data = dataset.read(1)
                if normalize:
                    band_data = normalize_data(band_data)
                if equalize:
                    band_data = equalize_histogram(band_data)

                plt.imshow(band_data, cmap='gray')
                plt.title(f"{title_prefix}Single Band of {file_path}")
                plt.colorbar()
                plt.show()

    except Exception as e:
        print(f"An error occurred while loading the file: {e}")

def stack_bands(file_paths):
    """
    Stack multiple single-banded GeoTIFF files into a multi-band array.
    Now, this function can also stack multiple bands for composite visualization.
    """
    try:
        band_list = []

        # Read each file and ensure it has only 1 band
        for file_path in file_paths:
            with rasterio.open(file_path) as dataset:
                if dataset.count != 1:
                    print(f"File {file_path} is not single-banded. Skipping.")
                    continue
                band_data = dataset.read(1)
                band_list.append(band_data)

        # Ensure at least two bands to stack
        if len(band_list) < 2:
            print("Not enough single-banded files to stack. Need at least 2.")
            return None

        # Stack bands along a new axis (multi-band array)
        stacked_data = np.stack(band_list, axis=-1)
        print(f"Stacked {len(band_list)} bands successfully.")

        return stacked_data

    except Exception as e:
        print(f"An error occurred while stacking bands: {e}")
        return None

def display_stacked_data(stacked_data, visualize_bands=(0, 1, 2)):
    """
    Display the stacked data as a composite of the selected bands.
    """
    try:
        # Normalize the stacked data for visualization
        stacked_data = stacked_data.astype(np.float32)  # Convert to float for better scaling
        stacked_data -= stacked_data.min()  # Normalize to 0
        stacked_data /= stacked_data.max()  # Normalize to 1

        # Create the RGB composite image
        rgb_image = stacked_data[:, :, visualize_bands]

        # Display the RGB composite image
        plt.imshow(rgb_image)
        plt.title("Composite Image (False Color or Multi-band)")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"An error occurred while displaying the stacked data: {e}")

def process_folder(folder_path):
    """
    Process all GeoTIFF files in a folder, with options for normalization and stacking.
    """
    # List all GeoTIFF files in the folder
    geotiff_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]

    if not geotiff_files:
        print("No GeoTIFF files found in the folder.")
        return

    print(f"Found {len(geotiff_files)} GeoTIFF files in the folder:")
    for idx, file_name in enumerate(geotiff_files, 1):
        print(f"{idx}. {file_name}")

    # Ask the user which files they want to display first
    choice = input("Enter the numbers of the files you want to display (comma separated): ")
    selected_files = [int(num.strip()) - 1 for num in choice.split(",")]

    # Display selected files
    for idx in selected_files:
        geotiff_file = geotiff_files[idx]
        file_path = os.path.join(folder_path, geotiff_file)
        display_geotiff(file_path)

    # Ask the user which files they want to normalize after viewing
    normalize_choice = input("Do you want to normalize any of the displayed files? (y/n): ").strip().lower()
    if normalize_choice == 'y':
        # Let user select files to normalize
        normalize_files = input("Enter the numbers of the files you want to normalize (comma separated): ")
        normalize_files = [int(num.strip()) - 1 for num in normalize_files.split(",")]

        # Normalize and display the selected files
        for idx in normalize_files:
            geotiff_file = geotiff_files[idx]
            file_path = os.path.join(folder_path, geotiff_file)
            display_geotiff(file_path, normalize=True)

    # After normalization, ask if the user wants to perform histogram equalization
    equalize_choice = input("Do you want to perform histogram equalization on any of the normalized files? (y/n): ").strip().lower()
    if equalize_choice == 'y':
        # Let user select files to equalize
        equalize_files = input("Enter the numbers of the files you want to equalize (comma separated): ")
        equalize_files = [int(num.strip()) - 1 for num in equalize_files.split(",")]

        # Perform histogram equalization and display the selected files
        for idx in equalize_files:
            geotiff_file = geotiff_files[idx]
            file_path = os.path.join(folder_path, geotiff_file)
            display_geotiff(file_path, normalize=True, equalize=True)

    # After histogram equalization, ask if the user wants to stack single-banded files
    stack_choice = input("Do you want to stack single-banded GeoTIFF files? (y/n): ").strip().lower()
    if stack_choice == 'y':
        # Filter only single-banded files
        single_banded_files = []
        for file_name in geotiff_files:
            file_path = os.path.join(folder_path, file_name)
            with rasterio.open(file_path) as dataset:
                if dataset.count == 1:
                    single_banded_files.append(file_path)

        if not single_banded_files:
            print("No single-banded GeoTIFF files available for stacking.")
        else:
            print(f"Found {len(single_banded_files)} single-banded files available for stacking.")
            # Ask user which single-banded files to stack
            print("Select the single-banded files to stack:")
            for i, file in enumerate(single_banded_files, 1):
                print(f"{i}. {file}")
            selected_stack_files = input("Enter the numbers of the files to stack (comma separated): ")
            selected_stack_files = [int(num.strip()) - 1 for num in selected_stack_files.split(",")]

            # Get the selected files and stack them
            stack_files = [single_banded_files[i] for i in selected_stack_files]
            stacked_data = stack_bands(stack_files)

            # Display the stacked data if stacking was successful
            if stacked_data is not None:
                print("Displaying the stacked data.")
                display_stacked_data(stacked_data)

def main():
    # Ask user if they want to load a single file or a folder
    choice = input("Do you want to process a (1) single file or (2) a folder? Enter 1 or 2: ")

    if choice == '1':
        # Ask for the file path of a single GeoTIFF
        file_path = input("Please enter the path to the GeoTIFF file: ")

        if not os.path.isfile(file_path):
            print("The file path is invalid.")
            return

        # Display the GeoTIFF file
        display_geotiff(file_path)

        # Ask if the user wants to normalize the file
        normalize_choice = input("Do you want to normalize the file? (y/n): ").strip().lower()
        normalize = normalize_choice == 'y'

        # Ask if the user wants to equalize the histogram
        equalize_choice = input("Do you want to perform histogram equalization on the file? (y/n): ").strip().lower()
        equalize = equalize_choice == 'y'

        # Display the processed version if user chooses so
        display_geotiff(file_path, normalize=normalize, equalize=equalize)

    elif choice == '2':
        # Ask for the folder path
        folder_path = input("Please enter the folder path containing GeoTIFF files: ")

        if not os.path.isdir(folder_path):
            print("The folder path is invalid.")
            return

        # Process the folder and display selected files first
        process_folder(folder_path)

    else:
        print("Invalid choice. Please run the program again and select either 1 or 2.")

if __name__ == "__main__":
    main()



