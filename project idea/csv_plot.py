import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class DataPlotter:
    def __init__(self, file_path):
        # Load the CSV data with the original structure
        self.file_path = file_path
        self.data = pd.read_csv(file_path, encoding="ISO-8859-1")

    def display_all_columns(self):
        print("\nColumns with their data types:")
        for col in self.data.columns:
            print(f"{col} (data type: {self.data[col].dtype})")

    def get_plot_type(self):
        print("\nSelect the type of plot:")
        print("1. 1D Plot (Single column)")
        print("2. 2D Plot (Two columns)")
        print("3. 3D Plot (Three columns)")

        plot_type = input("Enter your choice (1, 2, or 3): ")
        while plot_type not in {'1', '2', '3'}:
            print("Invalid choice. Please enter 1, 2, or 3.")
            plot_type = input("Enter your choice (1, 2, or 3): ")
        return plot_type

    def plot_1d(self):
        print("\nAvailable 1D plot types:")
        print("1. Histogram")
        print("2. Line Plot")
        print("3. Box Plot")
        print("4. Violin Plot")  # Added
        print("5. KDE Plot")  # Added

        plot_choice = input("Select the plot type (1, 2, 3, 4, or 5): ")
        while plot_choice not in {'1', '2', '3', '4', '5'}:
            print("Invalid choice. Please select 1, 2, 3, 4, or 5.")
            plot_choice = input("Select the plot type (1, 2, 3, 4, or 5): ")

        selected_column = input("\nEnter the column name for the 1D plot: ")
        while selected_column not in self.data.columns:
            print(f"Invalid choice. {selected_column} is not a valid column.")
            selected_column = input("Enter the column name for the 1D plot: ")

        print(f"\nFirst few rows of the selected column '{selected_column}':")
        print(self.data[[selected_column]].head())

        min_val = self.data[selected_column].min()
        max_val = self.data[selected_column].max()
        precision = max(self.data[selected_column].apply(lambda x: len(str(x).split('.')[-1]) if isinstance(x, float) else 0))
        print(f"\nCurrent range of '{selected_column}' is from {min_val} to {max_val}.")
        print(f"Please enter your desired range with precision up to {precision} decimal places.")

        min_input = float(input(f"Enter minimum value for {selected_column}: "))
        max_input = float(input(f"Enter maximum value for {selected_column}: "))

        min_input = round(min_input, precision)
        max_input = round(max_input, precision)

        print(f"\nSelected range for '{selected_column}' is from {min_input} to {max_input}.")

        filtered_data = self.data[(self.data[selected_column] >= min_input) & (self.data[selected_column] <= max_input)]
        print(f"\nNumber of entries within the selected range: {len(filtered_data)}")

        title = input("Enter a title for the plot: ")

        plt.figure(figsize=(8, 6))

        if plot_choice == '1':  # Histogram
            plt.hist(filtered_data[selected_column], bins=30, color='blue', alpha=0.7)
            plt.xlabel(selected_column)
            plt.ylabel('Frequency')
        elif plot_choice == '2':  # Line Plot
            plt.plot(filtered_data[selected_column], color='blue', alpha=0.7)
            plt.xlabel('Index')
            plt.ylabel(selected_column)
        elif plot_choice == '3':  # Box Plot
            sns.boxplot(data=filtered_data[selected_column], color='blue')
            plt.ylabel(selected_column)
        elif plot_choice == '4':  # Violin Plot (New)
            sns.violinplot(data=filtered_data[selected_column], color='blue')
            plt.ylabel(selected_column)
        elif plot_choice == '5':  # KDE Plot (New)
            sns.kdeplot(filtered_data[selected_column], color='blue')
            plt.xlabel(selected_column)

        plt.title(title)
        plt.grid(True)
        plt.show()

    def plot_2d(self):
        print("\nAvailable 2D plot types:")
        print("1. Scatter Plot")
        print("2. Line Plot")
        print("3. Bar Plot")
        print("4. Heatmap")  # Added
        print("5. Pairplot")  # Added

        plot_choice = input("Select the plot type (1, 2, 3, 4, or 5): ")
        while plot_choice not in {'1', '2', '3', '4', '5'}:
            print("Invalid choice. Please select 1, 2, 3, 4, or 5.")
            plot_choice = input("Select the plot type (1, 2, 3, 4, or 5): ")

        x_column = input("\nEnter the column name for the x-axis: ")
        while x_column not in self.data.columns:
            print(f"Invalid choice. {x_column} is not a valid column.")
            x_column = input("Enter the column name for the x-axis: ")

        y_column = input("\nEnter the column name for the y-axis: ")
        while y_column not in self.data.columns:
            print(f"Invalid choice. {y_column} is not a valid column.")
            y_column = input("Enter the column name for the y-axis: ")

        print(f"\nFirst few rows of the selected columns '{x_column}' and '{y_column}':")
        print(self.data[[x_column, y_column]].head())

        for col in [x_column, y_column]:
            min_val = self.data[col].min()
            max_val = self.data[col].max()
            precision = max(self.data[col].apply(lambda x: len(str(x).split('.')[-1]) if isinstance(x, float) else 0))
            print(f"\nCurrent range of '{col}' is from {min_val} to {max_val}.")
            print(f"Please enter your desired range for {col} with precision up to {precision} decimal places.")

            min_input = float(input(f"Enter minimum value for {col}: "))
            max_input = float(input(f"Enter maximum value for {col}: "))

            min_input = round(min_input, precision)
            max_input = round(max_input, precision)

            print(f"\nSelected range for '{col}' is from {min_input} to {max_input}.")

            filtered_data = self.data[(self.data[col] >= min_input) & (self.data[col] <= max_input)]
            print(f"\nNumber of entries within the selected range for '{col}': {len(filtered_data)}")

        title = input("Enter a title for the plot: ")

        plt.figure(figsize=(8, 6))

        if plot_choice == '1':  # Scatter Plot
            plt.scatter(filtered_data[x_column], filtered_data[y_column], color='blue', alpha=0.7)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
        elif plot_choice == '2':  # Line Plot
            plt.plot(filtered_data[x_column], filtered_data[y_column], color='blue', alpha=0.7)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
        elif plot_choice == '3':  # Bar Plot
            plt.bar(filtered_data[x_column], filtered_data[y_column], color='blue', alpha=0.7)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
        elif plot_choice == '4':  # Heatmap (New)
            corr_matrix = filtered_data.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title(title)
        elif plot_choice == '5':  # Pairplot (New)
            sns.pairplot(filtered_data[[x_column, y_column]], hue="category_column")  # 'category_column' needs to be a categorical column

        plt.title(title)
        plt.grid(True)
        plt.show()

    def plot_3d(self):
        print("\nAvailable 3D plot types:")
        print("1. 3D Scatter Plot")
        print("2. 3D Surface Plot")
        print("3. Time-Series Plot")  # Added

        plot_choice = input("Select the plot type (1, 2, or 3): ")
        while plot_choice not in {'1', '2', '3'}:
            print("Invalid choice. Please select 1, 2, or 3.")
            plot_choice = input("Select the plot type (1, 2, or 3): ")

        x_column = input("\nEnter the column name for the x-axis: ")
        while x_column not in self.data.columns:
            print(f"Invalid choice. {x_column} is not a valid column.")
            x_column = input("Enter the column name for the x-axis: ")

        y_column = input("\nEnter the column name for the y-axis: ")
        while y_column not in self.data.columns:
            print(f"Invalid choice. {y_column} is not a valid column.")
            y_column = input("Enter the column name for the y-axis: ")

        z_column = input("\nEnter the column name for the z-axis: ")
        while z_column not in self.data.columns:
            print(f"Invalid choice. {z_column} is not a valid column.")
            z_column = input("Enter the column name for the z-axis: ")

        print(f"\nFirst few rows of the selected columns '{x_column}', '{y_column}', and '{z_column}':")
        print(self.data[[x_column, y_column, z_column]].head())

        for col in [x_column, y_column, z_column]:
            min_val = self.data[col].min()
            max_val = self.data[col].max()
            precision = max(self.data[col].apply(lambda x: len(str(x).split('.')[-1]) if isinstance(x, float) else 0))
            print(f"\nCurrent range of '{col}' is from {min_val} to {max_val}.")
            print(f"Please enter your desired range for {col} with precision up to {precision} decimal places.")

            min_input = float(input(f"Enter minimum value for {col}: "))
            max_input = float(input(f"Enter maximum value for {col}: "))

            min_input = round(min_input, precision)
            max_input = round(max_input, precision)

            print(f"\nSelected range for '{col}' is from {min_input} to {max_input}.")

            filtered_data = self.data[(self.data[col] >= min_input) & (self.data[col] <= max_input)]
            print(f"\nNumber of entries within the selected range for '{col}': {len(filtered_data)}")

        title = input("Enter a title for the plot: ")

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        if plot_choice == '1':  # 3D Scatter Plot
            ax.scatter(filtered_data[x_column], filtered_data[y_column], filtered_data[z_column], color='blue', alpha=0.7)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_zlabel(z_column)
        elif plot_choice == '2':  # 3D Surface Plot
            X = filtered_data[x_column].values
            Y = filtered_data[y_column].values
            Z = filtered_data[z_column].values
            ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
        elif plot_choice == '3':  # Time-Series Plot (New)
            plt.plot(filtered_data[x_column], filtered_data[y_column], color='blue', alpha=0.7)
            plt.xlabel(x_column)
            plt.ylabel(y_column)

        ax.set_title(title)
        plt.show()

    def plot(self):
        self.display_all_columns()
        plot_type = self.get_plot_type()

        if plot_type == '1':  # 1D Plot
            self.plot_1d()
        elif plot_type == '2':  # 2D Plot
            self.plot_2d()
        elif plot_type == '3':  # 3D Plot
            self.plot_3d()

# Example usage:
file_path = '/content/mappings.csv'
plotter = DataPlotter(file_path)
plotter.plot()
6-4