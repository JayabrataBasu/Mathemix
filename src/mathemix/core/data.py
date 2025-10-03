import pyarrow.csv as pv
import pyarrow as pa

class DataFrame:
    """
    A container for a dataset, built on Apache Arrow's Table object.
    """
    def __init__(self):
        self.table: pa.Table | None = None

    def load_csv(self, filepath: str) -> str:
        """
        Loads data from a CSV file into the DataFrame.
        Includes robust error handling for a missing file.

        Args:
            filepath: The path to the CSV file.

        Returns:
            A string message indicating success or failure.
        """
        try:
            self.table = pv.read_csv(filepath)
            # Let's get some info for the success message
            num_rows = self.table.num_rows
            num_cols = self.table.num_columns
            return f"✅ Success: Loaded {num_rows} observations of {num_cols} variables."
        except FileNotFoundError:
            return f"❌ Error: The file '{filepath}' was not found."
        except Exception as e:
            return f"❌ Error: An unexpected error occurred: {e}"

    def __repr__(self) -> str:
        if self.table:
            return f"DataFrame with {self.table.num_rows} rows and {self.table.num_columns} columns."
        return "Empty DataFrame"
    
    # This code goes inside the DataFrame class in data.py

    def describe(self) -> list[dict] | str:
        """
        Calculates summary statistics for all numeric columns in the table.

        Returns:
            A list of dictionaries, each containing stats for a variable,
            or an error string if no data is loaded.
        """
        if self.table is None:
            return "❌ Error: No data loaded. Use the 'load' command first."

        import pyarrow.compute as pc

        results = []
        for idx, col in enumerate(self.table.columns):
            # Only calculate for columns that are a numeric type (int or float)
            if pa.types.is_floating(col.type) or pa.types.is_integer(col.type):
                count = pc.count(col).as_py()
                mean_val = pc.mean(col).as_py()
                mean = mean_val if mean_val is not None else float('nan')
                stddev_val = pc.sqrt(pc.variance(col)).as_py()
                stddev = stddev_val if stddev_val is not None else float('nan')
                min_val = pc.min(col).as_py()
                max_val = pc.max(col).as_py()
                col_name = self.table.schema.names[idx]
                results.append({
                    "Variable": col_name,
                    "Obs": count,
                    "Mean": mean,
                    "Std. Dev.": stddev,
                    "Min": min_val,
                    "Max": max_val
                })
        return results
        
        

