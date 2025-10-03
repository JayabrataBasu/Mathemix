from src.mathemix.core.data import DataFrame

def run_cli():
    """
    Runs the main Command-Line Interface (CLI) loop.
    """
    print("--- Mathemix Statistics ---")
    print("Type 'exit' to quit.")

    # This will hold our main dataset
    df = DataFrame()

    while True:
        command_input = input(">> ").strip()

        if not command_input:
            continue

        if command_input.lower() == 'exit':
            print("Exiting Mathemix.")
            break

        parts = command_input.split()
        command = parts[0].lower()
        args = parts[1:]

        if command == 'load':
            if not args:
                print("Usage: load <filepath>")
                continue
            filepath = args[0]
            message = df.load_csv(filepath)
            print(message)
            # For now, let's also see the DataFrame's representation
            print(df)
        # This code replaces the 'else' block in the while loop in main.py

        elif command == 'describe':
            results = df.describe()
            if isinstance(results, str):
                # This handles the "No data loaded" error message
                print(results)
            else:
                # Format and print the results table
                header = (f"{'Variable':<15} | {'Obs':>7} | {'Mean':>12} | "
                          f"{'Std. Dev.':>12} | {'Min':>10} | {'Max':>10}")
                print(header)
                print("-" * len(header))
                for row in results:
                    print(f"{row['Variable']:<15} | {row['Obs']:>7} | {row['Mean']:>12.4f} | "
                          f"{row['Std. Dev.']:>12.4f} | {row['Min']:>10} | {row['Max']:>10}")
        else:
            print(f"Unknown command: '{command}'")

if __name__ == "__main__":
    run_cli()