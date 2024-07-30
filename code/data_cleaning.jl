using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")

using CSV, DataFrames

# Function to read and process the file
function read_data_with_metadata(file_path, out_path)
    # Open the file
    file = open(file_path, "r")
    try
        # Read all lines until the data starts
        lines = readlines(file)
        
        # Extract column names from line 17-22
        column_names = split(join(lines[17:22], " "), r"\d+\)")  # column names from lines 17-22, separated by some number + ") "
        column_names = [replace(name, " " => "") for name in column_names] # remove spaces
        column_names = column_names[2:end] #remove first cell (empty cell)
        # Read CSV data starting from line 24
        data = CSV.File(file_path; header=column_names, skipto=24) |> DataFrame

        # Writing the DataFrame to a CSV file
        CSV.write(out_path, data)
        println("Data saved to '$out_path'")
    finally
        close(file)
    end
end