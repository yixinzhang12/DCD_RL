# Load necessary libraries
if (!require("data.table")) install.packages("data.table", dependencies = TRUE)
library(data.table)

# Function to read and process a single file
clean_file <- function(file_path) {
  # Read all lines from the file
  lines <- readLines(file_path)
  
  # Find the line numbers for the last semicolon and the line with -----
  last_semicolon_line <- max(grep(";", lines))
  dash_line <- grep("-----", lines)[1]
  
  # Extract column names from lines between last semicolon and -----
  column_lines <- lines[(last_semicolon_line + 1):(dash_line - 1)]
  
  # Extract column names from lines 17-22
  column_names <- unlist(strsplit(paste(column_lines, collapse = " "), "\\d+\\)"))
  column_names <- gsub(" ", "", column_names) # remove spaces
  column_names <- column_names[-1] # remove first cell (empty cell)
  
  # Read CSV data starting from line 24
  data <- fread(file_path, skip = 23, header = FALSE) # skip first 23 lines
  setnames(data, column_names)
  
  # Construct the output file path
  output_file <- sub("\\.txt$", "_cleaned.csv", file_path)
  
  # Writing the DataFrame to a CSV file
  fwrite(data, output_file)
  print(paste("Cleaned data saved to", output_file))
}

