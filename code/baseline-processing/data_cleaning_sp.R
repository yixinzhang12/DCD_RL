# Load necessary libraries
if (!require("data.table")) install.packages("data.table", dependencies = TRUE)
library(data.table)
library(dplyr)

# Function to read and process a single file
clean_robot_file <- function(file_path) {
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

  # Sampling rate
  sampling_rate = 500
  # delta_t = 1 / sampling_rate
  # Compute angular velocities (angle/delta_t = angle * sampling rate)
  angular_velocities = diff(data$`APAngle(deg)`)*sampling_rate
  angular_velocities <- c(0, angular_velocities)
  # Add angular velocity as a new column to the angle data frame
  data$angular_velocity <- angular_velocities
  data <- data %>% select("APAngle(deg)", "angular_velocity", "EMGTrigger", "EMGSinewave")

  # Get the run-length encoding of the EMGTrigger column
  trigger_rle <- rle(data$EMGTrigger)

  # Calculate the cumulative lengths to find positions in the dataframe
  cumulative_lengths <- cumsum(trigger_rle$lengths)
  
  # Get the start position of a list of fives
  start_pos <- cumulative_lengths[1] + 1
  
  # adjust end position based on trial types (pre/post has 60s, trial has 300s)
  if (grepl("pre\\d+\\.txt$|post\\d+\\.txt$", file_path)) {
    cat("Processing 'pre' or 'post' file:", file_path, "\n")
    end_pos <- start_pos + 500*60 - 1
    # Subset the dataframe
    df_subset <- data[start_pos:end_pos, ]
  } else if (grepl("trial\\d+\\.txt$", file_path)) {
    cat("Processing 'trial' file:", file_path, "\n")
    # end_pos <- start_pos + 500*300 - 1
    end_pos <- start_pos + 500*180 - 1
    # Subset the dataframe
    df_subset <- data[start_pos:end_pos, ]
  } else {
    cat("Skipping file (pattern not matched):", file_path, "\n")
  }
  # Construct the output file path
  output_file <- sub("\\.txt$", "_cleaned.csv", file_path)
  
  # Writing the DataFrame to a CSV file
  fwrite(df_subset, output_file)
  print(paste("Cleaned data saved to", output_file))
}

clean_emg_file <- function(file_path) {
  data <- read.table(file_path, skip = 1, sep = "", header = FALSE)
  colnames(data) <- c("sine_wave", "CH1", "CH2", "CH3", "CH4", "TRG")
  data$TRG = round(data$TRG)
  # Get the run-length encoding of the EMGTrigger column
  trigger_rle <- rle(data$TRG)
  cumulative_lengths <- cumsum(trigger_rle$lengths)
  start_pos <- cumulative_lengths[1] + 1
  if (grepl("pre\\d+\\.txt$|post\\d+\\.txt$", file_path)) {
    cat("Processing 'pre' or 'post' file:", file_path, "\n")
    end_pos <- start_pos + 2000*60 - 1
    # Subset the dataframe
    df_subset <- data[start_pos:end_pos, ]
  } else if (grepl("trial\\d+\\.txt$", file_path)) {
    cat("Processing 'trial' file:", file_path, "\n")
    # end_pos <- start_pos + 2000*300 - 1
    end_pos <- start_pos + 2000*180 - 1
    # Subset the dataframe
    df_subset <- data[start_pos:end_pos, ]
  } else {
    cat("Skipping file (pattern not matched):", file_path, "\n")
  }
  # Construct the output file path
  output_file <- sub("\\.txt$", "_cleaned.csv", file_path)
  
  # Writing the DataFrame to a CSV file
  fwrite(df_subset, output_file)
  print(paste("Cleaned data saved to", output_file))
}  