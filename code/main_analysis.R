# Source the external R script containing the processing function
source("code/data_cleaning.R")
source("code/exploratory_analysis.R")

# Function to process all Trial#.txt files in the data folder
process_all_trials <- function(folder_name) {
  data_folder <- file.path(folder_name, "data")
  results_folder <- file.path(folder_name, "results")
  
  if (!dir.exists(results_folder)) {
    dir.create(results_folder)
  }
  
  raw_files <- list.files(data_folder, pattern = "Trial\\d+\\.txt$", full.names = TRUE)
  
  for (file in raw_files) {
    read_and_process_file(file)
  }
  
  cleaned_files <- list.files(data_folder, pattern = "Trial\\d+_cleaned\\.csv$", full.names = TRUE)
  
  for (file in cleaned_files) {
    visualize_angles(file, results_folder)
  }
}

# Example usage
folder_name <- "Izzy_Pilot"
process_all_trials(folder_name)
