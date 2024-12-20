# Source the external R script containing the processing function
source("code/data_cleaning.R")
source("code/exploratory_analysis.R")

# Function to process all Trial#.txt files in the data folder
process_all_trials <- function(folder_name) {
  data_folder <- file.path(folder_name, "data/robot")
  results_folder <- file.path(folder_name, "results/robot")
  
  if (!dir.exists(results_folder)) {
    dir.create(results_folder)
  }
  
  raw_files <- list.files(data_folder, pattern = "trial\\d+\\.txt$", full.names = TRUE)
  
  for (file in raw_files) {
    clean_file(file)
  }
  
  cleaned_files <- list.files(data_folder, pattern = "trial\\d+_cleaned\\.csv$", full.names = TRUE)
  
  for (file in cleaned_files) {
    visualize_angles(file, results_folder)
  }
}

# Example usage
folder_name <- "Pilot1"
process_all_trials(folder_name)
