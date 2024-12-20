if (!require("zoo")) install.packages("zoo", dependencies = TRUE)
library(zoo)

# Function to visualize angles and save plots to the specified results folder
visualize_angles <- function(data_path, results_folder) {
  data <- read.csv(data_path, header = TRUE)
  RollingMean <- rollapply(data$APAngle.deg., width = 15000, FUN = mean, by = 500, align = "center")
  RollingVariance <- rollapply(data$APAngle.deg., width = 15000, FUN = var, by = 500, align = "center")
  
  # Remove _cleaned.csv from the basename
  base_name <- sub("_cleaned\\.csv$", "", basename(data_path))
  plot_path_mean <- file.path(results_folder, paste0(base_name, "_RollingMean.png"))
  plot_path_variance <- file.path(results_folder, paste0(base_name, "_RollingVariance.png"))
  
  png(plot_path_mean)
  plot(RollingMean, type = "l", main = paste("Rolling Mean of APAngle", base_name), xlab = "Index", ylab = "Mean")
  dev.off()
  
  png(plot_path_variance)
  plot(RollingVariance, type = "l", main = paste("Rolling Variance of APAngle", base_name), xlab = "Index", ylab = "Variance")
  dev.off()
}