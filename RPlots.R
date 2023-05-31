total_params = c(1472.63617, 710.64193, 204.93281, 202.22145)


param_size = c(561.77, 271.09, 78.18, 77.14)
for_back_size = c(481.8, 69.08, 904.52, 1114.8)
total_size = c(1055.57, 975.17, 994.7, 1203.94)


models <- c("Baseline", "Slim", "Seperable", "Slim+Seperable" )


mod <- data.frame(total_params, param_size, for_back_size, total_size)

barplot(total_params*10000, col="red", name = c("Baseline", "Slim", "Seperable", "Slim+Seperable" ))

barplot(as.matrix(mod),beside = TRUE, col = c(rgb(1,0,0,0.1),rgb(1,0,0,0.3), rgb(1,0,0,0.65), rgb(1,0,0,1)), xlab = "models", names=c("Parameters 1:100.000", "Parameters in MB", "F&B propagation MB", "Total MB"))
#axis(side=1, tick = c("Parameters 1:100.000", "Parameters in MB", "forw-back propagation MB", "Total MB"))
legend("top",c("Baseline","Slim kernels", "Seperable kernels", "Slim+Seperable kernels"),
       fill = c(rgb(1,0,0,0.1),rgb(1,0,0,0.3), rgb(1,0,0,0.65), rgb(1,0,0,1))
)
