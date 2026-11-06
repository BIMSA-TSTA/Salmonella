rm(list = ls())

library(data.table)
library(ggplot2)

input_file <- "D:/bio_project/进化树/gene_presence_absence_01.csv"
out_dir    <- "D:/bio_project/fig/fig2"

cat("正在读取 0/1 矩阵...\n")
binary_df <- fread(input_file, data.table = FALSE)

rownames(binary_df) <- binary_df[[1]]
binary_df <- binary_df[, -1, drop = FALSE]

binary_matrix <- as.matrix(binary_df)
mode(binary_matrix) <- "numeric"

valid_values <- unique(as.vector(binary_matrix))
valid_values <- valid_values[!is.na(valid_values)]

if (!all(valid_values %in% c(0, 1))) {
  stop("输入文件不是纯 0/1 矩阵，请检查 gene_presence_absence_01.csv。")
}

cat("矩阵维度（gene x genome）: ", nrow(binary_matrix), " x ", ncol(binary_matrix), "\n")

pa_logical <- t(binary_matrix) > 0

n_genomes <- nrow(pa_logical)
n_genes   <- ncol(pa_logical)

cat("Genome 数量: ", n_genomes, "\n")
cat("Gene cluster 数量: ", n_genes, "\n\n")

permutations <- 100

set.seed(123)

pan_matrix  <- matrix(0L, nrow = n_genomes, ncol = permutations)
core_matrix <- matrix(0L, nrow = n_genomes, ncol = permutations)

cat("开始进行真实随机抽样，共", permutations, "次...\n\n")

for (p in seq_len(permutations)) {
  cat("正在处理第", p, "/", permutations, "次随机抽样...\n")
  
  # 随机打乱 genome 顺序
  shuffled_pa <- pa_logical[sample.int(n_genomes), , drop = FALSE]
  
  # 记录每个 gene 在当前已加入 genome 中出现的累计次数
  presence_counts <- integer(n_genes)
  
  for (i in seq_len(n_genomes)) {
    # 累加第 i 个 genome 的存在情况
    presence_counts <- presence_counts + shuffled_pa[i, ]
    
    # Pan-genome: 只要出现过一次就算
    pan_matrix[i, p] <- sum(presence_counts > 0L)
    
    # 95% Soft-core:
    # 出现次数 >= 当前已加入 genome 数的 95%
    # 用 ceiling 保证阈值是整数且逻辑明确
    softcore_threshold <- ceiling(0.95 * i)
    core_matrix[i, p]  <- sum(presence_counts >= softcore_threshold)
  }
}

cat("\n真实抽样计算完成。\n\n")

pan_mean  <- rowMeans(pan_matrix)
pan_sd    <- apply(pan_matrix, 1, sd)

core_mean <- rowMeans(core_matrix)
core_sd   <- apply(core_matrix, 1, sd)

plot_data <- data.frame(
  Genomes   = seq_len(n_genomes),
  Pan_mean  = pan_mean,
  Pan_sd    = pan_sd,
  Core_mean = core_mean,
  Core_sd   = core_sd
)

# 保存统计结果
fwrite(
  plot_data,
  file.path(out_dir, "Pan_Core_Curve_95SoftCore_Summary.csv")
)

cat("统计结果已保存：Pan_Core_Curve_95SoftCore_Summary.csv\n\n")

df_pan <- data.frame(
  Genomes = plot_data$Genomes,
  Genes   = plot_data$Pan_mean,
  SD      = plot_data$Pan_sd,
  Type    = "Pan-genome"
)

df_core <- data.frame(
  Genomes = plot_data$Genomes,
  Genes   = plot_data$Core_mean,
  SD      = plot_data$Core_sd,
  Type    = "Core-genome"
)

final_plot_df <- rbind(df_pan, df_core)

ymax_plot <- max(final_plot_df$Genes + final_plot_df$SD, na.rm = TRUE)

p_real <- ggplot(final_plot_df, aes(x = Genomes, y = Genes, color = Type, fill = Type)) +
  geom_ribbon(
    aes(ymin = Genes - SD, ymax = Genes + SD),
    alpha = 0.18,
    color = NA
  ) +
  geom_line(linewidth = 1.2) +
  
  scale_color_manual(values = c("Pan-genome" = "#b185db",
                                "Core-genome" = "#4cc9f0")) +
  scale_fill_manual(values = c("Pan-genome" = "#b185db",
                               "Core-genome" = "#4cc9f0")) +

  scale_x_continuous(
    expand = c(0, 0),
    limits = c(0, n_genomes)
  ) +
  scale_y_continuous(
    expand = c(0, 0),
    limits = c(0, ymax_plot * 1.02)
  ) +
  
  theme_classic() +
  theme(
    legend.position = c(0.80, 0.50),
    legend.title = element_blank(),
    legend.text = element_text(size = 16),
    axis.text = element_text(color = "black", size = 16),
    axis.title = element_text(size = 15, face = "bold"),
    axis.line = element_line(linewidth = 0.8)
  ) +
  labs(
    x = "Number of genomes",
    y = "Number of genes"
  )

print(p_real)

ggsave(
  filename = file.path(out_dir, "Figure_1b_Real_95SoftCore.pdf"),
  plot = p_real,
  width = 6,
  height = 5,
  dpi = 800,
  bg = "white"
)

ggsave(
  filename = file.path(out_dir, "Figure_1b_Real_95SoftCore.png"),
  plot = p_real,
  width = 6,
  height = 5,
  dpi = 800,
  bg = "white"
)

cat("图片已保存：\n")
cat(file.path(out_dir, "Figure_1b_Real_95SoftCore.pdf"), "\n")
cat(file.path(out_dir, "Figure_1b_Real_95SoftCore.png"), "\n\n")

cat("开始进行 Heap's law 拟合，计算 gamma...\n")

min_n_for_fit <- 10

fit_df <- data.frame(
  Genomes = plot_data$Genomes,
  Pan_mean = plot_data$Pan_mean
)

fit_df <- fit_df[
  fit_df$Genomes >= min_n_for_fit &
    fit_df$Pan_mean > 0,
]

heaps_fit <- lm(
  log(Pan_mean) ~ log(Genomes),
  data = fit_df
)

gamma <- coef(heaps_fit)[["log(Genomes)"]]
log_kappa <- coef(heaps_fit)[["(Intercept)"]]
kappa <- exp(log_kappa)
r_squared <- summary(heaps_fit)$r.squared

cat("\nHeap's law fitting result:\n")
cat("Formula: P(n) = kappa * n^gamma\n")
cat("kappa =", round(kappa, 6), "\n")
cat("gamma =", round(gamma, 6), "\n")
cat("R_squared =", round(r_squared, 6), "\n\n")

# 保存 gamma 结果
heaps_result <- data.frame(
  Parameter = c("kappa", "gamma", "R_squared", "min_n_for_fit", "n_genomes", "n_gene_clusters"),
  Value = c(kappa, gamma, r_squared, min_n_for_fit, n_genomes, n_genes)
)

fwrite(
  heaps_result,
  file.path(out_dir, "Heaps_Law_Gamma_Result.csv")
)

cat("Heap's law gamma 结果已保存：Heaps_Law_Gamma_Result.csv\n\n")

# 生成拟合曲线
plot_data$Pan_fitted_heaps <- kappa * (plot_data$Genomes ^ gamma)

p_heaps <- ggplot(plot_data, aes(x = Genomes)) +
  geom_point(
    aes(y = Pan_mean),
    size = 1.2,
    alpha = 0.45,
    color = "#b185db"
  ) +
  geom_line(
    aes(y = Pan_fitted_heaps),
    linewidth = 1.2,
    color = "black"
  ) +
  theme_classic() +
  theme(
    legend.position = "none",
    axis.text = element_text(color = "black", size = 14),
    axis.title = element_text(size = 15, face = "bold"),
    axis.line = element_line(linewidth = 0.8),
    plot.title = element_text(size = 14, face = "bold")
  ) +
  labs(
    x = "Number of genomes",
    y = "Pan-genome size",
    title = paste0("Heap's law fit: gamma = ", round(gamma, 3),
                   ", R² = ", round(r_squared, 3))
  )

ggsave(
  filename = file.path(out_dir, "Heaps_Law_Fit.pdf"),
  plot = p_heaps,
  width = 6,
  height = 5,
  dpi = 800,
  bg = "white"
)

ggsave(
  filename = file.path(out_dir, "Heaps_Law_Fit.png"),
  plot = p_heaps,
  width = 6,
  height = 5,
  dpi = 800,
  bg = "white"
)

cat("Heap's law 拟合图已保存：\n")
cat(file.path(out_dir, "Heaps_Law_Fit.pdf"), "\n")
cat(file.path(out_dir, "Heaps_Law_Fit.png"), "\n\n")

cat("第 1 个 genome 时：\n")
cat("Pan =", plot_data$Pan_mean[1], "\n")
cat("Core =", plot_data$Core_mean[1], "\n\n")

cat("最后一个 genome 时（平均）：\n")
cat("Pan =", round(tail(plot_data$Pan_mean, 1), 2), "\n")
cat("Core =", round(tail(plot_data$Core_mean, 1), 2), "\n\n")

cat("全部完成。\n")