# Input data

Raw genome data and metadata are not included in this repository.

The analysis expects:

| File | Description |
|---|---|
| `gene_presence_absence_cleaned.csv` | Pangenome gene presence–absence matrix (6,433 isolates) |
| `file_with_host_date_ip.csv` | Sample metadata with host source, date and isolation information |

Source:

- National Microbiology Data Center: https://nmdc.cn/clsgdbv2

Place the two tables in this directory (or in the working directory) and edit the path constants at the top of the scripts if needed.

The fitted model artifacts used by the HSS-SHAPi scripts are also not included because of size. They are produced by the final training pipeline and are expected in the paths set in `hss/bootstrap.py` and `hss/permutation1000.py`.
