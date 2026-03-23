# Customer Segmentation Analysis: Comprehensive Report

## Executive Summary
This report details an extensive exploration of customer segmentation using clustering techniques on the Mall Customers dataset. The project aimed to identify optimal clustering methods and feature sets for segmenting customers based on demographic and behavioral data. Through rigorous experimentation with K-Means, DBSCAN, Hierarchical Clustering, and Gaussian Mixture Models (GMM), combined with feature engineering (introducing "Spending Efficiency"), this project achieved nuanced insights into customer groups. Key findings include the superiority of Spending Efficiency features and the balanced performance of K-Means and DBSCAN.

**Date of Analysis**: March 23, 2026  
**Dataset**: Mall_Customers.csv (200 samples, 5 features)  
**Tools Used**: Python, Jupyter Notebooks, Scikit-Learn, Matplotlib, Seaborn, Pandas, NumPy  

---

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Engineering](#feature-engineering)
4. [Clustering Methodology](#clustering-methodology)
5. [Experimental Results](#experimental-results)
   - [K-Means Clustering](#k-means-clustering)
   - [DBSCAN Clustering](#dbscan-clustering)
   - [Hierarchical Clustering](#hierarchical-clustering)
   - [Gaussian Mixture Models](#gaussian-mixture-models)
6. [Comparative Analysis](#comparative-analysis)
7. [Visualizations](#visualizations)
8. [Conclusions and Recommendations](#conclusions-and-recommendations)
9. [Appendices](#appendices)

---

## Dataset Overview
The Mall Customers dataset contains information about 200 mall customers, sourced from Kaggle. It includes the following features:

- **CustomerID**: Unique identifier (integer, 1-200)
- **Gender**: Categorical (Male/Female)
- **Age**: Numerical (integer, 18-70)
- **Annual Income (k$)**: Numerical (integer, 15-137)
- **Spending Score (1-100)**: Numerical (integer, 1-99)

**Key Statistics**:
- Total Samples: 200
- No missing values
- No duplicate entries
- Balanced gender distribution (~56% Female, 44% Male)

The dataset focuses on behavioral segmentation, with Age, Income, and Spending Score as primary clustering variables.

---

## Data Preprocessing
### Initial Checks
- **Null Values**: None detected across all features.
- **Duplicates**: No duplicate rows found.
- **Outliers**: Visual inspection via box plots revealed no extreme outliers requiring removal.

### Standardization
All numerical features were standardized using `StandardScaler` to ensure equal weighting in distance-based algorithms (K-Means, DBSCAN, Hierarchical).

### Feature Selection
- Excluded CustomerID and Gender for clustering (ID is irrelevant; Gender was not explored in this analysis).
- Focused on Age, Annual Income, and Spending Score.

---

## Feature Engineering
To enhance clustering performance, we introduced **Spending Efficiency** as a derived feature:

**Formula**: `Spending_Efficiency = Spending_Score / Annual_Income`

- **Rationale**: This metric captures how efficiently a customer spends relative to their income, potentially revealing behavioral patterns not evident in raw scores.
- **Impact**: Replaced Annual Income and Spending Score with Age and Spending Efficiency, reducing dimensionality while preserving interpretability.
- **Validation**: Spending Efficiency ranges from ~0.007 to ~0.72, with a mean of ~0.37.

This engineering step was pivotal, as efficiency-based features consistently outperformed original features in silhouette scores.

---

## Clustering Methodology
We evaluated four clustering algorithms, tuning hyperparameters using Silhouette Score (ranges from -1 to 1; higher is better) as the primary metric. Additional metrics included number of clusters and noise ratio (for density-based methods).

### General Evaluation Function
A custom `evaluate_model` function was used:
- Calculates Silhouette Score (excluding noise for DBSCAN).
- Computes noise ratio (percentage of unclustered points).
- Ensures at least 2 clusters for valid scoring.

### Hyperparameter Tuning
- **K-Means/DBSCAN/Hierarchical**: Tested k/eps/components from 2-10.
- **DBSCAN**: Eps tuned from 0.1-1.0 in 0.001 increments, with min_samples=5; filtered for 2-7 clusters, <40% noise, Silhouette >0.3.

---

## Experimental Results

### K-Means Clustering
**Algorithm Overview**: Partitions data into k spherical clusters by minimizing within-cluster variance.

**Tuning Results** (Original Features):
- Best k=6 (Silhouette: 0.426)
- Scores: k=2: 0.296, k=3: 0.357, ..., k=9: 0.401

**Tuning Results** (Efficiency Features):
- Best k=3 (Silhouette: 0.453)
- Scores: k=2: 0.296, k=3: 0.453, ..., k=9: 0.401

**Cluster Profiles** (Efficiency, k=3):
- Cluster 0: Age 28.2, Income 87.8, Spending 79.3 (High spenders)
- Cluster 1: Age 56.2, Income 54.3, Spending 49.1 (Moderate)
- Cluster 2: Age 32.7, Income 86.5, Spending 81.5 (High spenders)

### DBSCAN Clustering
**Algorithm Overview**: Density-based clustering that identifies core points and expands clusters; handles noise and arbitrary shapes.

**Tuning Results** (Efficiency Features):
- Best eps=0.778, min_samples=5
- Clusters: 4, Silhouette: 0.406, Noise: 0.0%
- Other candidates: eps=0.779 (Clusters: 4, Sil: 0.406), etc.

**Cluster Profiles**:
- Effective at separating dense regions; no noise in optimal config.

### Hierarchical Clustering
**Algorithm Overview**: Agglomerative method building a hierarchy via linkage (ward linkage used).

**Tuning Results** (Efficiency Features):
- Best k=2 (Silhouette: 0.296)
- Scores: k=2: 0.296, k=3: 0.282, ..., k=9: 0.241

**Cluster Profiles** (k=2):
- Simple binary split; lower performance compared to others.

### Gaussian Mixture Models
**Algorithm Overview**: Probabilistic model assuming Gaussian distributions; soft clustering.

**Tuning Results** (Efficiency Features):
- Best components=2 (Silhouette: 0.296)
- Scores: Similar to Hierarchical, peaking at 2 components.

**Cluster Profiles**:
- Overlapped with Hierarchical results; not superior.

---

## Comparative Analysis
| Method       | Features          | Clusters | Silhouette Score | Noise Ratio | Best Params          |
|--------------|-------------------|----------|------------------|-------------|----------------------|
| K-Means     | Original         | 6        | 0.426           | N/A        | k=6                 |
| K-Means     | Efficiency       | 3        | 0.453           | N/A        | k=3                 |
| DBSCAN      | Efficiency       | 4        | 0.406           | 0.0%       | eps=0.778, min=5    |
| Hierarchical| Efficiency       | 2        | 0.296           | N/A        | k=2                 |
| GMM         | Efficiency       | 2        | 0.296           | N/A        | components=2        |

**Key Insights**:
- **Efficiency Features Superior**: Consistently higher Silhouette (e.g., K-Means 0.453 vs. 0.426).
- **K-Means Best Overall**: Balanced clusters, high interpretability.
- **DBSCAN Strong Alternative**: Handles density well, zero noise.
- **Hierarchical/GMM Underperform**: Struggle with this dataset's structure.

---

## Visualizations
All plots use PCA for 2D projection to visualize clusters.

- **Box Plots**: Distribution of Age, Income, Spending Score.
- **Pair Plots**: Feature relationships.
- **Tuning Plots**: Silhouette scores vs. hyperparameters.
- **Cluster Scatter Plots**: PCA projections colored by clusters.

*(See extracted_images/ for all PNG files)*

---

## Conclusions and Recommendations
### Key Takeaways
1. **Feature Engineering Wins**: Spending Efficiency unlocks better segmentation.
2. **K-Means Recommended**: For simplicity and performance.
3. **DBSCAN for Density**: If noise or shapes are concerns.
4. **Avoid Hierarchical/GMM**: Lower scores; not ideal here.

### Business Implications
- **Target High Spenders**: Clusters 0 & 2 (young, high income/spending).
- **Retention Focus**: Moderate clusters for loyalty programs.
- **Marketing Strategies**: Personalized based on efficiency profiles.

### Future Work
- Explore Gender integration.
- Try other features (e.g., income categories).
- Validate with external metrics (e.g., sales data).
- Scale to larger datasets.

---

## Appendices
- **Code Snippets**: Available in notebooks.
- **Full Silhouette Logs**: See notebook outputs.
- **Image Files**: 16 extracted PNGs in extracted_images/.

*Report Generated Automatically from Notebook Analysis.*
