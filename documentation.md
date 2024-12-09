# **README: Spatio-Temporal Beam-Level Traffic Forecasting Solution for ITU**

### **By [Abdourahamane Ide Salifou]**  
*First-Year AI Engineering Student, Carnegie Mellon University Africa*

---
## **Table of Contents**

1. [Introduction](#introduction)
2. [Challenge Overview](#challenge-overview)
3. [Dataset Description](#dataset-description)
4. [Exploratory Data Analysis and Key Insights](#exploratory-data-analysis-and-key-insights)
5. [Feature Engineering](#feature-engineering)
6. [Model Architecture](#model-architecture)
7. [Blending Approach](#blending-approach)
8. [Training and Evaluation](#training-and-evaluation)
9. [Conclusion](#conclusion)

---

## **1. Introduction**

This documentation presents an innovative and carefully designed solution for the **Spatio-Temporal Beam-Level Traffic Forecasting Challenge**, organized by the **International Telecommunication Union (ITU)** as part of the **AI for Good** initiative. This competition required participants to develop state-of-the-art models capable of accurately predicting **traffic throughput volume (DLThpVol)** at the beam level in telecommunication networks. This task is critical for optimizing network resource allocation, minimizing congestion, and ultimately improving user experience and energy efficiency.

### **Objective**
Our solution aims to build a **multivariate time-series forecasting model** that leverages cutting-edge machine learning techniques, domain knowledge, and a thorough understanding of network dynamics to forecast traffic volumes with high precision. The solution is designed to offer **scalability and robustness** for real-world traffic scenarios encountered in modern communication networks.Our solution combines **deep domain knowledge in telecommunications** with **state-of-the-art machine learning** techniques to achieve precise traffic forecasts.

---

## **2. Challenge Overview**

In this challenge, we are tasked with predicting **beam-level traffic throughput volumes** for two target weeks (5w-6w and 10w-11w) using five weeks of historical data. The high-resolution dataset provided includes hourly throughput volumes for multiple beams across base stations, accompanied by **Physical Resource Block (PRB) utilization** and **user count**. The data’s multivariate, spatio-temporal nature requires sophisticated models capable of capturing both temporal dependencies and spatial relationships between different beams.

### **Relevance and Impact**
Accurate forecasting of traffic volume helps telecommunication operators dynamically allocate resources, reduce energy consumption, manage congestion, and optimize user experience. This project aligns with ITU’s goals of **AI for Good**, supporting **energy conservation**, **sustainable development**, and **improving digital infrastructure**.

---

## **3. Dataset Description**

The dataset contains hourly **beam-level traffic data** for five weeks across **30 base stations** with **3 cells** and **32 beams per base station**. Each row represents the traffic volume for a given hour, beam, and cell, along with other features like **PRB utilization** and **user count**.

### **Key Features:**
- **DLThpVol**: The target variable, representing traffic throughput volume.
- **PRB Utilization**: Measures network resource usage.
- **User Count**: The number of active users connected to each beam.
- **Time Features**: Hourly granularity data, with 24 hours a day, and weekly periodicity.

The data is multivariate, requiring the capture of both **temporal patterns** (such as daily/weekly cycles) and **spatial relationships** (interactions across beams within cells and base stations).

---

## **4. Exploratory Data Analysis and Key Insights**

In the initial phase, we conducted **exploratory data analysis (EDA)** to understand the underlying patterns and statistical properties of the data. This provided key insights that informed our feature engineering decisions and model architecture.

### **4.1 Time Series Decomposition**

We decomposed the time series data into **trend**, **seasonality**, and **residual** components for all beams. Aggregating across all beams revealed strong **cyclical patterns**, consistent with telecommunications traffic behavior.

**Key Statistics:**
- **Mean Trend**: 0.33
- **Trend Std**: 0.02
- **Seasonality Peak-to-Trough Ratio**: -0.86
- **Residual Autocorrelation @ 24 hours**: 0.15

#### **Interpretation**:
- The **trend** indicates steady growth in traffic volume, which is expected in a growing network environment.
- The **seasonality** confirms the presence of strong daily and weekly patterns, which supports the use of cyclical time features.
- The **residuals** are relatively low, suggesting that most variability can be explained by the model. The autocorrelation value at 24 hours highlights periodic dependencies.

![Time serie decomposition plot](https://drive.google.com/uc?export=view&id=1FV6y7LwVRfsfuyh8eQ5JtvQ9ITyln03B)
![residuals plot](https://drive.google.com/uc?export=view&id=1ZNBzdnJSV_ZrcOnEzmMlqmFJuDkS0tar)

---

### **4.2 PRB Utilization and Congestion**

PRB utilization reflects how much of the network's resources are being used. We calculated the average PRB utilization across all beams and identified periods of congestion when utilization exceeded a threshold of 0.8.

**Key Statistics:**
- **Mean PRB Utilization**: 0.54
- **PRB Utilization Std**: 0.21
- **Congestion Ratio**: 4.76%
- **Average Congestion Duration**: 0.82 hours

#### **Interpretation**:
- A congestion ratio of 4.76% indicates that the network operates close to capacity during peak hours. This justifies the use of congestion-based features in the model.

**[Plot: PRB Utilization and Congestion Detection]**

---

### **4.3 Rolling Averages (Smoothing)**

We calculated **3-hour** and **12-hour rolling averages** for DLThpVol to smooth out short-term fluctuations and capture long-term trends.

**Key Statistics:**
- **3-hour Rolling Average**: Mean = 0.32, Std = 0.11, Range = 0.47
- **12-hour Rolling Average**: Mean = 0.32, Std = 0.07, Range = 0.32

#### **Interpretation**:
- These rolling averages help smooth out spikes and drops in traffic, providing the model with clearer trends. They are useful for reducing noise while preserving important patterns.

**[Plot: 3-hour and 12-hour Rolling Averages]**

---

## **5. Feature Engineering**

The success of our solution is largely attributed to **feature engineering**, which is grounded in telecommunications domain knowledge and data-driven insights from EDA. Here, we explain the creative rationale behind each feature.

### **Cyclical Time-Based Features**

- **Rationale (Telecommunications Domain Knowledge)**: Traffic follows strong daily and weekly cycles (e.g., traffic peaks during work hours and drops overnight). To capture these patterns, we encode time in cyclical features.
- **ML Principle**: By using sine and cosine transformations, the model can capture time's periodic nature, ensuring the continuity between the last and first hours of the day.

**Features Created**:
- **`hour_sin`, `hour_cos`**: Capture the cyclical nature of 24-hour traffic patterns.
- **`is_weekend`**: Encodes differences in traffic behavior on weekends vs. weekdays.
- **`is_peak_hour`**: Identifies peak hours (e.g., commute times) where traffic volume is expected to spike.

### **Lag Features**

- **Rationale (Telecommunications Domain Knowledge)**: Traffic volume at any hour is influenced by past values, especially at immediate (1-hour), daily (24-hour), and weekly (168-hour) intervals.
- **ML Principle**: Lag features provide the model with temporal context, enabling it to learn from past traffic volumes and make more informed predictions.

**Features Created**:
- **1-hour lag**: Captures immediate dependencies in traffic.
- **24-hour lag**: Captures daily repetitions in the data.
- **168-hour lag**: Captures weekly traffic patterns.

### **Congestion Interaction Features**

- **Rationale (Telecommunications Domain Knowledge)**: High PRB utilization often leads to congestion, which in turn affects throughput volume. This interaction is dynamic and time-dependent.
- **ML Principle**: By multiplying the congestion flag (derived from PRB utilization) with time-based features (`hour_sin`, `hour_cos`), we capture how network congestion impacts throughput at specific hours. This feature helps the model account for dynamic congestion patterns and their effects on traffic volume.


**Feature Created**:
- **`congestion_interaction`**: Combines PRB utilization with time-based features to model how congestion at specific times of the day impacts traffic.

### **Rolling Averages**

- **Rationale (Telecommunications Domain Knowledge)**: Network traffic fluctuates over short and long periods, but averaging helps smooth these fluctuations to reveal underlying trends.
- **ML Principle**: Rolling averages reduce noise in the data, allow the model to capture smoother traffic trends over different window sizeswindows (3-hour and 12-hour), reducing the effect of outliers. This feature helps the model identify underlying traffic trends and variability, enhancing its ability to generalize.

**Features Created**:
- **3-hour rolling average**: Captures short-term smoothing of traffic fluctuations.
- **12-hour rolling average**: Captures longer-term smoothing of traffic patterns.



---

## **6. Model Architecture**

Our solution employs a **hybrid model** that blends a **Neural Network (NN)** and **XGBoost (XGB)**. Each model was tailored to address the specific challenges posed by the data.

### **Neural Network (Conv1D + GRU + Multi-Head Attention)**

The **neural network architecture** was designed to capture both **short-term temporal dependencies** and **long-range interactions** in the data.

- **Conv1D Layers**: These layers apply dilated convolutions to capture localized temporal patterns efficiently. Dilations allow the model to “look back” at longer sequences without increasing computational cost.
  
- **GRU (Gated Recurrent Unit)**: GRUs are effective for time-series data as they retain relevant information from previous time steps, making them ideal for capturing long-range dependencies in traffic patterns.

- **Multi-Head Attention**: This mechanism allows the model to focus on specific time points that are most relevant for prediction, enhancing the model's ability to capture relationships between distant time steps.It dynamically learns which parts of the sequence are most relevant for predicting future traffic volumes.

- **Residual Connections**: Residual connections improve the training process by enabling better gradient flow and ensuring deeper layers still retain meaningful information from earlier layers.These connections help mitigate vanishing gradient problems and enable deeper architectures to be trained effectively.
- **Final Dense Layer**: The final output layer predicts the traffic volume for each beam.


**Advantages**:
- **Temporal Dependencies**: The GRU and Conv1D layers ensure the model captures both short-term and long-term patterns.
- **Dynamic Focus**: Multi-Head Attention enables the model to focus dynamically on critical time steps.

### **XGBoost Model**

XGBoost complements the neural network by capturing **non-linear interactions** between the features that the neural network might miss.

- **Strength**: XGBoost excels at handling structured data and non-linear relationships, making it an ideal complement to the deep learning model’s strengths in capturing sequential patterns.



---

## **7. Blending Approach**

Both models are blended using **weighted averaging**, with optimal weights determined through experimentation:
- **NN Weight**: 0.5
- **XGBoost Weight**: 0.5

The neural network excels at capturing temporal and spatial patterns, while XGBoost adds non-linear predictive power, leading to robust predictions.

---

## **8. Training and Evaluation**

### **Neural Network Training**:
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Epochs** : 100
- **Batch size**: 32
- **Early Stopping**: Stops training when validation loss plateaus.
- **Learning Rate Scheduler**: Reduces learning rate when improvement stalls.

### **XGBoost

 Training**:
- **Learning Rate**: 0.05
- **Max Depth**: 3
- **Number of Estimators**: 50

### **Final Performance:**
- **Validation MAE**: **0.1689**
- **Runtime**: **1 hour 10 minutes on Kaggle GPU T4x2**

---

## **9. Conclusion**

This solution presents a **robust, innovative, and scalable** approach for **spatio-temporal beam-level traffic forecasting**. By leveraging creative feature engineering techniques and a hybrid model architecture, we were able to effectively capture the complexities of telecommunications traffic.

### **Why this solution excels**:
- **Domain-specific feature engineering** captures critical traffic behaviors, such as cyclicality, congestion, and lag effects.
- **Blended modeling approach** balances the strengths of neural networks for sequence data and XGBoost for non-linear relationships.
- **Scalability and adaptability** make this solution suitable for real-world deployment, offering ITU a practical tool for optimizing network flow and resource allocation.

---


The solution outlined in this documentation aligns closely with the **AI for Good** initiative, providing actionable insights to support **sustainable network management** and improve **global telecommunications infrastructure**.

--- 




