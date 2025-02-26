# **Alphabet Soup Charity Deep Learning Model Report**

## **Overview of the Analysis**
The purpose of this analysis was to develop a deep learning model to predict whether an **Alphabet Soup-funded organization** will be successful based on its features. A binary classification neural network was built, trained, and optimized to improve its predictive accuracy.

The dataset contained categorical and numerical variables related to **funding applications**, and preprocessing steps were performed to clean and transform the data before feeding it into the neural network.

---

## **Results**

### **Data Preprocessing**
- **Target Variable:**  
  - `IS_SUCCESSFUL` (Binary: 1 for successful, 0 for unsuccessful)
- **Feature Variables:**  
  - All remaining columns after removing identifiers and non-predictive attributes.
- **Removed Variables:**  
  - `EIN` and `NAME` (as they are unique identifiers and do not contribute to prediction).
- **Handling Categorical Data:**
  - `APPLICATION_TYPE` and `CLASSIFICATION` contained many unique categories.
  - Rare categories were grouped under the `"Other"` label to reduce dimensionality.
  - Categorical variables were converted into numerical features using **one-hot encoding** (`pd.get_dummies()`).
- **Splitting Data:**
  - The dataset was split into **training (75%)** and **testing (25%)** sets using `train_test_split()`.
- **Feature Scaling:**
  - **StandardScaler** was applied to normalize numerical features and ensure consistent feature distributions.

---

### **Compiling, Training, and Evaluating the Model**

#### **Initial Model Configuration (AlphabetSoupCharity.ipynb)**
- **Model Architecture:**
  - **Input Layer:** Number of features in the dataset.
  - **Hidden Layer 1:** 9 neurons, **ReLU** activation.
  - **Hidden Layer 2:** 18 neurons, **ReLU** activation.
  - **Output Layer:** 1 neuron, **Sigmoid** activation (for binary classification).
- **Compilation:**
  - **Loss Function:** Binary Crossentropy
  - **Optimizer:** Adam
  - **Metric:** Accuracy
- **Training:**
  - Model was trained for **100 epochs**.
- **Performance Results:**
  - **Loss:** **0.5528**
  - **Accuracy:** **72.97%**
  - The accuracy did **not** meet the **75% target**, so further optimization was required.

---

### **Optimization Efforts (AlphabetSoupCharity_Optimization.ipynb)**
To improve the model's accuracy, the following optimizations were attempted:

1. **Adjusting Hidden Layers and Neurons:**
   - Added an **extra hidden layer** to the network.
   - Adjusted neurons in each layer to improve learning capacity.
   
2. **Modified Training Strategy:**
   - Reduced epochs to **20** (to prevent overfitting).
   - Introduced **validation split (15%)** during training.

3. **Activation Functions:**
   - Used **ReLU** activation for hidden layers.
   - **Sigmoid** remained in the output layer.

4. **Results After Optimization:**
   - **Loss:** **0.5560**
   - **Accuracy:** **72.70%**
   - Despite optimizations, the model's accuracy slightly decreased.

---

## **Summary**
- **Final Accuracy Achieved:**  
  - **Initial Model:** **72.97%**
  - **Optimized Model:** **72.70%**
- **Did the model achieve 75% accuracy?**  
  - ❌ **No**, despite optimization efforts, the model's accuracy did not reach 75%.

---

### **Final Conclusion**
The deep learning model developed for Alphabet Soup was able to classify **successful funding applications** with approximately **73% accuracy**. However, even after optimization, the model failed to reach the **75% target**, indicating that additional tuning, feature engineering, or alternative machine learning models may be needed for better performance.

---
