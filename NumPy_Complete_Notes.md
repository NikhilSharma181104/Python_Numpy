# Complete NumPy Guide

## Table of Contents
1. [Introduction to NumPy](#introduction-to-numpy)
2. [Setting up Jupyter Notebooks](#setting-up-jupyter-notebooks)
3. [NumPy Arrays Basics](#numpy-arrays-basics)
4. [Array Creation Methods](#array-creation-methods)
5. [Array Attributes and Methods](#array-attributes-and-methods)
6. [Array Indexing and Slicing](#array-indexing-and-slicing)
7. [Array Operations](#array-operations)
8. [Broadcasting and Copy Concepts](#broadcasting-and-copy-concepts)
9. [Matrix Operations](#matrix-operations)
10. [Advanced Array Manipulation](#advanced-array-manipulation)
11. [Essential Data Analytics & Data Cleaning Tools](#essential-data-analytics--data-cleaning-tools)
12. [Practical Exercises](#practical-exercises)

---

## Introduction to NumPy

### What is NumPy?
NumPy (Numerical Python) is a powerful library for Python that provides support for arrays and mathematical operations. It was created by combining two earlier libraries:
- **Numeric Library** (1995) - Created by Jim Hugunin
- **Numarray** (early 2000s) - Developed by Space Telescope Science Institute
- **NumPy** (2005) - Created by Travis Oliphant by combining both libraries

### Why NumPy?
- **Speed**: Written in C, making it up to 50 times faster than Python lists
- **Memory Efficient**: Uses less memory compared to Python lists
- **Vectorized Operations**: Supports element-wise operations without explicit loops
- **Broadcasting**: Allows operations between arrays of different shapes

### Key Differences: Lists vs Arrays

| Feature | Python Lists | NumPy Arrays |
|---------|-------------|--------------|
| Data Types | Heterogeneous (mixed types) | Homogeneous (same type) |
| Performance | Slow (interpreted) | Fast (compiled C code) |
| Memory | Less efficient | More efficient |
| Operations | Manual loops required | Vectorized operations |

---

## Setting up Jupyter Notebooks

### Installation Options

#### 1. Anaconda Distribution (Recommended)
```bash
# Download from anaconda.com
# Install Anaconda Navigator
# Launch Jupyter Notebook from Navigator
```

#### 2. VS Code with Jupyter Extension
```bash
# Install Jupyter extension in VS Code
# Create .ipynb files
# Select Python kernel
```

#### 3. Google Colab
- Cloud-based solution
- Pre-installed packages
- GPU support available
- Automatic sync with Google Drive

### Basic Jupyter Operations
- **Create new cell**: Press `B` (below) or `A` (above)
- **Run cell**: `Shift + Enter`
- **Cell types**: Code, Markdown
- **Auto-completion**: `Shift + Tab` for function help

---

## NumPy Arrays Basics

### Importing NumPy
```python
import numpy as np
```

### Creating Arrays from Lists
```python
# 1D Array (Vector)
arr = np.array([1, 2, 3, 4, 5])
print(arr)  # [1 2 3 4 5]

# 2D Array (Matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
```

### Data Type Conversion
```python
# Mixed types get converted to same type
mixed = np.array([1, 2.5, "hello"])
print(mixed)  # ['1' '2.5' 'hello'] - all strings

# Numbers get converted to floats if one float present
numbers = np.array([1, 2, 3.5])
print(numbers)  # [1.  2.  3.5] - all floats
```

### Dimensions
- **1D**: Vector (single row)
- **2D**: Matrix (rows and columns)
- **3D+**: Tensors (used in deep learning)

---

## Array Creation Methods

### 1. Using `np.arange()`
```python
# Similar to Python's range()
arr = np.arange(1, 11)  # [1 2 3 4 5 6 7 8 9 10]

# With step
arr = np.arange(1, 11, 2)  # [1 3 5 7 9]
```

### 2. Creating Arrays of Zeros and Ones
```python
# 1D arrays
zeros = np.zeros(6)  # [0. 0. 0. 0. 0. 0.]
ones = np.ones(5)    # [1. 1. 1. 1. 1.]

# 2D arrays
zeros_2d = np.zeros((3, 4))  # 3 rows, 4 columns of zeros
ones_2d = np.ones((2, 3))    # 2 rows, 3 columns of ones
```

### 3. Linear Spacing with `np.linspace()`
```python
# Creates evenly spaced numbers over specified interval
arr = np.linspace(1, 5, 5)  # [1. 2. 3. 4. 5.]
arr = np.linspace(0, 1, 100)  # 100 evenly spaced numbers from 0 to 1
```

### 4. Random Number Generation
```python
# Random numbers between 0 and 1
random_arr = np.random.rand(5)  # 5 random numbers

# Random integers
random_int = np.random.randint(10, 20, size=10)  # 10 random integers between 10-20

# Normal distribution (mean=0, std=1)
normal_arr = np.random.randn(5)  # Standard normal distribution
```

---

## Array Attributes and Methods

### Important Attributes
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Shape: dimensions of array
print(arr.shape)  # (2, 3) - 2 rows, 3 columns

# Size: total number of elements
print(arr.size)   # 6

# Data type
print(arr.dtype)  # int64 (or int32 depending on system)
```

### Useful Methods
```python
arr = np.array([1, 2, 3, 4, 5])

# Statistical methods
print(arr.min())    # 1
print(arr.max())    # 5
print(arr.sum())    # 15
print(arr.mean())   # 3.0
print(arr.std())    # Standard deviation

# Index of min/max values
print(arr.argmin()) # 0 (index of minimum value)
print(arr.argmax()) # 4 (index of maximum value)
```

### Reshaping and Resizing
```python
arr = np.arange(1, 13)  # [1 2 3 4 5 6 7 8 9 10 11 12]

# Reshape to 2D
reshaped = arr.reshape(3, 4)  # 3 rows, 4 columns
print(reshaped)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

# Resize (modifies original array)
arr.resize(4, 3)
```

---

## Array Indexing and Slicing

### 1D Array Indexing
```python
arr = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

# Single element
print(arr[0])   # 11
print(arr[-1])  # 20

# Slicing
print(arr[1:5])    # [12 13 14 15]
print(arr[:5])     # [11 12 13 14 15]
print(arr[5:])     # [16 17 18 19 20]
print(arr[::2])    # [11 13 15 17 19] - every 2nd element
```

### 2D Array Indexing
```python
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Single element: [row, column]
print(matrix[0, 0])  # 1
print(matrix[1, 2])  # 6

# Entire row
print(matrix[0])     # [1 2 3]
print(matrix[0, :])  # [1 2 3] - explicit all columns

# Entire column
print(matrix[:, 0])  # [1 4 7]

# Slicing submatrix
print(matrix[0:2, 1:3])  # [[2 3]
                         #  [5 6]]
```

### Boolean Indexing
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Create boolean mask
mask = arr % 2 == 0  # [False True False True False True False True False True]

# Apply mask to get even numbers
even_numbers = arr[mask]  # [2 4 6 8 10]

# Direct boolean indexing
even_direct = arr[arr % 2 == 0]  # [2 4 6 8 10]
```

---

## Array Operations

### Arithmetic Operations
```python
a1 = np.array([1, 2, 3, 4, 5])
a2 = np.array([6, 7, 8, 9, 10])

# Element-wise operations
print(a1 + a2)   # [ 7  9 11 13 15]
print(a1 - a2)   # [-5 -5 -5 -5 -5]
print(a1 * a2)   # [ 6 14 24 36 50]
print(a1 / a2)   # [0.16666667 0.28571429 0.375 0.44444444 0.5]
print(a1 ** a2)  # [1 128 6561 262144 9765625]
```

### Aggregate Operations with Axis
```python
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Sum all elements
print(np.sum(matrix))  # 45

# Sum along axis
print(np.sum(matrix, axis=0))  # [12 15 18] - column-wise sum
print(np.sum(matrix, axis=1))  # [ 6 15 24] - row-wise sum

# Other operations
print(np.mean(matrix, axis=0))  # Column-wise mean
print(np.std(matrix, axis=1))   # Row-wise standard deviation
```

---

## Broadcasting and Copy Concepts

### Broadcasting
Broadcasting allows operations between arrays of different shapes:

```python
arr = np.array([10, 20, 30, 40])

# Scalar broadcasting
result = arr + 10  # [20 30 40 50]
result = arr * 2   # [20 40 60 80]

# 2D broadcasting
matrix = np.arange(1, 13).reshape(3, 4)
result = matrix + 10  # Adds 10 to each element
```

### Deep vs Shallow Copy
```python
# Original array
a = np.arange(1, 21)

# Shallow copy (shares memory)
b = a
b[0] = 99
print(a[0])  # 99 - original array is modified

# Deep copy (separate memory)
a = np.arange(1, 21)  # Reset
c = a.copy()
c[0] = 99
print(a[0])  # 1 - original array unchanged

# Slicing creates views (shares memory for arrays)
slice_arr = a[0:5]
slice_arr[0] = 999
print(a[0])  # 999 - Modifying the slice MUTATES original array 'a' because it is a VIEW!

# To make an independent slice (copy), explicitly use .copy():
safe_slice = a[0:5].copy()
safe_slice[0] = 123
print(a[0])  # 999 - 'a' remains unchanged!

# Reassignment (e.g. slice_arr = slice_arr * 10) creates a new array and breaks the view connection.
# But in-place operations (e.g. slice_arr *= 10) will modify the original array!
```

---

## Matrix Operations

### Matrix Multiplication
```python
# Element-wise multiplication
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

element_wise = a * b  # [[5 12] [21 32]]

# Matrix multiplication (dot product)
matrix_mult = a @ b   # [[19 22] [43 50]]
# or
matrix_mult = np.dot(a, b)  # Same result
```

### Matrix Transpose
```python
matrix = np.array([[1, 2], [3, 4]])
transposed = matrix.T  # [[1 3] [2 4]]
```

---

## Advanced Array Manipulation

### Stacking Arrays
```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Vertical stack (row-wise)
v_stack = np.vstack((a, b))  # [[1 2 3 4]
                             #  [5 6 7 8]]

# Horizontal stack (column-wise)
h_stack = np.hstack((a, b))  # [1 2 3 4 5 6 7 8]

# Column stack (creates 2D array)
c_stack = np.column_stack((a, b))  # [[1 5]
                                   #  [2 6]
                                   #  [3 7]
                                   #  [4 8]]
```

### Splitting Arrays
```python
arr = np.arange(1, 17).reshape(4, 4)

# Horizontal split
h_split = np.hsplit(arr, 2)  # Split into 2 parts horizontally

# Vertical split
v_split = np.vsplit(arr, 2)  # Split into 2 parts vertically

# Access individual splits
for i, split in enumerate(h_split):
    print(f"Split {i}:")
    print(split)
```

---

## Essential Data Analytics & Data Cleaning Tools

### 1. Conditional Logic & Feature Creation: `np.where()`
`np.where(condition, [x, y])` is a vectorized ternary operator (`if condition then x else y`):
- When given `(condition, x, y)`: returns elements chosen from `x` or `y` based on condition.
- When given only `(condition)`: returns tuple of indices where condition is `True`.

```python
# Create new flag or categorical feature
scores = np.array([45, 82, 91, 58, 73, 33, 95])
status = np.where(scores >= 60, "Pass", "Fail")
# Output: ['Fail' 'Pass' 'Pass' 'Fail' 'Pass' 'Fail' 'Pass']

# Conditional adjustment without mutating original
adjusted = np.where(scores < 50, scores + 5, scores)

# Find indices of matching elements
high_scorers_idx = np.where(scores >= 85)[0]  # indices [2, 6]
```

### 2. Missing Values (`NaN`) Handling
Real-world datasets contain missing data represented as `np.nan` (floating-point NaN).
- **Detection**: `np.isnan(arr)` returns a boolean mask.
- **Filtering**: `arr[~np.isnan(arr)]` removes NaNs.
- **Safe Aggregation**: Standard functions like `np.mean()` return `nan` when NaNs are present. Use `np.nanmean()`, `np.nansum()`, `np.nanstd()`, and `np.nanmedian()` instead.
- **Imputation**: `np.nan_to_num(arr, nan=replacement_val)` replaces NaNs with a chosen value (e.g. 0 or mean).

```python
raw_data = np.array([85.0, 92.0, np.nan, 78.0, np.nan, 88.0])

# Detection
missing_count = np.sum(np.isnan(raw_data))  # 2

# NaN-safe statistics
safe_mean = np.nanmean(raw_data)  # 85.75
safe_std = np.nanstd(raw_data)

# Imputation (replacing NaNs with column average)
imputed_data = np.nan_to_num(raw_data, nan=safe_mean)
```

### 3. Frequency Analysis & Value Counts: `np.unique()`
Explore categorical distributions or find unique values:
- `return_counts=True`: Returns frequencies of each unique item (NumPy's equivalent to Pandas `value_counts()`).
- `axis=0`: Deduplicates table rows.

```python
ratings = np.array([5, 4, 3, 5, 2, 4, 5, 3, 4, 5])
values, counts = np.unique(ratings, return_counts=True)
for val, count in zip(values, counts):
    print(f"Rating {val}: {count} occurrences")

# Deduplicating 2D records
transactions = np.array([[101, 500], [102, 300], [101, 500]])
unique_records = np.unique(transactions, axis=0)
```

### 4. Sorting & Ranking Multi-dimensional Data: `np.argsort()`
In tabular data, sorting a single column breaks row correspondence! `np.argsort()` returns the **indices** that would sort the array, enabling full-table sorting.

```python
# Table: [ID, Math, Science]
students = np.array([
    [1, 78, 92],
    [2, 95, 81],
    [3, 62, 70]
])

# Sort whole table by Math marks (column 1) - Ascending:
math_idx = np.argsort(students[:, 1])
sorted_by_math = students[math_idx]

# Sort whole table by Science marks (column 2) - Descending:
sci_desc_idx = np.argsort(students[:, 2])[::-1]
sorted_by_sci = students[sci_desc_idx]

# Top student index
top_student = students[sci_desc_idx[0]]
```

### 5. Outlier Capping & Clamping: `np.clip()`
Used in data cleaning for **Winsorization** (bounding extreme values without deleting rows):

```python
# Cap values strictly within [min, max]
raw_scores = np.array([45, 88, -12, 105, 92, 115, 76, -5])
capped_scores = np.clip(raw_scores, a_min=0, a_max=100)
# [-12 becomes 0, 105 and 115 become 100]

# Percentile-based capping (Winsorization)
salaries = np.array([25000, 32000, 28000, 150000, 30000, 350000])
p5 = np.percentile(salaries, 5)
p95 = np.percentile(salaries, 95)
cleaned_salaries = np.clip(salaries, p5, p95)
```

---

## Practical Exercises

### Exercise 1: Valid Sudoku Checker
```python
import numpy as np

# Sample Sudoku grid
sudoku = np.array([
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9]
])

def validate_sudoku(s):
    # Check rows
    row_sums = np.sum(s, axis=1)
    if not np.all(row_sums == 45):
        return False
    
    # Check columns
    col_sums = np.sum(s, axis=0)
    if not np.all(col_sums == 45):
        return False
    
    # Check 3x3 blocks
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            block = s[i:i+3, j:j+3]
            if np.sum(block) != 45:
                return False
    
    return True

print("Sudoku is valid:", validate_sudoku(sudoku))
```

### Exercise 2: Student Data Analysis
```python
# Student data: [Age, Math_Score, Science_Score]
data = np.array([
    [18, 85, 92],
    [19, 92, 88],
    [17, 76, 85],
    [18, 65, 70],
    [20, 90, 95]
])

# 1. Get shape of matrix
print("Shape:", data.shape)

# 2. Find average age
avg_age = np.mean(data[:, 0])
print("Average age:", avg_age)

# 3. Extract math marks
math_marks = data[:, 1]
print("Math marks:", math_marks)

# 4. Find highest science marks
highest_science = np.max(data[:, 2])
print("Highest science marks:", highest_science)

# 5. Students who scored more than 90 in math
high_math_students = data[data[:, 1] > 90]
print("Students with math > 90:")
print(high_math_students)

# 6. Increase math marks by 5
data[:, 1] += 5
print("Updated data:")
print(data)

# 7. Count students younger than 19
young_students = data[data[:, 0] < 19]
print("Number of students younger than 19:", len(young_students))

# 8. Calculate average marks in each subject
subject_averages = np.mean(data[:, 1:], axis=0)
print("Subject averages:", subject_averages)

# 9. Students with at least 80 in both subjects
both_high = data[(data[:, 1] >= 80) & (data[:, 2] >= 80)]
print("Students with both subjects >= 80:")
print(both_high)

# 10. Replace science marks < 75 with 0
data[data[:, 2] < 75, 2] = 0
print("Final data:")
print(data)

# 11. Create a performance flag with np.where
performance = np.where(data[:, 1] >= 85, "High Performer", "Standard")
print("Performance Flag:", performance)

# 12. Handle missing exam scores (NaN) and impute with column average
data_float = data.astype(float).copy()
data_float[3, 2] = np.nan  # Simulated missing score
clean_mean = np.nanmean(data_float[:, 2])
data_float[:, 2] = np.nan_to_num(data_float[:, 2], nan=clean_mean)

# 13. Frequency distribution of student ages with np.unique
ages, counts = np.unique(data[:, 0], return_counts=True)
print("Age counts:", dict(zip(ages, counts)))

# 14. Rank students by Science marks using np.argsort
rank_indices = np.argsort(data[:, 2])[::-1]
print("Students ranked by Science (descending):\n", data[rank_indices])

# 15. Cap marks to valid boundaries [0, 100] using np.clip
sanitized_marks = np.clip(data[:, 1], a_min=0, a_max=100)
print("Sanitized Math marks:", sanitized_marks)
```

---

## Key Takeaways

1. **NumPy is essential** for data science and machine learning
2. **Arrays are homogeneous** - all elements must be the same type
3. **Vectorized operations** eliminate the need for explicit loops
4. **Broadcasting** allows operations between different shaped arrays
5. **Indexing and slicing** work similarly to Python lists but with more dimensions
6. **Boolean indexing** is powerful for filtering data
7. **Memory efficiency** and speed make NumPy ideal for large datasets

---

## Common Patterns and Best Practices

### 1. Always import NumPy as np
```python
import numpy as np
```

### 2. Use vectorized operations instead of loops
```python
# Bad
result = []
for i in range(len(arr)):
    result.append(arr[i] * 2)

# Good
result = arr * 2
```

### 3. Use appropriate data types
```python
# For integers
arr = np.array([1, 2, 3], dtype=np.int32)

# For floats
arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
```

### 4. Check array shapes when debugging
```python
print("Array shape:", arr.shape)
print("Array dtype:", arr.dtype)
```

### 5. Use axis parameter for operations on specific dimensions
```python
# Sum along rows (axis=1) or columns (axis=0)
row_sums = np.sum(matrix, axis=1)
col_sums = np.sum(matrix, axis=0)
```

This comprehensive guide covers all the essential NumPy concepts discussed in the video transcript. Practice these examples and exercises to master NumPy fundamentals!