
# Assignment -2
Decompose PyTorch operations using only the supported operations in C++ API, and develop comprehensive Gtest cases to validate the decomposed operations, ensuring all edge cases are covered. Then extend your custom c++ ops to python using pybind11 and write pytest to validate the same. The implementation should be thoroughly documented, with well-structured and maintainable code.


## Ops needs to be Decomposed:
* torch.minimum


---

## Prerequisites

A compatible operating system (e.g. Linux, macOS, Windows).

A compatible **C++ compiler** that supports at least C++11.

**CMake** and a compatible build tool for building the project.

Install the pybind11 for wrapping the c++ code.

---

## Installation

1. **Clone the repository:**
    ```

        https://github.com/sharavana20/pytorch_assignments.git

        cd pytorch_assignments

        git checkout assignment-2
    ```

    ---

2. **Building the src folder**
    ```
        cd assignment_two/src/

        mkdir build
        
        cd build
        
        cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..

        make
    ```

    ---

3. **Building the tests folder**

    *Make sure you have cloned the googletest repository inside the tests folder.*
    ```
    
        cd assignment_two/tests/

        mkdir build
        
        cd build
        
        cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..

        make
    ```

    ---

4. **Building the pybind_c++ folder**
    ```
        cd assignment_two/pybind_c++/
    ```
    *set the path for libtorch in **file comm** in cmake and then build it*

    ```
        
        mkdir build
        
        cd build
        
        cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..

        make
    ```

    ---



5. **Creating a virtual environment for pybind11_c++ folder**

    To install, make sure you have installed **Python 3.10**
    ```
        cd assignment_two/pybind_c++/

        python -m venv venv

        source venv/bin/activate - On Linux/MacOS

        venv\Scripts\activate - On Windows
    ```

    
    *Installing it for the virtual environment*

            pip install -r requirements.txt

    ---


---
## Testing

1. **Testing with gtest**

*Make sure you have the build folder*

```
    cd assignment_two/tests/build

    ./test_min
```

*Currently, I added the 3 testcases for each ops.*

---

2. **Testing with pytest**


    ```
    cd assignment_two/pybind_c++

    pytest test_cus_min.py

    ```
