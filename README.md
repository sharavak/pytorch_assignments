
# Assignment -3
Decompose PyTorch operations using only the supported operations, and develop comprehensive pytest cases to validate the decomposed operations, ensuring all edge cases are covered. The implementation should be thoroughly documented, with well-structured and maintainable code.


## Ops needs to be Decomposed:
* torch.addbmm 
* torch.logaddexp


---

## Prerequisites

A compatible operating system (e.g. Linux, macOS, Windows).

A compatible **C++ compiler** that supports at least C++11.

**CMake** and a compatible build tool for building the project.

---

## Installation

1. **Clone the repository:**
    ```

        https://github.com/sharavana20/pytorch_assignments.git

        cd pytorch_assignments

        git checkout assignment-3
    ```

    ---

2. **Building the src folder**
    ```
        cd lib/assignment_three/src/

        mkdir build
        
        cd build
        
        cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..

        make
    ```

    ---

3. **Building the tests folder**

    *Make sure you have cloned the googletest repository inside the tests folder.*
    ```
    
        cd lib/assignment_three/tests/

        mkdir build
        
        cd build
        
        cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..

        make
    ```

    ---

4. **Building the pybind_c++ folder**
    ```
        cd lib/assignment_three/pybind_c++/
    ```
    *set the path for libtorch in **file command** in cmake and then build it*

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
        cd lib/assignment_three/pybind_c++/

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
    cd lib/assignment_three/tests/build

    ./test_cusops
```

*Currently, I added the 3 testcases for each ops.*

---

2. **Testing with pytest**


    ```
    cd lib/assignment_three/pybind_c++

    pytest test_addbmm.py

    pytest test_logaddexp.py
    ```


