
# Assignment -1
Decompose PyTorch operations using only the supported operations, and develop comprehensive pytest cases to validate the decomposed operations, ensuring all edge cases are covered. The implementation should be thoroughly documented, with well-structured and maintainable code.

## Supported ops
* torch.add
* torch.sub
* torch.mul
* torch.div

## Ops needs to be Decomposed:
* torch.pow
* torch.nn.Softmax



## Table of Contents

- [Installation](#installation)
- [Setting up the Development Environment](#setting-up-the-development-environment)
- [Running Tests](#running-tests)

---

## Installation

1. **Clone the repository:**

    `https://github.com/sharavana20/pytorch_assignments.git`

    `cd pytorch_assignments`

    `git checkout assignment-1`

2. **Create a virtual environment**

    To install, make sure you have installed Python 3.12

    `python3 -m venv venv`

    `source venv/bin/activate - On Linux/MacOS`

    `venv\Scripts\activate - On Windows`

3. **Install project dependencies:**

    `pip install -r requirements.txt`



## Testing

`cd lib/assignment_one/tests`

`pytest test_pow.py`

`pytest test_softmax.py`
