from my_module import func
import pytest

@pytest.fixture
def input_value():
    return 3

def test_method1(input_value):
    assert func(input_value) >= 8 and func(input_value) <= 10

# def test_method2():
#     assert func(4) == 8
