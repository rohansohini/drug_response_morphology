'''
Run directly: python3 tests/analyses/0_naive_vs_vem/test_allocation.py
'''
import os
import sys

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'analyses', '0_naive_vs_vem')
sys.path.insert(0, ANALYSIS_DIR)
from allocation import water_fill_allocate


def test_equal_distribution_when_capacity_allows():
    result = water_fill_allocate({'a': 100, 'b': 100, 'c': 100}, budget=30)
    assert sum(result.values()) == 30
    assert result == {'a': 10, 'b': 10, 'c': 10}
    print("PASS: equal distribution when every group has ample capacity")


def test_capped_group_redistributes_leftover():
    # 'a' can only give 5; the other 25 units should split evenly between b and c
    result = water_fill_allocate({'a': 5, 'b': 100, 'c': 100}, budget=30)
    assert sum(result.values()) == 30
    assert result['a'] == 5
    assert result['b'] == 12 or result['b'] == 13
    assert result['c'] == 12 or result['c'] == 13
    assert result['b'] + result['c'] == 25
    print("PASS: a capped group's shortfall is redistributed to the rest")


def test_budget_exceeds_total_availability():
    result = water_fill_allocate({'a': 5, 'b': 5, 'c': 5}, budget=100)
    assert result == {'a': 5, 'b': 5, 'c': 5}
    assert sum(result.values()) == 15  # never exceeds what's actually available
    print("PASS: allocation never exceeds total real availability, even with a huge budget")


def test_small_budget_many_groups():
    # budget smaller than the number of groups - no group should get more than 1
    result = water_fill_allocate({'a': 10, 'b': 10, 'c': 10, 'd': 10}, budget=2)
    assert sum(result.values()) == 2
    assert all(n <= 1 for n in result.values())
    print("PASS: a budget smaller than the group count still allocates exactly (no over/under)")


def test_empty_groups():
    result = water_fill_allocate({}, budget=50)
    assert result == {}
    print("PASS: no groups means an empty allocation, not an error")


def test_zero_budget():
    result = water_fill_allocate({'a': 10, 'b': 10}, budget=0)
    assert result == {'a': 0, 'b': 0}
    print("PASS: zero budget allocates nothing to anyone")


if __name__ == '__main__':
    test_equal_distribution_when_capacity_allows()
    test_capped_group_redistributes_leftover()
    test_budget_exceeds_total_availability()
    test_small_budget_many_groups()
    test_empty_groups()
    test_zero_budget()
    print("\nAll tests passed.")
