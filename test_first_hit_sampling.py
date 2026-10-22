import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pytest

import first_hit_sampling
from first_hit_sampling import create_random_one_array, draw_until_hit, pop_random

def test_pop_random_removes_one_element():
  """pop_random は要素を 1 つ取り除き、取り除いた値を返す。"""
  arr = np.array([10, 20, 30, 40])
  new_arr, val = pop_random(arr)
  assert new_arr.size == 3
  assert val in arr
  assert sorted(new_arr.tolist() + [val]) == arr.tolist()

def test_pop_random_does_not_modify_input():
  """pop_random は元の配列を書き換えない。"""
  arr = np.array([1, 2, 3])
  pop_random(arr)
  assert arr.tolist() == [1, 2, 3]

def test_pop_random_empty_raises():
  """空の配列を渡すと IndexError になる。"""
  with pytest.raises(IndexError):
    pop_random(np.array([], dtype=int))

def test_create_random_one_array():
  """指定した長さで、1 がちょうど 1 つだけ入った配列を作る。"""
  arr = create_random_one_array(50)
  assert arr.size == 50
  assert arr.sum() == 1
  assert set(arr.tolist()) == {0, 1}

def test_create_random_one_array_dtype():
  """dtype を指定すると、その型の配列になる。"""
  assert create_random_one_array(10, dtype=float).dtype == np.float64

def test_draw_until_hit_single_element():
  """要素が 1 つ（あたり）だけなら、1 回で当たる。"""
  assert draw_until_hit(np.array([1])) == 1

def test_draw_until_hit_range():
  """引いた回数は 1 以上、配列の長さ以下になる。"""
  np.random.seed(0)
  for _ in range(100):
    count = draw_until_hit(create_random_one_array(20))
    assert 1 <= count <= 20

def test_draw_until_hit_no_hit_raises():
  """あたりがなければ、配列が空になった時点で IndexError になる。"""
  with pytest.raises(IndexError):
    draw_until_hit(np.zeros(5, dtype=int))

def test_simulate_hit_histogram(monkeypatch):
  """simulate_hit_histogram はヒストグラムを 1 つ描く。

  CI には日本語フォントがないので、既定のフォントに差し替える。

  引数:
    monkeypatch: pytest の monkeypatch フィクスチャ
  """
  monkeypatch.setattr(first_hit_sampling, "jp_font", fm.FontProperties())
  monkeypatch.setattr(plt, "show", lambda: None)
  plt.close("all")
  first_hit_sampling.simulate_hit_histogram(trials=20, length=10)
  ax = plt.gcf().axes[0]
  assert sum(patch.get_height() for patch in ax.patches) == 20
  plt.close("all")
