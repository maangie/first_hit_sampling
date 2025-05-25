import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Any, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.font_manager as fm

# フォントの明示的指定
font_path = "~/Library/Fonts/VL-Gothic-Regular.ttf"
jp_font = fm.FontProperties(fname=font_path)
font_name = jp_font.get_name()

def pop_random(arr_np: NDArray[np.int_]) -> Tuple[NDArray[np.int_], int]:
  """
  arr_np からランダムに要素を削除し、
  (削除後の新しい配列, 削除した要素の値) を返す関数。

  Parameters
  ----------
  arr_np : np.ndarray
    削除対象のNumPy配列。

  Returns
  -------
  new_arr : np.ndarray
    要素が1つ減った新しい配列。
  val : int
    削除された要素の値。

  Raises
  ------
  IndexError
    配列が空だったらエラーを出すばい。
  """
  n = arr_np.size
  if n == 0:
    raise IndexError("配列が空だっちゃけん、要素を削除できんばい。")
  else:
    # ランダムにインデックスを1つ選択
    idx = np.random.randint(0, n)
    val = arr_np[idx]
    # np.delete は新しい配列を返すけん上書き
    new_arr = np.delete(arr_np, idx)
    return new_arr, val

def create_random_one_array(length: int = 1000, dtype: Any=int) -> NDArray[Any]:
  """
  指定した長さのゼロ配列を作って、その中のランダムな1箇所を1にする関数。

  Parameters
  ----------
  length : int
    配列の長さ（デフォルトは1000）。
  dtype : data-type
    配列のデータ型（デフォルトは int）。

  Returns
  -------
  arr : np.ndarray
    要素がすべて0で、ランダムに1つだけ1が入った配列。
  """
  # ゼロで埋めた配列を作成
  arr = np.zeros(length, dtype=dtype)
  # ランダムにインデックスを1つ選んで1をセット
  idx = np.random.choice(arr.size, size=1, replace=False)[0]
  arr[idx] = 1
  return arr

def draw_until_hit(arr_np: NDArray[np.int_]) -> int:
  """
  arr_np からランダムに要素を取り出し（pop_random を使って捨てつつ）、
  値が 1 になるまで何回くじを引いたか（回数）を返す関数。

  Parameters
  ----------
  arr_np : NDArray[np.int_]
    0 と 1 だけが入った一次元整数型配列。

  Returns
  -------
  count : int
    あたり（値が1）が出るまでの引いた回数。

  Raises
  ------
  IndexError
    配列が途中で空になった場合（安全策として）。
  """
  count = 0
  current = arr_np
  while True:
    count += 1
    current, val = pop_random(current)
    if val == 1:
      # あたり見つけたけんループば抜けるばい
      break
    # 空配列になったらエラーにしとくと安全たい
    if current.size == 0:
      raise IndexError("配列が空になったばい…あたりが見つからんばい。")
  return count

def simulate_hit_histogram(
  trials: int = 10000,
  length: int = 1000
) -> None:
  """
  draw_until_hit() を使って「あたりが出るまでに何回引いたか」を
  指定回数(trials)シミュレーションし、そのヒストグラムを表示する関数。

  Parameters
  ----------
  trials : int
    シミュレーション回数（デフォルトは10000）。
  length : int
    1回のくじ配列の長さ（デフォルトは1000）。
  """
  results: List[int] = []
  for _ in range(trials):
    # 1. くじ配列を作成
    arr = create_random_one_array(length)
    # 2. あたりが出るまで引いて回数を取得
    count = draw_until_hit(arr)
    results.append(count)

  # ヒストグラム描画
  plt.figure()
  # ビンは1回目～最大回数まで
  plt.hist(results, bins=30)
  plt.xlabel("引いた回数", fontproperties=jp_font)
  plt.ylabel("発生回数", fontproperties=jp_font)
  plt.title(f"{trials} 回のシミュレーション結果ヒストグラム", fontproperties=jp_font)
  plt.tight_layout()
  plt.show()

simulate_hit_histogram(trials=10000, length=1000)
