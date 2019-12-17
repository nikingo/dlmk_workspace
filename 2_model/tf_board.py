#coding: UTF-8
import tensorflow as tf
import os

if __name__ == "__main__":

    sess = tf.InteractiveSession()

    # TensorBoard情報出力ディレクトリ
    base_path = os.path.abspath(os.path.dirname(__file__))
    log_dir = os.path.join(base_path, 'log_data\\log_data_test')
    print(log_dir)

    # 計算グラフ定義
    a = tf.constant(1, name='a')
    b = tf.constant(2, name='b')
    op_add = tf.add(a , b)

    # このコマンドで`op_add`をグラフ上に出力
    tf.summary.scalar('op_add', op_add)

    # グラフを書く
    summary_writer = tf.summary.FileWriter(log_dir , sess.graph)

    # 実行
    sess.run(op_add)

    # SummaryWriterクローズ
    summary_writer.close()