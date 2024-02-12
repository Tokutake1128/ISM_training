import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt




def calculate_reachability_with_and(matrix):
    rea_matrix = matrix.copy()
    matrix_list = [rea_matrix.copy()]

    while True:
        rea_next_power = np.dot(rea_matrix, matrix) > 0  # 現在の可到達行列と初期の行列のAND演算
        rea_next_power = rea_next_power.astype(int)
        
        if np.array_equal(rea_matrix, rea_next_power):  # 可到達行列が変化しなくなったら終了
            break
        
        matrix_list.append(rea_next_power.copy())
        rea_matrix = rea_next_power.copy()

    return  rea_matrix,matrix_list


def main():
    st.title('可到達行列の算出イメージ')

    # ユーザーからの行列入力
    n = st.number_input("行列のサイズを入力してください(最大9×9)", min_value=2, max_value=9, value=3, step=1)
    n = int(n)

    df = pd.DataFrame(np.zeros((n, n), dtype=int), columns=range(1, n+1), index=range(1, n+1))

    # チェックボックス
    for i in df.index:
        cols = st.columns(n)
        for j, col in enumerate(cols, start=1):
            if i == j:
                df.at[i, j] = 1
                col.checkbox(f"{i},{j}", value=True, disabled=True, key=f"{i},{j}")
            else:
                checked = col.checkbox(f"{i},{j}", value=bool(df.at[i, j]), key=f"{i},{j}")
                df.at[i, j] = 1 if checked else 0
    
    #単位行列
    identity_matrix = np.eye(n, dtype=int)
    #ユーザーの入力値のみを反映したdf
    adjacency_matrix = df.copy()
    adjacency_matrix[identity_matrix == 1] = 0
    
    #計算用マトリクス
    unit_matrix = df.values

    
    st.write("隣接行列（Adjacency matrix）:A")
    st.dataframe(adjacency_matrix)

    st.write("単位行列(Identity matrix):I")
    st.dataframe(pd.DataFrame(identity_matrix, columns=df.columns, index=df.index))

    # 初期の行列と単位行列を表示
    st.write("A+I")
    st.dataframe(pd.DataFrame(unit_matrix, columns=df.columns, index=df.index))

    # 可到達行列の計算と結果表示
    if st.button('可到達行列の計算'):
        rea_matrix,matrix_list = calculate_reachability_with_and(unit_matrix)
        
        # 各累乗の行列を表示
        for idx, matrix in enumerate(matrix_list, start=1):
            st.write(f"{idx}乗の結果:")
            st.dataframe(pd.DataFrame(matrix, columns=df.columns, index=df.index))
            reachability = idx 
        st.write(f"可到達行列({reachability}乗={reachability+1}乗)")
        st.dataframe(pd.DataFrame(rea_matrix, columns=df.columns, index=df.index))
    
        df_rea = pd.DataFrame(rea_matrix, columns=df.columns, index=df.index)
        df_rea["SUM"] = df_rea.sum(axis=1)
        unique_counts = df_rea['SUM'].nunique()
        st.write(f"ノードの階層数：{unique_counts}階層")
        st.write("※階層数はノードの階層を示しています")


        # ネットワークグラフ
        G = nx.DiGraph()
        for i in range(1, n + 1):
            G.add_node(str(i))

        for i in adjacency_matrix.index:
            for j in adjacency_matrix.columns[adjacency_matrix.loc[i] == 1]:
                G.add_edge(str(i), str(j))

        pos_dict = {}
        levels = df_rea['SUM'].unique().tolist()
        levels.sort()

        # 各レベルでのノードを保持
        level_nodes = {level: [] for level in levels}
        for node, row in df_rea.iterrows():
            level_nodes[row['SUM']].append(str(node))

        # 2つ以上上のノードにリンクするノードの特定
        upward_links = {}
        for node in G.nodes():
            node_level = df_rea.at[int(node), 'SUM']
            for target in G.successors(node):
                target_level = df_rea.at[int(target), 'SUM']
                if target_level > node_level + 1:
                    if node not in upward_links:
                        upward_links[node] = []
                    upward_links[node].append(target)

        # X軸位置の調整
        x_positions = {level: 0 for level in levels}  # 各レベルでの現在のX軸位置
        for level in levels:
            nodes = level_nodes[level]
            for node in nodes:
                if node in upward_links:
                    # 2つ以上上のノードにリンクするノードはX軸を調整
                    pos_dict[node] = (x_positions[level] + 1, -level)
                    x_positions[level] += 2  # 調整した分だけX軸位置を変更
                else:
                    # それ以外のノードは現在のX軸位置に配置
                    pos_dict[node] = (x_positions[level], -level)
                    x_positions[level] += 1  

        # 既に使われている位置を避ける
        occupied_positions = set(pos_dict.values())
        for node, pos in pos_dict.items():
            while pos in occupied_positions:
                pos = (pos[0] + 1, pos[1])
                pos_dict[node] = pos
            occupied_positions.add(pos)

        # グラフの描画
        fig, ax = plt.subplots()
        nx.draw(G, pos_dict, with_labels=True, arrows=True, ax=ax)
        
        # リンクの上にラベルを表示（オンオフ切り替えが出来ない，仕様？）
        labels = {(str(u), str(v)): f"{u}→{v}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos_dict, edge_labels=labels)
        
        st.pyplot(fig)

if __name__ == '__main__':
    main()
