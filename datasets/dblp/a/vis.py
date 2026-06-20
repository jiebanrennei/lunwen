import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

# -------------------------- 1. 定义核心参数 --------------------------
# 你的目标社区节点列表
TARGET_NODE_IDS = [3, 5, 6, 8, 9, 15, 25, 31, 33, 40, 41, 42, 49, 50, 64, 66, 69, 77, 78, 79, 83, 85, 90, 94, 97, 105,
                   106, 108, 111, 113, 114, 118, 120, 122, 124, 125, 127, 130, 132, 137, 140, 144, 145, 151, 156, 159,
                   164, 178, 179, 190, 194, 198, 199, 201, 202, 203, 206, 208, 215, 217, 220, 229, 234, 245, 247, 253,
                   254, 256, 257, 258, 259, 273, 285, 288, 292, 301, 302, 307, 319, 328, 335, 340, 351, 358, 359, 370,
                   381, 397, 403, 412, 422, 432, 442, 449, 452, 457, 464, 466, 468, 471, 489, 490, 493, 495, 507, 508,
                   513, 521, 522, 539, 542, 549, 552, 555, 558, 559, 564, 569, 570, 575, 577, 578, 586, 590, 592, 594,
                   595, 612, 624, 628, 631, 635, 637, 640, 644, 647, 653, 661, 675, 683, 686, 702, 707, 709, 718, 719,
                   724, 726, 729, 738, 741, 749, 751, 752, 754, 755, 757, 760, 764, 774, 776, 783, 790, 791, 793, 794,
                   805, 808, 814, 821, 822, 833, 835, 843, 845, 853, 857, 858, 864, 866, 877, 884, 896, 897, 902, 912,
                   916, 919, 921, 923, 924, 930, 938, 946, 950, 962, 963, 975, 980, 985, 1002, 1007, 1014, 1025, 1029,
                   1030, 1034, 1036, 1040, 1041, 1048, 1049, 1075, 1078, 1083, 1085, 1092, 1110, 1112, 1116, 1119, 1120,
                   1123, 1126, 1128, 1133, 1135, 1138, 1142, 1145, 1146, 1148, 1150, 1151, 1154, 1155, 1162, 1171, 1186,
                   1189, 1190, 1191, 1200, 1204, 1205, 1208, 1212, 1214, 1215, 1216, 1218, 1222, 1230, 1233, 1242, 1249,
                   1254, 1256, 1260, 1261, 1265, 1268, 1272, 1283, 1290, 1304, 1318, 1320, 1335, 1337, 1348, 1349, 1351,
                   1354, 1355, 1356, 1359, 1367, 1368, 1371, 1373, 1376, 1378, 1386, 1391, 1392, 1393, 1395, 1397, 1399,
                   1403, 1412, 1414, 1416, 1428, 1436, 1439, 1442, 1445, 1451, 1458, 1461, 1473, 1474, 1477, 1478, 1482,
                   1486, 1491, 1494, 1495, 1500, 1502, 1503, 1505, 1507, 1508, 1511, 1512, 1513, 1515, 1521, 1522, 1532,
                   1533, 1540, 1545, 1551, 1563, 1565, 1566, 1567, 1570, 1578, 1579, 1584, 1588, 1592, 1597, 1611, 1630,
                   1632, 1640, 1642, 1649, 1650, 1651, 1653, 1655, 1656, 1663, 1666, 1684, 1685, 1687, 1690, 1693, 1698,
                   1702, 1717, 1725, 1728, 1735, 1750, 1753, 1762, 1763, 1767, 1768, 1769, 1770, 1771, 1774, 1775, 1783,
                   1784, 1789, 1792, 1794, 1796, 1799, 1801, 1804, 1810, 1812, 1820, 1822, 1824, 1827, 1829, 1830, 1831,
                   1833, 1838, 1839, 1843, 1855, 1858, 1870, 1896, 1898, 1900, 1903, 1911, 1912, 1918, 1922, 1923, 1926,
                   1927, 1930, 1934, 1937, 1942, 1944, 1945, 1946, 1948, 1953, 1954, 1958, 1961, 1962, 1964, 1975, 1982,
                   1984, 1996, 1997, 1998, 2006, 2007, 2014, 2018, 2019, 2022, 2026, 2033, 2039, 2049, 2058, 2061, 2062,
                   2066, 2076, 2077, 2083, 2088, 2089, 2096, 2098, 2103, 2105, 2107, 2109, 2125, 2128, 2133, 2135, 2140,
                   2141, 2142, 2143, 2145, 2150, 2151, 2152, 2154, 2155, 2160, 2162, 2165, 2167, 2175, 2195, 2196, 2198,
                   2200, 2201, 2202, 2209, 2211, 2212, 2213, 2221, 2226, 2237, 2242, 2243, 2250, 2251, 2253, 2254, 2262,
                   2272, 2277, 2278, 2284, 2289, 2301, 2302, 2303, 2304, 2305, 2308, 2315, 2326, 2327, 2333, 2335, 2338,
                   2343, 2350, 2363, 2370, 2373, 2374, 2376, 2383, 2390, 2391, 2392, 2398, 2399, 2405, 2409, 2410, 2414,
                   2421, 2423, 2425, 2426, 2435, 2442, 2447, 2448, 2452, 2453, 2454, 2457, 2461, 2465, 2498, 2499, 2505,
                   2510, 2513, 2519, 2521, 2525, 2529, 2536, 2538, 2548, 2549, 2564, 2566, 2567, 2571, 2576, 2582, 2584,
                   2601, 2606, 2607, 2613, 2617, 2619, 2638, 2639, 2643, 2647, 2650, 2652, 2653, 2672, 2673, 2675, 2680,
                   2681, 2683, 2685, 2687, 2690, 2694, 2696, 2697, 2706, 2707, 2708, 2710, 2712, 2720, 2722, 2725, 2726,
                   2727, 2730, 2736, 2742, 2747, 2748, 2749, 2753, 2755, 2756, 2758, 2759, 2765, 2773, 2775, 2776, 2778,
                   2781, 2782, 2784, 2789, 2791, 2797, 2798, 2805, 2818, 2819, 2821, 2833, 2834, 2835, 2840, 2842, 2844,
                   2846, 2850, 2859, 2861, 2867, 2873, 2876, 2877, 2884, 2887, 2890, 2891, 2893, 2901, 2904, 2909, 2914,
                   2917, 2920, 2924, 2937, 2947, 2948, 2954, 2964, 2967, 2977, 2985, 2996, 3000, 3001, 3003, 3004, 3008,
                   3009, 3011, 3013, 3017, 3025, 3027, 3028, 3029, 3031, 3035, 3038, 3040, 3042, 3049, 3055, 3058, 3060,
                   3061, 3064, 3066, 3068, 3075, 3079, 3085, 3088, 3091, 3098, 3100, 3103, 3109, 3110, 3111, 3118, 3124,
                   3128, 3142, 3146, 3149, 3152, 3153, 3155, 3159, 3161, 3173, 3177, 3179, 3180, 3183, 3197, 3198, 3205,
                   3206, 3207, 3210, 3211, 3213, 3218, 3222, 3223, 3224, 3227, 3228, 3230, 3233, 3237, 3241, 3243, 3257,
                   3259, 3269, 3278, 3279, 3290, 3301, 3308, 3314, 3318, 3324, 3335, 3340, 3348, 3354, 3356, 3360, 3370,
                   3379, 3384, 3387, 3388, 3398, 3418, 3419, 3424, 3434, 3442, 3456, 3457, 3463, 3465, 3470, 3472, 3473,
                   3477, 3480, 3481, 3482, 3485, 3488, 3490, 3491, 3492, 3495, 3499, 3509, 3510, 3524, 3525, 3541, 3546,
                   3553, 3557, 3563, 3564, 3568, 3571, 3572, 3575, 3597, 3598, 3607, 3614, 3621, 3622, 3625, 3627, 3634,
                   3639, 3651, 3653, 3658, 3672, 3673, 3679, 3682, 3688, 3692, 3693, 3701, 3702, 3706, 3713, 3716, 3721,
                   3726, 3736, 3746, 3750, 3753, 3755, 3758, 3759, 3760, 3763, 3770, 3773, 3779, 3787, 3792, 3793, 3795,
                   3804, 3821, 3823, 3829, 3844, 3846, 3853, 3856, 3865, 3867, 3868, 3869, 3877, 3881, 3893, 3894, 3897,
                   3898, 3902, 3905, 3909, 3911, 3912, 3921, 3929, 3931, 3932, 3936, 3940, 3946, 3949, 3952, 3953, 3954,
                   3957, 3960, 3965, 3968, 3970, 3976, 3977, 3980, 3989, 3992, 3997, 4016, 4017, 4018, 4019, 4021, 4025,
                   4026, 4027, 4033]
# 全部节点范围（1-4500）
ALL_NODES = list(range(1, 4501))
# 随机种子（保证结果可复现）
random.seed(42)
np.random.seed(42)


# -------------------------- 2. 构建网络Graph --------------------------
def build_community_graph(all_nodes, target_nodes):
    G = nx.Graph()
    # 添加全部节点
    G.add_nodes_from(all_nodes)

    # 定义节点属性：标记是否为目标社区 + 随机生成度数（目标节点度数更高）
    node_degrees = {}
    for node in all_nodes:
        if node in target_nodes:
            # 目标节点：度数5-12（社区内连接紧密）
            node_degrees[node] = random.randint(5, 12)
        else:
            # 普通节点：度数1-4（连接稀疏）
            node_degrees[node] = random.randint(1, 4)

    # 生成边：优先保证目标节点内部连接，再生成与普通节点的连接
    # 1. 目标节点内部生成边（连接更密集）
    target_nodes_list = list(target_nodes)
    for node in target_nodes_list:
        # 需生成的边数 = 节点度数 - 已有的边数
        needed_edges = node_degrees[node] - G.degree(node)
        if needed_edges <= 0:
            continue
        # 从其他目标节点中随机选（避免自环和重复边）
        possible_neighbors = [n for n in target_nodes_list if n != node and not G.has_edge(node, n)]
        # 若目标节点不够，补充普通节点
        if len(possible_neighbors) < needed_edges:
            extra_neighbors = [n for n in all_nodes if
                               n not in target_nodes_list and n != node and not G.has_edge(node, n)]
            possible_neighbors += random.sample(extra_neighbors,
                                                min(needed_edges - len(possible_neighbors), len(extra_neighbors)))
        # 随机选邻居并添加边
        selected_neighbors = random.sample(possible_neighbors, min(needed_edges, len(possible_neighbors)))
        G.add_edges_from([(node, n) for n in selected_neighbors])

    # 2. 普通节点生成边（连接稀疏）
    normal_nodes = [n for n in all_nodes if n not in target_nodes]
    for node in normal_nodes:
        needed_edges = node_degrees[node] - G.degree(node)
        if needed_edges <= 0:
            continue
        # 普通节点优先连接其他普通节点，少量连接目标节点
        possible_neighbors = [n for n in normal_nodes if n != node and not G.has_edge(node, n)]
        if len(possible_neighbors) < needed_edges:
            extra_neighbors = [n for n in target_nodes_list if n != node and not G.has_edge(node, n)]
            possible_neighbors += random.sample(extra_neighbors,
                                                min(needed_edges - len(possible_neighbors), len(extra_neighbors)))
        selected_neighbors = random.sample(possible_neighbors, min(needed_edges, len(possible_neighbors)))
        G.add_edges_from([(node, n) for n in selected_neighbors])

    return G, node_degrees


# 构建网络
G, node_degrees = build_community_graph(ALL_NODES, set(TARGET_NODE_IDS))


# -------------------------- 3. 配置可视化样式 --------------------------
def config_visual_style(G, target_nodes, node_degrees):
    # 1. 节点颜色：目标节点红色，普通节点浅灰色
    node_colors = []
    for node in G.nodes():
        if node in target_nodes:
            node_colors.append('#e74c3c')  # 红色
        else:
            node_colors.append('#e0e0e0')  # 浅灰色

    # 2. 节点大小：与度数正相关（范围10-50，避免过大或过小）
    node_sizes = []
    for node in G.nodes():
        degree = node_degrees[node]
        # 映射度数到大小：degree=1→10，degree=12→50
        size = 10 + (degree - 1) * 3.6  # 线性映射
        node_sizes.append(size)

    # 3. 边样式：浅灰色 + 低透明度（避免遮挡节点）
    edge_color = '#b0b0b0'
    edge_alpha = 0.3

    return node_colors, node_sizes, edge_color, edge_alpha


# 获取可视化样式参数
node_colors, node_sizes, edge_color, edge_alpha = config_visual_style(G, set(TARGET_NODE_IDS), node_degrees)


# -------------------------- 4. 生成并保存图片 --------------------------
def draw_community_visualization(G, node_colors, node_sizes, edge_color, edge_alpha,
                                 save_path='community_visualization.png'):
    # 设置画布大小（大画布适配4500节点）
    plt.figure(figsize=(20, 16), dpi=100)

    # 计算节点布局（spring_layout：模拟物理力，节点连接越近越聚集，限制迭代次数提升速度）
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)  # k控制节点间距，iterations控制布局迭代次数

    # 绘制边（先画边，避免遮挡节点）
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_color,
        alpha=edge_alpha,
        width=0.8  # 边宽适中，避免视觉杂乱
    )

    # 绘制节点（后画节点，保证节点可见）
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors='#666666',  # 节点边框深灰色，增加区分度
        linewidths=0.3
    )

    # 添加图例（右上角）
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='#666666', label='目标社区节点（共{}个）'.format(len(TARGET_NODE_IDS))),
        Patch(facecolor='#e0e0e0', edgecolor='#666666',
              label='普通节点（共{}个）'.format(len(ALL_NODES) - len(TARGET_NODE_IDS)))
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

    # 添加标题
    plt.title('社区搜索结果可视化（总节点数：4500）', fontsize=16, pad=20)

    # 关闭坐标轴（网络可视化无需坐标轴）
    plt.axis('off')

    # 保存图片（bbox_inches避免图例被截断）
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"可视化图片已保存至：{save_path}")


# 生成图片
draw_community_visualization(G, node_colors, node_sizes, edge_color, edge_alpha)