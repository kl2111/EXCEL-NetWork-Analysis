import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
from pathlib import Path
import os
import base64

st.set_page_config(page_title="通用网络分析", layout="wide")
st.title("通用网络关系分析工具")

# 侧边栏 - 分析模式选择
analysis_mode = st.sidebar.radio(
    "选择分析模式",
    ["二分网络分析", "共现网络分析"],
    index=0
)

# 文件上传
uploaded_file = st.file_uploader("上传数据文件（CSV或Excel）", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # 读取数据
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.subheader("数据预览")
    st.dataframe(df.head())
    
    # 按分析模式配置界面
    if analysis_mode == "二分网络分析":
        st.subheader("二分网络配置")
        
        all_columns = df.columns.tolist()
        
        source_col = st.selectbox(
            "选择源节点列（第一类节点）", 
            options=all_columns,
            index=0 if all_columns else None
        )
        
        # 提供更智能的列选择逻辑，处理交替的目标节点-链接结构
        st.info("如果您的数据包含'目标节点列'和'链接列'交替排列，请只选择目标节点列（不选链接列）")
        
        # 尝试识别可能的目标节点列（假设链接列可能包含"link"或"url"）
        likely_target_cols = [col for col in all_columns 
                             if col != source_col 
                             and not any(term in col.lower() for term in ["link", "url", "网址", "链接"])]
        
        target_cols = st.multiselect(
            "选择目标节点列（第二类节点，可多选）", 
            options=[col for col in all_columns if col != source_col],
            default=likely_target_cols[:5] if likely_target_cols else None
        )
        
        # 链接列选项
        use_links = st.checkbox("使用节点链接数据（如有）", value=False)
        
        if use_links:
            st.info("请为每个目标节点列选择对应的链接列")
            link_cols = {}
            for col in target_cols:
                link_options = [link_col for link_col in all_columns 
                               if link_col != source_col and link_col not in target_cols]
                # 尝试智能匹配链接列（例如：如果有"节点1"和"节点1链接"）
                default_link = next((link for link in link_options if col in link or link in col), None)
                link_cols[col] = st.selectbox(
                    f"'{col}'的链接列", 
                    options=["无"] + link_options,
                    index=0 if default_link is None else link_options.index(default_link) + 1
                )
    else:  # 共现网络分析
        st.subheader("共现网络配置")
        
        all_columns = df.columns.tolist()
        
        # 共现实体列选择
        entity_cols = st.multiselect(
            "选择实体列（这些列中的值将被视为网络节点）", 
            options=all_columns,
            default=all_columns[:2] if len(all_columns) >= 2 else None
        )
        
        # 共现权重选项
        weight_col = st.selectbox(
            "选择权重列（可选，用于边的权重）", 
            options=["无"] + all_columns,
            index=0
        )
        
        # 共现定义方法
        cooccurrence_method = st.radio(
            "共现定义方法",
            ["同行共现（行级）", "字段组合（实体对）"],
            index=0,
            help="行级：同一行中出现的实体视为共现；实体对：数据已经是实体对形式，直接构建网络"
        )
    
    # 网络参数设置
    st.sidebar.subheader("网络参数设置")
    
    color_scheme = st.sidebar.selectbox(
        "节点颜色方案",
        options=["分类着色", "度中心性热力图"],
        index=0
    )
    
    network_height = st.sidebar.slider("网络图高度", 400, 1000, 600)
    
    show_physics_options = st.sidebar.checkbox("显示物理引擎选项", value=False)
    physics_enabled = st.sidebar.checkbox("启用物理引擎", value=True)
    
    # 高级分析选项
    st.sidebar.subheader("高级分析选项")
    run_community_detection = st.sidebar.checkbox("运行社区检测", value=True)
    show_top_nodes = st.sidebar.slider("显示关键节点数量", 5, 30, 10)
    min_edge_weight = st.sidebar.slider("最小边权重", 1.0, 10.0, 1.0, 0.1, 
                                       disabled=(analysis_mode != "共现网络分析" or weight_col == "无"))
    
    # 构建网络按钮
    if st.button("构建网络图"):
        if (analysis_mode == "二分网络分析" and source_col and target_cols) or \
           (analysis_mode == "共现网络分析" and entity_cols):
            
            with st.spinner("正在构建网络..."):
                # 创建图
                G = nx.Graph()
                
                # 根据分析模式构建网络
                if analysis_mode == "二分网络分析":
                    # 添加源节点（第一类节点）
                    sources = df[source_col].unique()
                    for source in sources:
                        if pd.notna(source) and source:  # 检查非空值
                            G.add_node(source, group=1, title=f"类别1: {source}")
                    
                    # 添加目标节点（第二类节点）和边
                    for _, row in df.iterrows():
                        source = row[source_col]
                        if not pd.notna(source) or not source:
                            continue
                            
                        for col in target_cols:
                            target = row[col]
                            if pd.notna(target) and target:  # 检查非空值
                                # 确保目标节点存在
                                if target not in G:
                                    G.add_node(target, group=2, title=f"类别2: {target}")
                                
                                # 检查是否有链接数据
                                link_info = ""
                                if 'use_links' in locals() and use_links and link_cols.get(col) != "无":
                                    link_data = row[link_cols[col]]
                                    if pd.notna(link_data) and link_data:
                                        link_info = f"<br>链接: <a href='{link_data}' target='_blank'>{link_data}</a>"
                                        # 更新节点标题以包含链接
                                        G.nodes[target]['title'] = f"类别2: {target}{link_info}"
                                
                                # 添加边
                                G.add_edge(source, target)
                else:  # 共现网络分析
                    if cooccurrence_method == "同行共现（行级）":
                        # 行级共现：同一行中的实体被视为共现
                        for _, row in df.iterrows():
                            # 获取当前行中的所有实体
                            entities = []
                            for col in entity_cols:
                                value = row[col]
                                if pd.notna(value) and value:
                                    # 如果是列表或集合形式，扩展它
                                    if isinstance(value, (list, set, tuple)):
                                        entities.extend(value)
                                    else:
                                        entities.append(value)
                            
                            # 为独特的实体添加节点
                            unique_entities = set(entities)
                            for entity in unique_entities:
                                if entity not in G:
                                    G.add_node(entity, group=1, title=f"实体: {entity}")
                            
                            # 添加所有可能的共现边
                            entities_list = list(unique_entities)
                            weight = 1
                            if weight_col != "无" and pd.notna(row[weight_col]):
                                try:
                                    weight = float(row[weight_col])
                                except (ValueError, TypeError):
                                    weight = 1
                            
                            for i in range(len(entities_list)):
                                for j in range(i+1, len(entities_list)):
                                    e1, e2 = entities_list[i], entities_list[j]
                                    # 更新边权重
                                    if G.has_edge(e1, e2):
                                        G[e1][e2]['weight'] += weight
                                    else:
                                        G.add_edge(e1, e2, weight=weight)
                    else:  # 字段组合（直接实体对）
                        if len(entity_cols) < 2:
                            st.error("实体对分析需要至少选择两列！")
                        else:
                            # 通常使用前两列作为实体对
                            entity1_col, entity2_col = entity_cols[0], entity_cols[1]
                            
                            # 遍历所有行
                            for _, row in df.iterrows():
                                entity1 = row[entity1_col]
                                entity2 = row[entity2_col]
                                
                                # 检查值有效性
                                if not pd.notna(entity1) or not entity1 or not pd.notna(entity2) or not entity2:
                                    continue
                                
                                # 添加节点
                                if entity1 not in G:
                                    G.add_node(entity1, group=1, title=f"实体: {entity1}")
                                if entity2 not in G:
                                    G.add_node(entity2, group=2, title=f"实体: {entity2}")
                                
                                # 添加边
                                weight = 1
                                if weight_col != "无" and pd.notna(row[weight_col]):
                                    try:
                                        weight = float(row[weight_col])
                                    except (ValueError, TypeError):
                                        weight = 1
                                
                                # 过滤低于最小权重的边
                                if weight >= min_edge_weight:
                                    if G.has_edge(entity1, entity2):
                                        G[entity1][entity2]['weight'] += weight
                                    else:
                                        G.add_edge(entity1, entity2, weight=weight)
                
                # 运行社区检测（如果启用）
                if run_community_detection and len(G.nodes()) > 0:
                    try:
                        from community import best_partition
                        partition = best_partition(G)
                        # 更新节点社区属性
                        nx.set_node_attributes(G, partition, 'community')
                    except ImportError:
                        st.warning("社区检测需要python-louvain包。可以运行 `pip install python-louvain` 安装。")
                        # 使用节点组作为默认社区
                        partition = {node: G.nodes[node].get('group', 0) for node in G.nodes()}
                        nx.set_node_attributes(G, partition, 'community')
                
                # 计算度中心性
                degree_centrality = nx.degree_centrality(G)
                
                # Pyvis可视化
                net = Network(height=f"{network_height}px", width="100%", directed=False, notebook=False)
                
                # 配置物理引擎
                if show_physics_options:
                    net.toggle_physics(physics_enabled)
                    if physics_enabled:
                        net.barnes_hut(spring_length=250, spring_strength=0.001, damping=0.09)
                
                # 添加节点到pyvis网络
                for node in G.nodes():
                    # 获取节点属性
                    group = G.nodes[node].get('group', 1)
                    community = G.nodes[node].get('community', 0)
                    
                    if color_scheme == "分类着色":
                        if run_community_detection:
                            # 使用社区ID生成颜色
                            colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", 
                                     "#1abc9c", "#d35400", "#c0392b", "#16a085", "#8e44ad"]
                            color = colors[community % len(colors)]
                        else:
                            # 使用组ID
                            color = "#3498db" if group == 1 else "#e74c3c"
                    else:
                        # 使用度中心性作为颜色梯度
                        value = degree_centrality[node]
                        intensity = int(255 * value)
                        if group == 1:  # 第一组
                            color = f"rgb(53, {intensity}, 255)"
                        else:  # 第二组
                            color = f"rgb(255, {intensity}, 53)"
                    
                    # 节点大小基于度中心性
                    size = 10 + 40 * degree_centrality[node]
                    
                    # 获取节点标题（悬停信息）
                    title = G.nodes[node].get('title', f"节点: {node}")
                    if run_community_detection:
                        title += f"<br>社区: {community}"
                    title += f"<br>中心性: {degree_centrality[node]:.4f}"
                    
                    # 节点连接数
                    degree = G.degree(node)
                    title += f"<br>连接数: {degree}"
                    
                    net.add_node(node, 
                                label=str(node), 
                                title=title,
                                color=color,
                                size=size)
                
                # 添加边，对于共现网络考虑权重
                for edge in G.edges(data=True):
                    source, target = edge[0], edge[1]
                    edge_data = edge[2]
                    
                    # 检查是否有权重
                    if 'weight' in edge_data:
                        weight = edge_data['weight']
                        # 权重影响边的宽度
                        width = 1 + min(weight, 10) / 2  # 最大宽度限制
                        net.add_edge(source, target, value=weight, title=f"权重: {weight:.2f}", width=width)
                    else:
                        net.add_edge(source, target)
                
                # 保存为HTML文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
                    path = tmpfile.name
                    net.save_graph(path)
                
                # 将网络信息添加到页面
                with open(path, 'r', encoding='utf-8') as f:
                    html_data = f.read()
                
                # 提供下载HTML文件的功能
                def get_binary_file_downloader_html(bin_file, file_label='文件'):
                    with open(bin_file, 'rb') as f:
                        data = f.read()
                    b64 = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}" class="download-link">下载 {file_label}</a>'
                    return href
                
                # 显示结果统计
                st.subheader("网络统计")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("节点总数", len(G.nodes()))
                
                if analysis_mode == "二分网络分析":
                    col2.metric("类别1节点数", len([n for n in G.nodes() if G.nodes[n].get('group', 0) == 1]))
                    col3.metric("类别2节点数", len([n for n in G.nodes() if G.nodes[n].get('group', 0) == 2]))
                else:
                    if run_community_detection:
                        communities = set(nx.get_node_attributes(G, 'community').values())
                        col2.metric("社区数量", len(communities))
                        col3.metric("平均每社区节点数", round(len(G.nodes()) / max(1, len(communities)), 1))
                    else:
                        col2.metric("第一类节点", len([n for n in G.nodes() if G.nodes[n].get('group', 0) == 1]))
                        col3.metric("第二类节点", len([n for n in G.nodes() if G.nodes[n].get('group', 0) == 2]))
                
                col4.metric("连接总数", len(G.edges()))
                
                # 密度和平均度
                density = nx.density(G)
                st.metric("网络密度", f"{density:.4f}")
                
                # 找出关键节点
                top_nodes = sorted([(node, degree_centrality[node]) 
                                  for node in G.nodes()], 
                                 key=lambda x: x[1], reverse=True)[:show_top_nodes]
                
                # 显示关键节点
                st.subheader(f"关键节点 (基于中心性，前{show_top_nodes}个)")
                node_df = pd.DataFrame(top_nodes, columns=["节点", "中心性得分"])
                if run_community_detection:
                    node_df["社区"] = [G.nodes[node[0]].get('community', 0) for node in top_nodes]
                st.dataframe(node_df)
                
                # 如果是共现网络且有权重，显示最强连接
                if analysis_mode == "共现网络分析" and any('weight' in G[u][v] for u, v in G.edges()):
                    # 找出权重最高的边
                    weighted_edges = [(u, v, G[u][v].get('weight', 1)) for u, v in G.edges()]
                    top_edges = sorted(weighted_edges, key=lambda x: x[2], reverse=True)[:show_top_nodes]
                    
                    st.subheader(f"最强连接 (前{show_top_nodes}个)")
                    edge_df = pd.DataFrame(top_edges, columns=["节点1", "节点2", "连接强度"])
                    st.dataframe(edge_df)
                
                # 显示社区信息（如果有）
                if run_community_detection and len(communities) > 1:
                    st.subheader("社区分析")
                    
                    # 获取每个社区的节点
                    community_nodes = {}
                    for node, com in partition.items():
                        if com not in community_nodes:
                            community_nodes[com] = []
                        community_nodes[com].append(node)
                    
                    # 显示主要社区
                    for com_id, nodes in sorted(community_nodes.items(), 
                                               key=lambda x: len(x[1]), 
                                               reverse=True)[:5]:  # 只显示前5个最大社区
                        with st.expander(f"社区 {com_id} ({len(nodes)} 个节点)"):
                            # 显示社区中的重要节点
                            com_centrality = {node: degree_centrality[node] for node in nodes}
                            top_com_nodes = sorted(com_centrality.items(), 
                                                 key=lambda x: x[1], 
                                                 reverse=True)[:min(5, len(nodes))]
                            
                            st.write("主要节点:")
                            for node, cent in top_com_nodes:
                                st.write(f"- {node} (中心性: {cent:.4f})")
                
                # 显示网络图
                st.subheader("网络可视化")
                st.components.v1.html(html_data, height=network_height+50)
                
                # 添加下载链接
                st.markdown(get_binary_file_downloader_html(path, '网络图HTML文件'), unsafe_allow_html=True)
                
                # 删除临时文件
                os.unlink(path)
                
                st.success("网络分析完成！您可以通过图形界面探索关系网络。")
        else:
            if analysis_mode == "二分网络分析":
                st.error("请选择源节点列和至少一个目标节点列")
            else:
                st.error("请选择至少一个实体列")

st.sidebar.markdown("""
### 使用说明
1. 上传数据文件（CSV或Excel格式）
2. 选择分析模式
3. 配置相应列和参数
4. 调整网络参数（可选）
5. 点击"构建网络图"按钮
6. 在图中探索关系网络:
   - 可拖拽节点
   - 滚轮缩放
   - 悬停查看详情
""")

# 关于和版权信息
st.sidebar.markdown("---")
st.sidebar.info(
    """
    © 2025 通用网络分析工具  
    版本：1.0.0
    """
)