# EXCEL-Network-Analysis

这是一个交互式网络分析应用，可用于分析各种实体间的关系，包括二分网络和共现网络。

演示：(excel-network-analysis.)[https://excel-network-analysis.streamlit.app/]

## 功能特点

- **多种分析模式**：支持二分网络分析和共现网络分析
- **灵活的数据输入**：支持CSV和Excel格式的数据
- **交互式可视化**：使用Pyvis实现直观、可交互的网络图
- **社区检测**：自动识别网络中的社区结构
- **关键节点识别**：基于中心性指标找出网络中的关键节点
- **网络统计**：提供全面的网络统计指标
- **数据导出**：支持下载分析结果和网络图

## 使用指南

1. 上传数据文件（CSV或Excel格式）
2. 选择适合您数据的分析模式
3. 根据提示配置列映射
4. 调整可视化参数（可选）
5. 点击"构建网络图"
6. 探索生成的网络和分析结果

## 数据格式要求

### 二分网络分析模式

数据应包含：
- 一列代表第一类节点（源节点）
- 一列或多列代表第二类节点（目标节点）
- 可选的链接列（用于提供额外信息）

### 共现网络分析模式

数据可以是以下格式之一：
- 行级格式：每行包含多个实体，同行中的实体被视为共现
- 实体对格式：每行代表两个实体之间的一个关系
- 可选的权重列用于表示关系强度

## 部署

该应用基于Streamlit开发，可以在Streamlit Cloud或自托管环境中运行。

### 本地运行

```bash
pip install -r requirements.txt
streamlit run network_analysis.py
```

### Streamlit Cloud部署

1. Fork这个仓库
2. 在Streamlit Cloud中部署应用
3. 享受在线版本的所有功能

## 依赖

- streamlit
- pandas
- networkx
- pyvis
- openpyxl
- python-louvain (社区检测)

## 许可

© 2025 通用网络分析工具
