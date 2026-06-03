# TShark 字段兼容性检测

## 简介

`check_tshark_fields.py` 是一个自动化工具，用于检测 `config.py` 中配置的 tshark 字段是否在本机环境中可用。这在以下场景特别有用：

- **更换机器**：不同版本的 Wireshark/TShark 支持的字段可能不同
- **版本升级**：升级 Wireshark 后检查新字段可用性  
- **问题诊断**：当预处理提取 0 个特征时，快速排查是否是字段不兼容导致
- **字段探索**：导出本机所有可用字段，便于添加新特征

## 快速开始

### 1. 基本检测
```bash
python preprocess/check_tshark_fields.py
```

**输出示例：**
```
✓ TShark 已安装: TShark (Wireshark) 4.6.3
✓ 本机支持 269348 个 tshark 字段
✓ config.py 配置了 72 个字段

======================================================================
TShark 字段兼容性检测报告
======================================================================

 统计信息:
   配置字段总数: 72
   本机支持数量: 72 (100.0%)
   不支持数量:   0 (0.0%)

 所有配置的字段在本机都支持！
======================================================================
```

### 2. 导出所有可用字段
```bash
python preprocess/check_tshark_fields.py --export-all
```

会在 `preprocess/` 目录下生成 `tshark_all_fields.txt`，包含本机所有 26万+ 可用字段。

### 3. 获取字段推荐
```bash
python preprocess/check_tshark_fields.py --recommend
```

会推荐一些常用但未配置的字段（HTTP、DNS、TLS、ICMP 等协议）。

### 4. 自动更新配置
```bash
python preprocess/check_tshark_fields.py --update-config
```

如果发现不兼容的字段，会提示是否自动从 `config.py` 中移除。

## 使用场景

### 场景 1：预处理提取 0 个特征

**症状：**
```
成功提取 0 个数据包特征
训练集样本数远低于预期
```

**解决步骤：**
```bash
# 1. 运行检测
python preprocess/check_tshark_fields.py

# 2. 如果发现不兼容字段，自动修复
python preprocess/check_tshark_fields.py --update-config
```

### 场景 2：更换机器或升级 Wireshark

**步骤：**
```bash
# 1. 检查当前环境
python preprocess/check_tshark_fields.py

# 2. 导出当前环境的所有字段备查
python preprocess/check_tshark_fields.py --export-all

# 3. 如果有不兼容，自动更新
python preprocess/check_tshark_fields.py --update-config
```

### 场景 3：添加新的特征字段

**步骤：**
```bash
# 1. 导出所有可用字段
python preprocess/check_tshark_fields.py --export-all

# 2. 查看特定协议的字段（例如 HTTP）
grep "^http\." preprocess/tshark_all_fields.txt > http_fields.txt

# 3. 手动编辑 config.py，添加需要的字段

# 4. 再次检测确认
python preprocess/check_tshark_fields.py
```

## 命令行参数

| 参数 | 说明 |
|------|------|
| `--update-config` | 自动更新 `config.py`，移除不支持的字段 |
| `--export-all` | 导出所有可用字段到 `tshark_all_fields.txt` |
| `--recommend` | 推荐常用但未配置的字段 |

## 工作原理

1. **读取本机字段**：运行 `tshark -G fields` 获取所有支持的字段
2. **解析配置**：从 `config.py` 的 `TSHARK_FIELDS` 中提取字段列表
3. **交叉比对**：检查每个配置字段是否在本机支持
4. **生成报告**：输出兼容性统计和不兼容字段清单
5. **自动修复**：可选地更新 `config.py` 移除不兼容字段

## 退出码

- `0`：所有字段兼容
- `1`：存在不兼容字段或检测失败

可在脚本中使用：
```bash
if python preprocess/check_tshark_fields.py; then
    echo "字段兼容，可以开始预处理"
    python preprocess/preprocess_dataset.py ...
else
    echo "存在不兼容字段，请修复后再运行"
fi
```

## 技术细节

### 字段提取格式

`tshark -G fields` 输出格式（Tab 分隔）：
```
F  <描述>  <字段名>  <类型>  <基础字段>  <bitmask>  <...>
```

脚本提取第 3 列（字段名）。

### Config.py 解析

使用正则表达式 `"([a-z0-9_\.]+)"` 匹配引号内的字段名，并筛选 `frame.`、`eth.`、`ip.`、`tcp.`、`udp.`、`data.` 开头的字段。
